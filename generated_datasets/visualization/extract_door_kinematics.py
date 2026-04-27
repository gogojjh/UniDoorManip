"""Extract door kinematics from the UniDoorManip dataset.

Reads bounding_box.json, handle_bounding.json, and mobility.urdf for each asset
and writes kinematics.json into the asset's own directory.

Usage:
    # Single asset (category is case-insensitive)
    python extract_door_kinematics.py --category leverdoor_ccw_pull --asset_id 99650069960003

    # All assets in one category
    python extract_door_kinematics.py --category LeverDoor_ccw_pull

    # All assets across all categories
    python extract_door_kinematics.py --all
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import List

import numpy as np


def _dump_json(obj: dict) -> str:
    """Serialize to JSON with dicts indented and arrays kept on one line."""
    raw = json.dumps(obj, indent=2)
    # Collapse any JSON array that spans multiple lines into a single line
    return re.sub(
        r'\[\s+([^\[\]]*?)\s+\]',
        lambda m: '[' + ', '.join(x.rstrip(',') for x in m.group(1).split()) + ']',
        raw,
        flags=re.DOTALL,
    )


DATASET_ROOT = Path(__file__).resolve().parent.parent
CATEGORIES = [
    "LeverDoor_ccw_pull",
    "LeverDoor_ccw_push",
    "LeverDoor_cw_pull",
    "LeverDoor_cw_push",
    "RoundDoor_ccw_pull",
    "RoundDoor_ccw_push",
    "RoundDoor_cw_pull",
    "RoundDoor_cw_push",
    "Cabinet_ccw_pull",
    "Cabinet_cw_pull",
    "Fridge_ccw_pull",
    "Fridge_cw_pull",
    "Safe",
    "Car",
    "Window_ccw_pull",
    "Window_cw_pull",
]
CATEGORY_MAP = {cat.lower(): cat for cat in CATEGORIES}

HANDLE_TYPE_MAP = {
    "9960": "lever",
    "9961": "dial",
    "9962": "round",
    "9968": "car",
    "9969": "window",
}

OPEN_DIRECTION_MAP = {
    "right-pull-door": "right-pull",
    "left-pull-door": "left-pull",
    "right-push-door": "right-push",
    "left-push-door": "left-push",
}

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else np.zeros_like(v)


def _project_perp(v: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Project v onto the plane perpendicular to axis."""
    a = _normalize(axis)
    return v - np.dot(v, a) * a


def _round_list(v: List[float], decimals: int = 6) -> List[float]:
    return [round(x, decimals) for x in v]


_JOINT_TYPE_MAP = {"revolute": 0, "prismatic": 1}

_KINEMATIC_PRIOR_DEFINITION = (
    "primary=door hinge/slide, secondary=handle/knob; type: 0=revolute, 1=prismatic. "
    "kinematic_prior = ["
    " primary_type(1), primary_axis(3), primary_limit_lower(1), primary_limit_upper(1), primary_close_direction(3),"
    " secondary_type(1), secondary_axis(3), secondary_limit_lower(1), secondary_limit_upper(1), secondary_close_direction(3)"
    " ]"
)


def compute_kinematic_prior(props: dict) -> dict:
    """Compute the kinematic prior in F_hinge coordinate.

    kinematic_prior = [
        primary_type,              # (1,) joint type: 0=revolute, 1=prismatic
        primary_axis,              # (3,) unit vector: rotation/translation axis
        primary_limit_lower,       # (1,) joint lower limit (closed = 0)
        primary_limit_upper,       # (1,) joint upper limit (revolute: rad, prismatic: m)
        primary_close_direction,   # (3,) unit vector: door edge direction when closed
        secondary_type,            # (1,) joint type: 0=revolute, 1=prismatic
        secondary_axis,            # (3,) unit vector: rotation/translation axis
        secondary_limit_lower,     # (1,) joint lower limit (rest = 0)
        secondary_limit_upper,     # (1,) joint upper limit
        secondary_close_direction, # (3,) unit vector: part pointing direction at rest
    ]

    All vectors are in F_hinge frame (x=right, y=up, z=out-of-door),
    origin at hinge joint.
    """
    hinge = props["hinge"]
    handle = props["handle"]

    primary_type = _JOINT_TYPE_MAP.get(hinge["joint_type"], 0)
    primary_axis = _normalize(np.array(hinge["axis"]))
    primary_limit_lower = hinge["lower_limit_rad"]
    primary_limit_upper = hinge["upper_limit_rad"]
    primary_close_dir = _normalize(_project_perp(
        np.array(handle["position_on_door"]), primary_axis))

    secondary_type = _JOINT_TYPE_MAP.get(handle["joint_type"], 0)
    secondary_axis = _normalize(np.array(handle["rotation_axis"]))
    secondary_limit_lower = handle["lower_limit_rad"]
    secondary_limit_upper = handle["upper_limit_rad"]
    secondary_close_dir = _normalize(_project_perp(
        np.array(handle["grasp_point"]), secondary_axis))

    prior = np.concatenate([
        [primary_type],
        primary_axis,
        [primary_limit_lower],
        [primary_limit_upper],
        primary_close_dir,
        [secondary_type],
        secondary_axis,
        [secondary_limit_lower],
        [secondary_limit_upper],
        secondary_close_dir,
    ])

    return {
        "definition": _KINEMATIC_PRIOR_DEFINITION,
        "coordinate_frame": "F_hinge: x=right, y=up, z=out-of-door; origin=hinge_joint",
        "primary_type": primary_type,
        "primary_axis": _round_list(primary_axis.tolist()),
        "primary_limit_lower": round(primary_limit_lower, 6),
        "primary_limit_upper": round(primary_limit_upper, 6),
        "primary_close_direction": _round_list(primary_close_dir.tolist()),
        "secondary_type": secondary_type,
        "secondary_axis": _round_list(secondary_axis.tolist()),
        "secondary_limit_lower": round(secondary_limit_lower, 6),
        "secondary_limit_upper": round(secondary_limit_upper, 6),
        "secondary_close_direction": _round_list(secondary_close_dir.tolist()),
        "vector": _round_list(prior.tolist()),
    }


def _parse_urdf(urdf_path: Path) -> dict:
    """Parse joint and mesh parameters from a URDF file."""
    text = urdf_path.read_text()

    robot_name_match = re.search(r'<robot name="([^"]+)"', text)
    robot_name = robot_name_match.group(1) if robot_name_match else "unknown"

    body_scale_match = re.search(
        r'filename="texture_dae/frame\.dae" scale="([\d.]+)', text
    )
    body_scale = float(body_scale_match.group(1)) if body_scale_match else 1.0

    handle_mesh_match = re.search(
        r'filename="texture_dae/(\d+)\.dae" scale="([\d.]+)', text
    )
    handle_scale = float(handle_mesh_match.group(2)) if handle_mesh_match else 1.0

    hinge = {"type": "revolute", "position": [0.0, 0.0, 0.0], "axis": [0.0, -1.0, 0.0],
             "lower_rad": 0.0, "upper_rad": 0.0}
    joint1_block = re.search(r'<joint name="joint_1".*?</joint>', text, re.DOTALL)
    if joint1_block:
        j1 = joint1_block.group(0)
        m = re.search(r'<joint name="joint_1" type="(\w+)"', j1)
        if m:
            hinge["type"] = m.group(1)
        m = re.search(r'<origin xyz="([\d.\-e]+) ([\d.\-e]+) ([\d.\-e]+)"', j1)
        if m:
            hinge["position"] = [float(m.group(i)) for i in (1, 2, 3)]
        m = re.search(r'<axis xyz="([\d.\-e]+) ([\d.\-e]+) ([\d.\-e]+)"', j1)
        if m:
            hinge["axis"] = [float(m.group(i)) for i in (1, 2, 3)]
        m = re.search(r'lower="([\d.\-e]+)" upper="([\d.\-e]+)"', j1)
        if m:
            hinge["lower_rad"] = float(m.group(1))
            hinge["upper_rad"] = float(m.group(2))

    handle_joint = {"type": "revolute", "position": [0.0, 0.0, 0.0], "axis": [0.0, 0.0, 1.0],
                    "lower_rad": 0.0, "upper_rad": 0.0}
    joint2_block = re.search(r'<joint name="joint_2".*?</joint>', text, re.DOTALL)
    if joint2_block:
        j2 = joint2_block.group(0)
        m = re.search(r'<joint name="joint_2" type="(\w+)"', j2)
        if m:
            handle_joint["type"] = m.group(1)
        m = re.search(r'<origin xyz="([\d.\-e]+) ([\d.\-e]+) ([\d.\-e]+)"', j2)
        if m:
            handle_joint["position"] = [float(m.group(i)) for i in (1, 2, 3)]
        m = re.search(r'<axis xyz="([\d.\-e]+) ([\d.\-e]+) ([\d.\-e]+)"', j2)
        if m:
            handle_joint["axis"] = [float(m.group(i)) for i in (1, 2, 3)]
        m = re.search(r'lower="([\d.\-e]+)" upper="([\d.\-e]+)"', j2)
        if m:
            handle_joint["lower_rad"] = float(m.group(1))
            handle_joint["upper_rad"] = float(m.group(2))

    return {
        "robot_name": robot_name,
        "body_scale": body_scale,
        "handle_scale": handle_scale,
        "hinge": hinge,
        "handle_joint": handle_joint,
    }


def extract_kinematics(asset_id: str, category: str) -> dict:
    """Extract kinematics for a single door asset."""
    asset_dir = DATASET_ROOT / category / asset_id
    if not asset_dir.exists():
        raise FileNotFoundError(f"Asset not found: {asset_dir}")

    body_id = asset_id[:7]
    handle_id = asset_id[7:]
    handle_type = HANDLE_TYPE_MAP.get(handle_id[:4], "unknown")

    with open(asset_dir / "bounding_box.json") as f:
        bbox = json.load(f)
    with open(asset_dir / "handle_bounding.json") as f:
        hbbox = json.load(f)

    urdf_data = _parse_urdf(asset_dir / "mobility.urdf")
    open_direction = OPEN_DIRECTION_MAP.get(urdf_data["robot_name"], urdf_data["robot_name"])

    # Raw bbox values are in F_raw (x=width, y=depth, z=height).
    # Convert to F_world (x=right, y=up, z=out-of-door): swap y<->z and negate depth.
    def _sw(v):
        return [v[0], v[2], -v[1]]

    # After z-negation the raw min/max z may be swapped; take element-wise min/max.
    def _sw_bbox(raw_min: list, raw_max: list):
        a, b = _sw(raw_min), _sw(raw_max)
        return ([min(a[i], b[i]) for i in range(3)],
                [max(a[i], b[i]) for i in range(3)])

    bb_min, bb_max = _sw_bbox(bbox["min"], bbox["max"])
    hb_min, hb_max = _sw_bbox(hbbox["handle_min"], hbbox["handle_max"])

    # URDF joint positions and axes are already in F_world (x=right, y=up, z=out-of-door).
    hinge = urdf_data["hinge"]
    handle_joint = urdf_data["handle_joint"]

    props = {
        "asset_id": asset_id,
        "body_id": body_id,
        "handle_id": handle_id,
        "door_type": category,
        "open_direction": open_direction,

        # All values are expressed in F_world: x=right, y=up, z=out-of-door (right-hand).
        # Origins: F_world/F_hinge at door body center/hinge; F_handle at handle joint.
        "coordinate_frame": "F_world: x=right, y=up, z=out-of-door",

        "door_body": {
            "mesh_scale": urdf_data["body_scale"],
            "width_m":  round(bb_max[0] - bb_min[0], 6),
            "height_m": round(bb_max[1] - bb_min[1], 6),
            "depth_m":  round(bb_max[2] - bb_min[2], 6),
            "bounding_box": {"min": bb_min, "max": bb_max},
        },

        "hinge": {
            "coordinate_frame": "F_world: origin=door_center, x=right, y=up, z=out-of-door",
            "joint_type": hinge["type"],
            "position": hinge["position"],
            "axis": hinge["axis"],
            "lower_limit_rad": hinge["lower_rad"],
            "upper_limit_rad": hinge["upper_rad"],
            "lower_limit_deg": round(math.degrees(hinge["lower_rad"]), 2),
            "upper_limit_deg": round(math.degrees(hinge["upper_rad"]), 2),
        },

        "handle": {
            "coordinate_frame": "position_on_door+rotation_axis: F_hinge (origin=hinge_joint, axes=F_world); bounding_box+grasp_point: F_handle (origin=handle_joint, axes=F_world)",
            "type": handle_type,
            "joint_type": handle_joint["type"],
            "mesh_scale": urdf_data["handle_scale"],
            "position_on_door": handle_joint["position"],
            "rotation_axis": handle_joint["axis"],
            "lower_limit_rad": handle_joint["lower_rad"],
            "upper_limit_rad": handle_joint["upper_rad"],
            "upper_limit_deg": round(math.degrees(handle_joint["upper_rad"]), 2),
            "width_m":  round(hb_max[0] - hb_min[0], 6),
            "height_m": round(hb_max[1] - hb_min[1], 6),
            "depth_m":  round(hb_max[2] - hb_min[2], 6),
            "bounding_box": {"min": hb_min, "max": hb_max},
            "grasp_point": hbbox["goal_pos"],
        },
    }

    if "lock_min" in hbbox:
        lk_min, lk_max = _sw_bbox(hbbox["lock_min"], hbbox["lock_max"])
        props["lock"] = {
            "bounding_box": {"min": lk_min, "max": lk_max}
        }

    props["kinematic_prior"] = compute_kinematic_prior(props)

    return props


def process_asset(asset_id: str, category: str) -> Path:
    """Extract and save kinematics.json into the asset directory."""
    props = extract_kinematics(asset_id, category)
    out_path = DATASET_ROOT / category / asset_id / "kinematics.json"
    out_path.write_text(_dump_json(props))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract UniDoorManip door kinematics into kinematics.json."
    )
    parser.add_argument(
        "--category",
        type=str,
        help="Category name, case-insensitive (e.g. leverdoor, LeverDoor). "
             "Required unless --all is set.",
    )
    parser.add_argument(
        "--asset_id",
        type=str,
        help="Single asset ID (e.g. 99650069960003). Requires --category.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all assets across all categories.",
    )
    args = parser.parse_args()

    if not args.all and args.category is None:
        parser.error("Provide --category <name> or --all.")

    if args.all and args.asset_id:
        parser.error("--all and --asset_id are mutually exclusive.")

    # Resolve category to canonical name
    if args.category is not None:
        category = CATEGORY_MAP.get(args.category.lower())
        if category is None:
            parser.error(
                f"Unknown category '{args.category}'. "
                f"Valid values: {list(CATEGORY_MAP)}"
            )

    if args.asset_id:
        out_path = process_asset(args.asset_id, category)
        print(f"Saved: {out_path}")
        return

    categories = CATEGORIES if args.all else [category]
    count = 0
    for cat in categories:
        cat_dir = DATASET_ROOT / cat
        if not cat_dir.exists():
            continue
        for asset_dir in sorted(cat_dir.iterdir()):
            if not asset_dir.is_dir():
                continue
            try:
                out_path = process_asset(asset_dir.name, cat)
                print(f"Saved: {out_path}")
                count += 1
            except Exception as e:
                print(f"SKIP {cat}/{asset_dir.name}: {e}")
    print(f"\nDone. Processed {count} assets.")


if __name__ == "__main__":
    main()
