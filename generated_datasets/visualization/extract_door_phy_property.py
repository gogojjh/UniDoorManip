"""Extract physical properties of door assets from the UniDoorManip dataset.

Reads bounding_box.json, handle_bounding.json, and mobility.urdf for each asset
and writes physical_property.json into the asset's own directory.

Usage:
    # Single asset (category is case-insensitive)
    python extract_door_phy_property.py --category leverdoor --asset_id 99650069960003

    # All assets in one category
    python extract_door_phy_property.py --category LeverDoor

    # All assets across all categories
    python extract_door_phy_property.py --all
"""

import argparse
import json
import math
import re
from pathlib import Path


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
CATEGORIES = ["LeverDoor", "RoundDoor", "Cabinet", "Fridge", "Safe", "Car", "Window"]
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

    hinge = {"position": [0.0, 0.0, 0.0], "axis": [0.0, -1.0, 0.0],
             "lower_rad": 0.0, "upper_rad": 0.0}
    joint1_block = re.search(r'<joint name="joint_1".*?</joint>', text, re.DOTALL)
    if joint1_block:
        j1 = joint1_block.group(0)
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

    handle_joint = {"position": [0.0, 0.0, 0.0], "axis": [0.0, 0.0, 1.0],
                    "lower_rad": 0.0, "upper_rad": 0.0}
    joint2_block = re.search(r'<joint name="joint_2".*?</joint>', text, re.DOTALL)
    if joint2_block:
        j2 = joint2_block.group(0)
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


def extract_physical_property(asset_id: str, category: str) -> dict:
    """Extract all physical properties for a single door asset."""
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

    # URDF joint origins are expressed with z=into-door; negate z to get F_world z=out-of-door.
    def _neg_z(v: list) -> list:
        return [v[0], v[1], -v[2]]

    hinge = urdf_data["hinge"]
    handle_joint = urdf_data["handle_joint"]
    hinge = {
        "position": _neg_z(hinge["position"]),
        "axis": _neg_z(hinge["axis"]),
        "lower_rad": hinge["lower_rad"],
        "upper_rad": hinge["upper_rad"],
    }
    handle_joint = {
        "position": _neg_z(handle_joint["position"]),
        "axis": _neg_z(handle_joint["axis"]),
        "lower_rad": handle_joint["lower_rad"],
        "upper_rad": handle_joint["upper_rad"],
    }

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
            "position": hinge["position"],
            "axis": hinge["axis"],
            "lower_limit_rad": hinge["lower_rad"],
            "upper_limit_rad": hinge["upper_rad"],
            "upper_limit_deg": round(math.degrees(hinge["upper_rad"]), 2),
        },

        "handle": {
            "coordinate_frame": "position_on_door+rotation_axis: F_hinge (origin=hinge_joint, axes=F_world); bounding_box+grasp_point: F_handle (origin=handle_joint, axes=F_world)",
            "type": handle_type,
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
            "grasp_point": _sw(hbbox["goal_pos"]),
        },
    }

    if "lock_min" in hbbox:
        lk_min, lk_max = _sw_bbox(hbbox["lock_min"], hbbox["lock_max"])
        props["lock"] = {
            "bounding_box": {"min": lk_min, "max": lk_max}
        }

    return props


def process_asset(asset_id: str, category: str) -> Path:
    """Extract and save physical_property.json into the asset directory."""
    props = extract_physical_property(asset_id, category)
    out_path = DATASET_ROOT / category / asset_id / "physical_property.json"
    out_path.write_text(_dump_json(props))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract UniDoorManip door physical properties into physical_property.json."
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
