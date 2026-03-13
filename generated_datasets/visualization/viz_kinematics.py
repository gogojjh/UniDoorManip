"""Visualize door kinematics from kinematics.json in F_world coordinates.

F_world / F_hinge: x=right, y=up, z=out-of-door.
Origins: F_world at door body center, F_hinge at hinge joint, F_handle at handle joint.

For plotting, F_world axes are mapped to matplotlib axes as:
  mpl_X = F_world X (right),  mpl_Y = F_world Z (out),  mpl_Z = F_world Y (up)
so that the door stands upright with height on the vertical axis.

Drawn elements:
  - Blue wireframe:   door body bounding box
  - Orange wireframe: handle bounding box
  - Red star:         grasp point
  - RGB triads at F_world, F_hinge, F_handle origins
  - Cyan arrow:       primary (hinge) axis with limit annotation
  - Darkcyan arrow:   primary close direction
  - Magenta arrow:    secondary (handle) axis with limit annotation
  - Darkmagenta arrow: secondary close direction

Usage:
    python viz_kinematics.py --category leverdoor --asset_id 99650089960001
    python viz_kinematics.py --category LeverDoor --output_dir ./viz_output
    python viz_kinematics.py --all --output_dir ./viz_output
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

DATASET_ROOT = Path(__file__).resolve().parent.parent
CATEGORIES = ["LeverDoor", "RoundDoor", "Cabinet", "Fridge", "Safe", "Car", "Window"]
CATEGORY_MAP = {cat.lower(): cat for cat in CATEGORIES}

_AXIS_COLORS = ["red", "green", "blue"]
_AXIS_LABELS = ["X", "Y", "Z"]


def _p(v):
    """Swap F_world (x,y,z) → matplotlib (x,z,y) so Y(up) maps to mpl vertical."""
    a = np.asarray(v, dtype=float)
    return np.array([a[0], a[2], a[1]])


def _box_edges(mn, mx):
    c = np.array([
        [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
        [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
        [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
    ])
    pairs = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    return [(c[i], c[j]) for i, j in pairs]


def _draw_box(ax, mn, mx, color: str, lw: float = 1.0, label: str = None) -> None:
    """Draw wireframe box. mn/mx are in F_world; swapped for plotting."""
    pmn, pmx = _p(mn), _p(mx)
    for k, (p0, p1) in enumerate(_box_edges(pmn, pmx)):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                color=color, lw=lw, label=label if k == 0 else None)


def _draw_frame(ax, origin: np.ndarray, scale: float, name: str,
                dot_color: str = "black") -> None:
    """Draw labeled RGB coordinate frame triad. origin is in F_world."""
    o = _p(origin)
    ax.scatter(*o, color=dot_color, s=40, zorder=9)
    ax.text(o[0], o[1], o[2] + scale * 0.28, name,
            fontsize=14, fontweight="bold", color="black",
            ha="center", va="bottom", zorder=10)
    # F_world axes [1,0,0], [0,1,0], [0,0,1] → swapped for mpl
    dirs_mpl = [_p([1, 0, 0]), _p([0, 1, 0]), _p([0, 0, 1])]
    for d, color, lbl in zip(dirs_mpl, _AXIS_COLORS, _AXIS_LABELS):
        ax.quiver(*o, *(d * scale), color=color, arrow_length_ratio=0.15)
        tip = o + d * (scale * 1.08)
        ax.text(*tip, lbl, fontsize=12, color="black",
                ha="center", va="center", fontweight="bold", zorder=10)


def _draw_arrow(ax, origin, direction, scale, color, label=None):
    """Draw arrow. origin/direction are in F_world; swapped for plotting."""
    o = _p(origin)
    d = _p(direction)
    if np.linalg.norm(d) < 1e-8:
        return
    ax.quiver(*o, *(d * scale), color=color, arrow_length_ratio=0.15,
              linewidth=2.0, label=label)


def visualize(props: dict, output_path: Path = None) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    db = props["door_body"]
    hinge = props["hinge"]
    handle = props["handle"]
    kp = props["kinematic_prior"]

    door_h = db["height_m"]
    ax_scale = door_h * 0.11
    arrow_scale = door_h * 0.25

    # Positions in F_world (origin=door_center)
    hinge_pos = np.array(hinge["position"])
    handle_pos_on_door = np.array(handle["position_on_door"])
    handle_pos_world = hinge_pos + handle_pos_on_door
    grasp_world = handle_pos_world + np.array(handle["grasp_point"])

    # Door body bbox (F_world, origin=door_center)
    db_bbox = db["bounding_box"]
    _draw_box(ax, np.array(db_bbox["min"]), np.array(db_bbox["max"]),
              color="steelblue", lw=1.2, label="Door body")

    # Handle bbox (F_handle → shift to F_world)
    hb_bbox = handle["bounding_box"]
    hb_min = np.array(hb_bbox["min"]) + handle_pos_world
    hb_max = np.array(hb_bbox["max"]) + handle_pos_world
    _draw_box(ax, hb_min, hb_max, color="darkorange", lw=1.5, label="Handle")

    # Grasp point
    gp = _p(grasp_world)
    ax.scatter(*gp, color="red", s=120, marker="*", zorder=6, label="Grasp point")

    # Coordinate frame triads
    _draw_frame(ax, np.zeros(3), ax_scale, "F_world", dot_color="black")
    _draw_frame(ax, hinge_pos, ax_scale, "F_hinge", dot_color="purple")
    _draw_frame(ax, handle_pos_world, ax_scale, "F_handle", dot_color="darkorange")

    # Primary axis (hinge) — orange
    primary_axis = np.array(kp["primary_axis"])
    primary_lo = kp["primary_limit_lower"]
    primary_hi = kp["primary_limit_upper"]
    _draw_arrow(ax, hinge_pos, primary_axis, arrow_scale,
                color="orange", label="Primary axis")
    tip = _p(hinge_pos + primary_axis * arrow_scale * 1.12)
    ax.text(*tip, f"{math.degrees(primary_lo):.0f}-{math.degrees(primary_hi):.0f}\u00b0",
            fontsize=9, color="orange", fontweight="bold", ha="center", va="bottom", zorder=10)

    # Primary close direction — darkorange
    primary_close = np.array(kp["primary_close_direction"])
    _draw_arrow(ax, hinge_pos, primary_close, arrow_scale * 0.6, color="darkorange",
                label="Primary close dir")

    # Secondary axis (handle) — magenta
    secondary_axis = np.array(kp["secondary_axis"])
    secondary_lo = kp["secondary_limit_lower"]
    secondary_hi = kp["secondary_limit_upper"]
    _draw_arrow(ax, handle_pos_world, secondary_axis, arrow_scale * 0.7,
                color="magenta", label="Secondary axis")
    tip2 = _p(handle_pos_world + secondary_axis * arrow_scale * 0.78)
    ax.text(*tip2, f"{math.degrees(secondary_lo):.0f}-{math.degrees(secondary_hi):.0f}\u00b0",
            fontsize=9, color="magenta", fontweight="bold", ha="center", va="bottom", zorder=10)

    # Secondary close direction — darkmagenta
    secondary_close = np.array(kp["secondary_close_direction"])
    _draw_arrow(ax, handle_pos_world, secondary_close, arrow_scale * 0.5,
                color="darkmagenta", label="Secondary close dir")

    # Legend
    handles_legend, _ = ax.get_legend_handles_labels()
    for color, lbl in zip(_AXIS_COLORS, ["X-axis", "Y-axis", "Z-axis"]):
        handles_legend.append(Line2D([0], [0], color=color, lw=2, label=lbl))
    ax.legend(handles=handles_legend, fontsize=7, loc="upper left",
              borderpad=0.3, labelspacing=0.2, handlelength=1.2, handletextpad=0.4)

    # Axis labels (mpl_X=right, mpl_Y=out, mpl_Z=up)
    ax.set_xlabel("X  right (m)")
    ax.set_ylabel("Z  out-of-door (m)")
    ax.set_zlabel("Y  up (m)")

    # prior_str = ", ".join(f"{v:.2f}" for v in kp["vector"])
    ax.set_title(
        f"{props['door_type']}  |  {props['asset_id']}\n"
        f"open: {props['open_direction']}  |  handle: {handle['type']}"
        f"  |  WxHxD: {db['width_m']:.2f}x{db['height_m']:.2f}x{db['depth_m']:.2f} m\n"
        f"In F_world (x=right, y=up, z=out-of-door; origin=door_center)",
        fontsize=8,
    )
    # mpl axes: X=width, Y=depth, Z=height → door stands tall
    ax.set_box_aspect([db["width_m"], db["depth_m"], db["height_m"]])
    ax.invert_xaxis()
    elev, azim, roll = ax.elev, ax.azim, ax.roll
    ax.view_init(elev=elev, azim=azim - 180, roll=roll)

    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=120)
        plt.close(fig)
    else:
        plt.show()


def process_asset(asset_id: str, category: str, output_dir: Path = None) -> Path:
    kin_path = DATASET_ROOT / category / asset_id / "kinematics.json"
    if not kin_path.exists():
        raise FileNotFoundError(f"kinematics.json not found: {kin_path}")
    with open(kin_path) as f:
        props = json.load(f)
    out_path = (output_dir / category / f"{asset_id}.png") if output_dir else None
    visualize(props, output_path=out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize UniDoorManip door kinematics in F_world coordinates."
    )
    parser.add_argument("--category", type=str,
        help="Category name, case-insensitive (e.g. leverdoor). Required unless --all.")
    parser.add_argument("--asset_id", type=str,
        help="Single asset ID. Requires --category.")
    parser.add_argument("--all", action="store_true", help="Process all assets.")
    parser.add_argument("--output_dir", type=str, default=None,
        help="Directory to save PNG figures. Required for batch mode.")
    args = parser.parse_args()

    if not args.all and args.category is None:
        parser.error("Provide --category <name> or --all.")
    if args.all and args.asset_id:
        parser.error("--all and --asset_id are mutually exclusive.")

    is_batch = args.all or (args.category and not args.asset_id)
    output_dir = Path(args.output_dir) if args.output_dir else None
    if is_batch and output_dir is None:
        parser.error("Batch mode requires --output_dir.")

    if args.category:
        category = CATEGORY_MAP.get(args.category.lower())
        if category is None:
            parser.error(f"Unknown category '{args.category}'. Valid: {list(CATEGORY_MAP)}")

    if args.asset_id:
        out = process_asset(args.asset_id, category, output_dir)
        if out:
            print(f"Saved: {out}")
        return

    categories = CATEGORIES if args.all else [category]
    count = 0
    for cat in categories:
        cat_dir = DATASET_ROOT / cat
        if not cat_dir.is_dir():
            continue
        for asset_dir in sorted(cat_dir.iterdir()):
            if not asset_dir.is_dir():
                continue
            try:
                out = process_asset(asset_dir.name, cat, output_dir)
                print(f"Saved: {out}")
                count += 1
            except Exception as e:
                print(f"SKIP {cat}/{asset_dir.name}: {e}")
    print(f"\nDone. Visualized {count} assets.")


if __name__ == "__main__":
    main()
