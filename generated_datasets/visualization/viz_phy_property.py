"""Visualize physical properties of UniDoorManip door assets.

All values in physical_property.json are in F_world: x=right, y=up, z=out-of-door.
Matplotlib display: X=right, Y=into-door (= −Z), Z=up (= Y).

Drawn elements:
  - Blue wireframe:   door body bounding box
  - Orange wireframe: handle bounding box
  - Red star:         grasp point
  - Labeled RGB triads (X=red, Y=green, Z=blue) at:
      F_world  – door body center (world origin)
      F_hinge  – hinge joint position (F_world orientation)
      F_handle – handle joint origin (F_door/F_world orientation at zero pose)

Usage:
    python viz_phy_property.py --category leverdoor --asset_id 99650089960001
    python viz_phy_property.py --category LeverDoor --output_dir ./viz_output
    python viz_phy_property.py --all --output_dir ./viz_output
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

DATASET_ROOT = Path(__file__).resolve().parent.parent
CATEGORIES = ["LeverDoor", "RoundDoor", "Cabinet", "Fridge", "Safe", "Car", "Window"]
CATEGORY_MAP = {cat.lower(): cat for cat in CATEGORIES}

_AXIS_COLORS = ["red", "green", "blue"]
_AXIS_LABELS = ["X", "Y", "Z"]

# F_world (x=right, y=up, z=out-of-door) → matplotlib (X=right, Y=into-door, Z=up)
# Axis mapping: fw_X→mpl_X, fw_Y→mpl_Z, fw_Z→mpl_(-Y)
_FW_AXES_MPL = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)


def _fw(v) -> np.ndarray:
    """Convert F_world (x=right, y=up, z=out-of-door) → matplotlib (X, Y=into-door, Z=up)."""
    v = np.asarray(v, dtype=float)
    return np.array([v[0], -v[2], v[1]])


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

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
    for k, (p0, p1) in enumerate(_box_edges(mn, mx)):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                color=color, lw=lw, label=label if k == 0 else None)


def _draw_frame(ax, origin_mpl: np.ndarray, scale: float, name: str,
                dot_color: str = "black") -> None:
    """Draw labeled F_world coordinate frame (origin already in matplotlib coords).

    X=red (right), Y=green (up = mpl_Z), Z=blue (out-of-door = mpl_−Y).
    """
    o = np.asarray(origin_mpl, dtype=float)
    tip_k = 1.08   # text placed slightly beyond arrow tip

    ax.scatter(*o, color=dot_color, s=40, zorder=9)
    ax.text(o[0], o[1], o[2] + scale * 0.28, name,
            fontsize=14, fontweight="bold", color="black",
            ha="center", va="bottom", zorder=10)

    for d, color, lbl in zip(_FW_AXES_MPL, _AXIS_COLORS, _AXIS_LABELS):
        ax.quiver(*o, *(d * scale), color=color)
        tip = o + d * (scale * tip_k)
        ax.text(*tip, lbl, fontsize=12, color="black",
                ha="center", va="center", fontweight="bold", zorder=10)


# ---------------------------------------------------------------------------
# Main visualisation
# ---------------------------------------------------------------------------

def visualize(props: dict, output_path: Path = None) -> None:
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    db   = props["door_body"]
    door_h = db["height_m"]
    ax_scale = door_h * 0.11

    # ── Door body bounding box ───────────────────────────────────────────────
    bb = db["bounding_box"]
    _draw_box(ax, _fw(bb["min"]), _fw(bb["max"]), color="steelblue", lw=1.2, label="Door body")

    # F_world frame at door body origin (0,0,0)
    _draw_frame(ax, np.zeros(3), ax_scale, name="F_world", dot_color="steelblue")

    # ── Hinge ────────────────────────────────────────────────────────────────
    hinge_pos = np.array(props["hinge"]["position"])
    hinge_pos_mpl = _fw(hinge_pos)

    # F_hinge frame at hinge position (F_world orientation)
    _draw_frame(ax, hinge_pos_mpl, ax_scale * 0.6, name="F_hinge", dot_color="purple")

    # ── Handle ───────────────────────────────────────────────────────────────
    handle_pos = hinge_pos + np.array(props["handle"]["position_on_door"])
    handle_pos_mpl = _fw(handle_pos)

    hb = props["handle"]["bounding_box"]
    _draw_box(ax, _fw(np.array(hb["min"]) + handle_pos),
              _fw(np.array(hb["max"]) + handle_pos),
              color="darkorange", lw=1.5, label="Handle")

    # F_handle frame at handle joint origin (F_door/F_world orientation at zero pose)
    _draw_frame(ax, handle_pos_mpl, ax_scale * 0.5, name="F_handle", dot_color="darkorange")

    # ── Grasp point ──────────────────────────────────────────────────────────
    grasp_pos = handle_pos + np.array(props["handle"]["grasp_point"])
    ax.scatter(*_fw(grasp_pos), color="red", s=120, marker="*", zorder=6, label="Grasp point")

    # ── Legend ───────────────────────────────────────────────────────────────
    axis_legend = [Line2D([0], [0], color=c, lw=1.5, label=f"{l}-axis")
                   for c, l in zip(_AXIS_COLORS, _AXIS_LABELS)]
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + axis_legend, fontsize=7, loc="upper left")

    # ── Labels & aspect ──────────────────────────────────────────────────────
    ax.set_xlabel("X  right (m)")
    ax.set_ylabel("−Z  into-door (m)")
    ax.set_zlabel("Y  up (m)")
    ax.set_title(
        f"{props['door_type']}  ·  {props['asset_id']}\n"
        f"open: {props['open_direction']}  |  handle: {props['handle']['type']}  |  "
        f"W×H×D  {db['width_m']:.3f} × {db['height_m']:.3f} × {db['depth_m']:.3f} m",
        fontsize=9,
    )
    ax.set_box_aspect([db["width_m"], db["depth_m"], db["height_m"]])

    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=120)
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Asset processing & CLI  (unchanged)
# ---------------------------------------------------------------------------

def process_asset(asset_id: str, category: str, output_dir: Path = None) -> Path:
    prop_path = DATASET_ROOT / category / asset_id / "physical_property.json"
    if not prop_path.exists():
        raise FileNotFoundError(f"physical_property.json not found: {prop_path}")
    with open(prop_path) as f:
        props = json.load(f)
    out_path = (output_dir / category / f"{asset_id}.png") if output_dir else None
    visualize(props, output_path=out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize UniDoorManip door physical properties."
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

