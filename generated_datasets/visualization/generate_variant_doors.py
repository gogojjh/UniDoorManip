"""Generate derived door categories by cloning assets and modifying URDF joint limits.

Supports:
  - push variants: negate hinge joint limits and rename pull robot to push
  - cw variants: negate handle joint limits (clockwise handle direction)

Usage examples:
    python generate_variant_doors.py --src-category LeverDoor_ccw_pull --dst-category LeverDoor_ccw_push --variant push
    python generate_variant_doors.py --src-category LeverDoor_ccw_pull --dst-category LeverDoor_cw_pull --variant cw
    python generate_variant_doors.py --src-category LeverDoor_cw_pull --dst-category LeverDoor_cw_push --variant push
    python generate_variant_doors.py --src-category RoundDoor_ccw_pull --dst-category RoundDoor_ccw_push --variant push
    python generate_variant_doors.py --src-category RoundDoor_ccw_pull --dst-category RoundDoor_cw_pull --variant cw
    python generate_variant_doors.py --src-category RoundDoor_cw_pull --dst-category RoundDoor_cw_push --variant push
    python generate_variant_doors.py --src-category Cabinet_ccw_pull --dst-category Cabinet_cw_pull --variant cw
    python generate_variant_doors.py --src-category Fridge_ccw_pull --dst-category Fridge_cw_pull --variant cw
    python generate_variant_doors.py --src-category Window_ccw_pull --dst-category Window_cw_pull --variant cw
"""

import argparse
import json
import re
import shutil
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parent.parent

PULL_TO_PUSH_NAME = {
    "right-pull-door": "right-push-door",
    "left-pull-door": "left-push-door",
}


def _negate_joint_limits(urdf_text: str, joint_name: str) -> str:
    """Negate a revolute joint limit block: [lower, upper] -> [-upper, -lower]."""

    def _negate_joint(match: re.Match) -> str:
        joint_block = match.group(0)

        def _flip(limit_match: re.Match) -> str:
            lower = float(limit_match.group(1))
            upper = float(limit_match.group(2))
            return f'lower="{-upper}" upper="{-lower}"'

        return re.sub(
            r'lower="([\d.\-e]+)" upper="([\d.\-e]+)"',
            _flip,
            joint_block,
            count=1,
        )

    return re.sub(
        rf'<joint name="{joint_name}".*?</joint>',
        _negate_joint,
        urdf_text,
        flags=re.DOTALL,
    )


def convert_urdf(urdf_text: str, variant: str) -> str:
    """Apply a variant transform to a URDF string."""
    if variant == "push":
        for old_name, new_name in PULL_TO_PUSH_NAME.items():
            urdf_text = urdf_text.replace(f'name="{old_name}"', f'name="{new_name}"')
        return _negate_joint_limits(urdf_text, "joint_1")
    if variant in ("cw", "reverse"):
        return _negate_joint_limits(urdf_text, "joint_2")
    raise ValueError(f"Unsupported variant: {variant}")


def copy_split_file(src_category: str, dst_category: str) -> None:
    src_split = DATASET_ROOT / src_category / "train_val_test_split.json"
    if not src_split.exists():
        return
    dst_split = DATASET_ROOT / dst_category / "train_val_test_split.json"
    split_data = json.loads(src_split.read_text())
    dst_split.write_text(json.dumps(split_data, indent=2) + "\n")


def regenerate_kinematics(dst_category: str) -> int:
    from extract_door_kinematics import CATEGORIES, CATEGORY_MAP, process_asset

    if dst_category not in CATEGORIES:
        CATEGORIES.append(dst_category)
        CATEGORY_MAP[dst_category.lower()] = dst_category

    count = 0
    for asset_dir in sorted((DATASET_ROOT / dst_category).iterdir()):
        if not asset_dir.is_dir():
            continue
        try:
            out_path = process_asset(asset_dir.name, dst_category)
            print(f"  Saved: {out_path}")
            count += 1
        except Exception as exc:
            print(f"  SKIP {asset_dir.name}: {exc}")
    return count


def generate_variant(src_category: str, dst_category: str, variant: str, dry_run: bool = False) -> None:
    src_dir = DATASET_ROOT / src_category
    dst_dir = DATASET_ROOT / dst_category

    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    assets = sorted(d.name for d in src_dir.iterdir() if d.is_dir())
    print(f"Found {len(assets)} {src_category} assets to convert into {dst_category}.")

    if dry_run:
        print("[DRY RUN] No files will be written.\n")

    for asset_id in assets:
        src_asset = src_dir / asset_id
        dst_asset = dst_dir / asset_id

        if dry_run:
            print(f"  Would create: {dst_asset}")
            continue

        if dst_asset.exists():
            shutil.rmtree(dst_asset)
        shutil.copytree(src_asset, dst_asset)

        urdf_path = dst_asset / "mobility.urdf"
        urdf_text = urdf_path.read_text()
        urdf_path.write_text(convert_urdf(urdf_text, variant))
        print(f"  Created: {dst_asset.relative_to(DATASET_ROOT)}")

    if dry_run:
        return

    copy_split_file(src_category, dst_category)
    print(f"\nRegenerating kinematics.json for {dst_category}...")
    count = regenerate_kinematics(dst_category)
    print(f"\nDone. Generated {count} {dst_category} assets with kinematics.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate derived door categories from existing assets.")
    parser.add_argument("--src-category", required=True, help="Source category folder name, e.g. RoundDoor.")
    parser.add_argument("--dst-category", required=True, help="Destination category folder name.")
    parser.add_argument(
        "--variant",
        required=True,
        choices=["push", "cw", "reverse"],
        help="Variant type: push negates hinge limits; cw (or reverse) negates handle limits for clockwise direction.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing files.")
    args = parser.parse_args()
    generate_variant(args.src_category, args.dst_category, args.variant, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
