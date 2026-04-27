import argparse
from generate_variant_doors import generate_variant

SRC_CATEGORY = "LeverDoor_ccw_pull"
DST_CATEGORY = "LeverDoor_cw_pull"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LeverDoor_cw_pull from LeverDoor_ccw_pull assets.")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing files.")
    args = parser.parse_args()
    generate_variant(SRC_CATEGORY, DST_CATEGORY, "reverse", dry_run=args.dry_run)


if __name__ == "__main__":
    main()
