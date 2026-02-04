#!/usr/bin/env python3
"""Test if all configured assets exist"""
import os
import sys

# Asset examples from door_asset_test.py
ASSET_EXAMPLES = [
    {"id": 0, "name": "Round Door Handle", "category": "RoundDoor", "asset_name": "99660019962014"},
    {"id": 1, "name": "Lever Door Handle", "category": "LeverDoor", "asset_name": "99650069960003"},
    {"id": 2, "name": "Cabinet Door", "category": "Cabinet", "asset_name": "99613029962004"},
    {"id": 3, "name": "Refrigerator Door", "category": "Fridge", "asset_name": "99614019960005"},
    {"id": 4, "name": "Safe Door", "category": "Safe", "asset_name": "99611129961204"},
    {"id": 5, "name": "Car Door", "category": "Car", "asset_name": "99670019968001"},
    {"id": 6, "name": "Window", "category": "Window", "asset_name": "99690049969510"},
]

print("Checking asset availability...\n")
all_exist = True

for asset in ASSET_EXAMPLES:
    asset_path = f"../generated_datasets/{asset['category']}/{asset['asset_name']}/mobility.urdf"
    exists = os.path.exists(asset_path)
    status = "✓" if exists else "✗"

    print(f"{status} [{asset['id']}] {asset['name']:25s} - {asset_path}")

    if not exists:
        all_exist = False
        # Try to find alternative assets
        category_path = f"../generated_datasets/{asset['category']}"
        if os.path.exists(category_path):
            alternatives = os.listdir(category_path)[:3]
            if alternatives:
                print(f"    Available alternatives: {', '.join(alternatives)}")

print()
if all_exist:
    print("✓ All assets are available!")
    sys.exit(0)
else:
    print("✗ Some assets are missing. Update ASSET_EXAMPLES in door_asset_test.py")
    sys.exit(1)
