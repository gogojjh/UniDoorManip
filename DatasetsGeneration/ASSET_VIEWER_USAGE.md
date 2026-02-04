# UniDoorManip Asset Viewer Usage Guide

## Overview

The `door_asset_test.py` script allows you to visualize different types of door/cabinet/window assets from the generated datasets using IsaacGym.

## Prerequisites

- IsaacGym installed and configured
- Generated asset datasets in `../generated_datasets/`
- Display environment for visualization

## Basic Usage

### 1. List All Available Assets

```bash
python door_asset_test.py --list
```

This will show all available asset examples:

```
======================================================================
Available Asset Examples:
======================================================================
  [0] Round Door Handle         - Door with round rotating handle
  [1] Lever Door Handle         - Door with lever-style handle
  [2] Cabinet Door              - Kitchen/office cabinet door
  [3] Refrigerator Door         - Refrigerator door
  [4] Safe Door                 - Safe box with combination lock
  [5] Car Door                  - Vehicle door
  [6] Window                    - Sliding or rotating window
======================================================================
```

### 2. Visualize a Specific Asset

```bash
python door_asset_test.py --example_id <ID>
```

**Examples:**

```bash
# View round door handle (default)
python door_asset_test.py --example_id 0

# View lever door handle
python door_asset_test.py --example_id 1

# View cabinet door
python door_asset_test.py --example_id 2

# View refrigerator door
python door_asset_test.py --example_id 3

# View safe door
python door_asset_test.py --example_id 4

# View car door
python door_asset_test.py --example_id 5

# View window
python door_asset_test.py --example_id 6
```

### 3. Default Behavior

If no `--example_id` is specified, it defaults to example 0 (Round Door Handle):

```bash
python door_asset_test.py
```

## Viewer Controls

Once the viewer window opens:

- **Mouse Left Button + Drag**: Rotate camera
- **Mouse Right Button + Drag**: Pan camera
- **Mouse Wheel**: Zoom in/out
- **ESC**: Exit viewer

## Adding Custom Assets

To add your own asset examples, edit the `ASSET_EXAMPLES` list in `door_asset_test.py`:

```python
ASSET_EXAMPLES = [
    {
        "id": 7,  # Unique ID
        "name": "My Custom Door",
        "category": "RoundDoor",  # Category folder name
        "asset_name": "99999999999999",  # Asset folder name
        "description": "Description of my custom door"
    },
    # ... more assets
]
```

## Troubleshooting

### Asset Not Found Error

If you see an error like:
```
Error: Asset not found at ../generated_datasets/RoundDoor/xxx/mobility.urdf
```

**Solutions:**
1. Check if the asset exists: `ls ../generated_datasets/RoundDoor/`
2. Run the test script: `python test_assets.py`
3. Update the asset name in `ASSET_EXAMPLES` to match existing assets

### Display Not Available

If you're running on a headless server without display:
- Use X11 forwarding: `ssh -X user@server`
- Use VNC or other remote desktop solution
- Use virtual display with xvfb (advanced)

### Invalid example_id

Make sure the ID you specify exists in the list. Use `--list` to see all valid IDs.

## Advanced: Animate Joints

To enable joint animation, uncomment the animation code in the simulation loop (lines 267-271):

```python
# Optional: Animate joints (uncomment to enable)
if num_dofs > 0 and frame_count % 60 == 0:
    # Slowly open/close joints
    angle = math.sin(frame_count * 0.01) * 0.5 + 0.5  # Range [0, 1]
    for dof_name, dof_idx in dof_dict.items():
        gym.set_dof_target_position(env, gym.find_actor_dof_handle(env, actor, dof_name), angle)
```

This will make doors/handles open and close automatically in a sinusoidal motion.

## Testing Assets

Verify all configured assets exist:

```bash
python test_assets.py
```

This script checks if all assets in `ASSET_EXAMPLES` are available and suggests alternatives if any are missing.

## Additional IsaacGym Arguments

The script supports standard IsaacGym arguments:

```bash
# Use PhysX physics engine
python door_asset_test.py --example_id 0 --physx

# Use Flex physics engine
python door_asset_test.py --example_id 0 --flex

# Set number of threads
python door_asset_test.py --example_id 0 --num_threads 4
```

Run `python door_asset_test.py -h` for full list of arguments.
