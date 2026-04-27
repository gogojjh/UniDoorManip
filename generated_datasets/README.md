# Generated Datasets - UniDoorManip

This directory contains procedurally generated articulated object assets for door manipulation tasks. Each asset is a composition of door body parts and handle parts, converted into URDF format for physics simulation.

## 📊 Dataset Statistics

| Category              | Asset Count | Description                                          |
|-----------------------|-------------|------------------------------------------------------|
| RoundDoor_ccw_pull    | 27          | Round knob counterclockwise + pull door              |
| RoundDoor_ccw_push    | 27          | Round knob counterclockwise + push door              |
| RoundDoor_cw_pull     | 27          | Round knob clockwise + pull door                     |
| RoundDoor_cw_push     | 27          | Round knob clockwise + push door                     |
| LeverDoor_ccw_pull    | 30          | Lever handle counterclockwise + pull door            |
| LeverDoor_ccw_push    | 30          | Lever handle counterclockwise + push door            |
| LeverDoor_cw_pull     | 30          | Lever handle clockwise + pull door                   |
| LeverDoor_cw_push     | 30          | Lever handle clockwise + push door                   |
| Cabinet_ccw_pull      | 14          | Cabinet: handle counterclockwise + pull door         |
| Cabinet_cw_pull       | 14          | Cabinet: handle clockwise + pull door                |
| Fridge_ccw_pull       | 11          | Fridge: handle counterclockwise + pull door          |
| Fridge_cw_pull        | 11          | Fridge: handle clockwise + pull door                 |
| Safe                  | 20          | Safe boxes with combination locks                    |
| Car                   | 10          | Vehicle doors                                        |
| Window_ccw_pull       | 35          | Window: handle counterclockwise + pull               |
| Window_cw_pull        | 35          | Window: handle clockwise + pull                      |
| **Total**             | **346**     | -                                                    |

### Naming Convention

Door variants follow a two-axis scheme: `{Type}_{handle}_{action}`
- **handle**: `ccw` (counterclockwise from front view) or `cw` (clockwise)
- **action**: `pull` (door opens toward viewer) or `push` (door opens away)

## 📁 Directory Structure

```
generated_datasets/
├── README.md                    # This file
├── Cabinet_ccw_pull/           # Cabinet: handle counterclockwise + pull
├── Cabinet_cw_pull/            # Cabinet: handle clockwise + pull
├── Car/                         # Car door assets
├── Fridge_ccw_pull/            # Fridge: handle counterclockwise + pull
├── Fridge_cw_pull/             # Fridge: handle clockwise + pull
├── LeverDoor_ccw_pull/         # Lever: counterclockwise + pull
├── LeverDoor_ccw_push/          # Lever: counterclockwise + push
├── LeverDoor_cw_pull/           # Lever: clockwise + pull
├── LeverDoor_cw_push/           # Lever: clockwise + push
├── RoundDoor_ccw_pull/          # Round: counterclockwise + pull
├── RoundDoor_ccw_push/          # Round: counterclockwise + push
├── RoundDoor_cw_pull/           # Round: clockwise + pull
├── RoundDoor_cw_push/           # Round: clockwise + push
├── Safe/                        # Safe door assets
├── Window_ccw_pull/            # Window: handle counterclockwise + pull
└── Window_cw_pull/             # Window: handle clockwise + pull
```

## 🗂️ Asset Structure

Each asset folder follows this naming convention: `[BodyID][HandleID]`

For example: `99660019962014`
- `9966001` - Body part ID
- `9962014` - Handle part ID

### Single Asset Directory

```
99660019962014/
├── mobility.urdf              # URDF robot description file
├── bounding_box.json          # Door body bounding box
├── handle_bounding.json       # Handle bounding box and grasp point
└── texture_dae/               # 3D models and textures
    ├── frame.dae             # Door frame mesh
    ├── board.dae             # Door board/panel mesh
    ├── [handle_id].dae       # Handle mesh (e.g., 9962014.dae)
    ├── [handle_id]-handle-*.dae  # Optional: separate handle parts
    ├── [handle_id]-lock-*.dae    # Optional: lock mechanism
    └── *.jpg                 # Texture images
```

## 📄 File Formats

### 1. `mobility.urdf`

URDF (Unified Robot Description Format) file describing the kinematic structure of the articulated object.

**Key Components:**
- **Links**: Rigid body parts (frame, board, handle, lock)
- **Joints**: Articulated connections between links
  - `joint_0`: Fixed joint connecting base to frame
  - `joint_1`: Revolute joint for door rotation (0° to ~86°)
  - `joint_2`: Revolute joint for handle rotation (if applicable)
  - `joint_3`: Fixed/revolute joint for lock mechanism (if exists)

**Example Structure:**
```xml
<robot name="right-pull-door">
  <link name="base"/>
  <link name="link_0">  <!-- Frame -->
  <link name="link_1">  <!-- Door board -->
  <link name="link_2">  <!-- Handle -->
  <joint name="joint_1" type="revolute">  <!-- Door hinge -->
    <limit lower="0" upper="1.5079644737231006"/>  <!-- ~86 degrees -->
  </joint>
</robot>
```

### 2. `bounding_box.json`

Defines the 3D bounding box of the door body after scaling.

**Format:**
```json
{
  "min": [x_min, y_min, z_min],  // Minimum corner coordinates (meters)
  "max": [x_max, y_max, z_max]   // Maximum corner coordinates (meters)
}
```

**Example:**
```json
{
  "min": [-0.387, -0.018, -0.881],
  "max": [0.387, 0.019, 0.904]
}
```

**Usage:**
- Collision detection
- Object placement (e.g., `initial_height = -min[2] + 0.1`)
- Workspace planning

### 3. `handle_bounding.json`

Defines the handle's bounding box and target grasp position after scaling.

**Format:**
```json
{
  "handle_min": [x, y, z],    // Handle minimum bounds (meters)
  "handle_max": [x, y, z],    // Handle maximum bounds (meters)
  "goal_pos": [x, y, z],      // Target grasp position in local frame (meters)
  "lock_min": [x, y, z],      // Optional: lock minimum bounds
  "lock_max": [x, y, z]       // Optional: lock maximum bounds
}
```

**Example:**
```json
{
  "handle_min": [-0.032, -0.061, -0.028],
  "handle_max": [0.032, -0.0002, 0.028],
  "goal_pos": [-0.0001, 0.0, 0.046]
}
```

**Usage:**
- `goal_pos`: Target end-effector position for grasping (in handle's local frame)
- Transform to world frame: `world_pos = quat_apply(handle_rotation, goal_pos) + handle_position`
- Grasp planning and trajectory generation

### 4. `texture_dae/` Directory

Contains COLLADA (.dae) 3D mesh files and texture images.

**Common Files:**
- `frame.dae`: Door frame/wall mount structure
- `board.dae`: Main door panel/board
- `[handle_id].dae`: Handle mesh (single-piece handles)
- `[handle_id]-handle-*.dae`: Multi-part handle (separate components)
- `[handle_id]-lock-*.dae`: Lock mechanism (if present)
- `*.jpg`: Texture maps for materials

## 🎯 Coordinate System

All assets use the following coordinate system:

```
        Z (up)
        |
        |
        +---- Y
       /
      /
     X
```

- **X-axis**: Left-right (door width direction)
- **Y-axis**: Front-back (depth/door opening direction)
- **Z-axis**: Up-down (height)

**Door Rotation:**
- Hinge typically along the Z-axis at door edge
- Positive rotation opens the door outward
- Joint limits: 0 to ~1.508 radians (0° to 86°)

## 🔧 Generation Parameters

Assets are generated with randomized scales within these ranges:

| Parameter           | Default  | Range            | Description           |
|---------------------|----------|------------------|-----------------------|
| Door Height         | 1.8m     | 1.7m - 1.9m      | Total door height     |
| Handle Length       | 0.064m   | 0.060m - 0.068m  | Handle size (round)   |
|                     | 0.12m    | 0.10m - 0.14m    | Handle size (lever)   |
| Joint Stiffness     | 5000.0   | -                | Position control      |
| Joint Damping       | 100.0    | -                | Velocity damping      |

## 🚀 Usage Examples

### Loading an Asset in IsaacGym

```python
from isaacgym import gymapi
import json

gym = gymapi.acquire_gym()
sim = gym.create_sim(...)

# Load asset
asset_root = "/path/to/generated_datasets/RoundDoor"
asset_file = "99660019962014/mobility.urdf"

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.collapse_fixed_joints = True
asset_options.vhacd_enabled = True

asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# Get bounding box for positioning
with open(f"{asset_root}/{asset_file.split('/')[0]}/bounding_box.json") as f:
    bbox = json.load(f)
    height = -bbox["min"][2] + 0.1  # Place 10cm above ground

# Create actor
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0, height)
actor = gym.create_actor(env, asset, pose, "door", 0, 1)
```

### Getting Grasp Target Position

```python
import json
import torch

# Load handle bounding info
with open("RoundDoor/99660019962014/handle_bounding.json") as f:
    handle_info = json.load(f)
    local_goal = torch.tensor(handle_info["goal_pos"])

# Get handle world transform
handle_state = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL)
handle_pos = handle_state["pose"]["p"][2]  # Assuming handle is link 2
handle_rot = handle_state["pose"]["r"][2]

# Transform to world frame
def quat_apply(quat, vec):
    # Quaternion rotation (see door_asset_test.py for implementation)
    ...

world_goal = quat_apply(handle_rot, local_goal) + handle_pos
```

### Visualizing Assets

Use the provided visualization tool:

```bash
# List all assets
python door_asset_test.py --list

# Visualize specific asset type
python door_asset_test.py --example_id 0  # Round door
python door_asset_test.py --example_id 2  # Cabinet
python door_asset_test.py --example_id 4  # Safe
```

## 📝 Naming Convention

### Body Part IDs
Body IDs are 7-digit numbers from the source 3D model database:
- `9965xxx`: Standard door bodies
- `9961xxx`: Cabinet bodies
- `9967xxx`: Car door bodies
- `9969xxx`: Window frames

### Handle Part IDs
Handle IDs are 7-digit numbers from the handle database:
- `9962xxx`: Round handles
- `9960xxx`: Lever handles
- `9968xxx`: Car door handles
- `9969xxx`: Window handles
- `9961xxx`: Safe combination locks

### Combined Asset Name
Format: `[Body_ID][Handle_ID]`
- Example: `99660019962014`
  - Body: `9966001` (standard door)
  - Handle: `9962014` (round handle)

## 🔬 Physical Properties

All assets include:
- **Visual Geometry**: High-resolution meshes with textures
- **Collision Geometry**: Simplified convex decomposition (VHACD)
- **Joint Limits**: Realistic angular constraints
- **Inertial Properties**: Automatically computed from meshes
- **Material Properties**: Friction and restitution coefficients

## 🛠️ Extending the Dataset

To generate additional assets:

1. Add new body/handle parts to `DatasetsGeneration/assets/`
2. Edit generation scripts in `DatasetsGeneration/`
3. Run generation script:
   ```bash
   cd DatasetsGeneration
   bash scripts/generate_round_door_datasets.sh
   ```
4. Generated assets will appear in this directory

See `DatasetsGeneration/README.md` for detailed generation instructions.

## 📚 Related Files

- `../DatasetsGeneration/door_asset_test.py` - Asset visualization tool
- `../DatasetsGeneration/README.md` - Dataset generation guide
- `../DatasetsGeneration/ASSET_VIEWER_USAGE.md` - Viewer documentation

## 📖 References

- [URDF Specification](http://wiki.ros.org/urdf/XML)
- [COLLADA Format](https://www.khronos.org/collada/)
- [IsaacGym Documentation](https://developer.nvidia.com/isaac-gym)

## 📄 License

Copyright (c) 2023-2024 UniDoorManip Project
See main repository LICENSE file for details.

---

**Note**: This dataset was procedurally generated by combining modular door body and handle components. Each asset includes realistic physics properties for simulation in IsaacGym or other robotics simulators.
