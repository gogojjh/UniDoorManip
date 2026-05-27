# Dataset Structure Visualization

## Complete Directory Tree Example

```
generated_datasets/
│
├── README.md                              # Dataset documentation
│
├── RoundDoor_ccw_pull/                    # Round knob ccw + pull (27 assets)
│   ├── 99650069962004/
│   │   ├── mobility.urdf                  # URDF description
│   │   ├── kinematics.json               # door_type, joint limits, handle pose
│   │   ├── bounding_box.json              # Body bounds: {"min": [...], "max": [...]}
│   │   ├── handle_bounding.json           # Handle bounds + grasp point
│   │   └── texture_dae/
│   │       ├── frame.dae                  # Door frame 3D model
│   │       ├── board.dae                  # Door panel 3D model
│   │       ├── 9962004.dae                # Handle 3D model
│   │       ├── texture_001.jpg            # Material textures
│   │       └── texture_002.jpg
│   └── ...
│
├── RoundDoor_ccw_push/                    # Round knob ccw + push (27 assets)
├── RoundDoor_cw_pull/                     # Round knob cw  + pull (27 assets)
├── RoundDoor_cw_push/                     # Round knob cw  + push (27 assets)
│
├── LeverDoor_ccw_pull/                    # Lever ccw + pull (30 assets)
│   ├── 99650069960003/
│   │   ├── mobility.urdf
│   │   ├── kinematics.json
│   │   ├── bounding_box.json
│   │   ├── handle_bounding.json
│   │   └── texture_dae/
│   │       ├── frame.dae
│   │       ├── board.dae
│   │       ├── 9960003-handle-right.dae   # Multi-part handle
│   │       ├── 9960003-lock-right.dae     # Separate lock
│   │       └── *.jpg
│   └── ...
│
├── LeverDoor_ccw_push/                    # Lever ccw + push (29 assets)
├── LeverDoor_cw_pull/                     # Lever cw  + pull (29 assets)
├── LeverDoor_cw_push/                     # Lever cw  + push (29 assets)
├── LeverDoor_direct_pull/                 # Lever already unlocked + pull (29 assets)
├── LeverDoor_direct_push/                 # Lever already unlocked + push (29 assets)
├── RoundDoor_direct_pull/                 # Round knob already unlocked + pull (26 assets)
├── RoundDoor_direct_push/                 # Round knob already unlocked + push (26 assets)
│
├── Cabinet_ccw_pull/                      # Cabinet ccw + pull (13 assets)
├── Cabinet_cw_pull/                       # Cabinet cw + pull (13 assets)
├── Cabinet_direct_pull/                   # Cabinet already unlocked + pull (13 assets)
│   ├── 99613029962004/
│   │   ├── mobility.urdf                  # Similar structure
│   │   ├── bounding_box.json
│   │   ├── handle_bounding.json
│   │   └── texture_dae/
│   │       ├── frame.dae
│   │       ├── board.dae
│   │       ├── 9962004.dae
│   │       └── *.jpg
│   └── ...
│
├── Fridge_ccw_pull/                       # Fridge ccw + pull (10 assets)
├── Fridge_cw_pull/                        # Fridge cw + pull (10 assets)
├── Fridge_direct_pull/                    # Fridge already unlocked + pull (10 assets)
│   ├── 99614019960005/
│   │   ├── mobility.urdf
│   │   ├── bounding_box.json
│   │   ├── handle_bounding.json
│   │   └── texture_dae/
│   └── ...
│
├── Safe_ccw_pull/                         # Safe ccw + pull (19 assets)
├── Safe_cw_pull/                          # Safe cw + pull (19 assets)
├── Safe_direct_pull/                      # Safe already unlocked + pull (19 assets)
│   ├── 99611129961204/
│   │   ├── mobility.urdf
│   │   ├── bounding_box.json
│   │   ├── handle_bounding.json           # Includes lock mechanism
│   │   └── texture_dae/
│   │       ├── frame.dae
│   │       ├── board.dae
│   │       ├── 9961204-handle-right.dae   # Dial/combination lock
│   │       └── *.jpg
│   └── ...
│
├── Car_pull/                              # Car doors (9 assets)
│   ├── 99670019968001/
│   │   ├── mobility.urdf
│   │   ├── bounding_box.json
│   │   ├── handle_bounding.json
│   │   └── texture_dae/
│   └── ...
│
├── Window_ccw_pull/                       # Window ccw + pull (34 assets)
├── Window_cw_pull/                        # Window cw + pull (34 assets)
├── Window_direct_pull/                    # Window already unlocked + pull (34 assets)
│   ├── 99690049969510/
│   │   ├── mobility.urdf
│   │   ├── bounding_box.json
│   │   ├── handle_bounding.json
│   │   └── texture_dae/
│   └── ...
│
└── franka_description/                    # Optional: Robot arm model
    └── ...
```

## File Size Reference

Typical asset sizes:

```
Asset Folder:                 ~2-10 MB total
├── mobility.urdf             1-3 KB
├── bounding_box.json         100-200 bytes
├── handle_bounding.json      200-300 bytes
└── texture_dae/
    ├── *.dae files           100-500 KB each
    └── *.jpg files           50-200 KB each
```

## JSON File Examples

### bounding_box.json
```json
{
  "min": [-0.3869287577502149, -0.01828568159404461, -0.8811774264659555],
  "max": [0.38716545844225786, 0.01934393211655931, 0.903539790407748]
}
```
- Units: meters
- Purpose: Physical bounds of entire door assembly
- Used for: Collision detection, placement, workspace planning

### handle_bounding.json (Simple Handle)
```json
{
  "handle_min": [-0.03212854248759632, -0.060963645409754116, -0.02779298686503621],
  "handle_max": [0.03183727375467628, -0.00019611334618363025, 0.027602995230990825],
  "goal_pos": [-0.00014563436650699325, 0.0, 0.04568991310578506]
}
```

### handle_bounding.json (With Lock)
```json
{
  "handle_min": [-0.0771708744756766, -0.04007292462027498, -0.0794845649402083],
  "handle_max": [0.07686467374452814, -0.00016150403395362098, 0.0734520127291316],
  "lock_min": [-0.05, -0.03, -0.06],
  "lock_max": [0.05, -0.001, 0.05],
  "goal_pos": [-0.0001531003654235974, -0.003016276052221656, 0.025117214769124985]
}
```
- `goal_pos`: Target grasp point in handle's local coordinate frame
- `lock_*`: Optional bounding box for lock mechanism

## URDF Structure Example

### Simple Round Handle Door
```xml
<robot name="right-pull-door">
  <!-- Base link (world anchor) -->
  <link name="base"/>

  <!-- Link 0: Door frame (fixed to world) -->
  <link name="link_0">
    <visual name="out-frame">
      <geometry>
        <mesh filename="texture_dae/frame.dae" scale="0.846 0.846 0.846"/>
      </geometry>
    </visual>
    <collision>...</collision>
  </link>

  <!-- Joint 0: Fixed joint base -> frame -->
  <joint name="joint_0" type="fixed">
    <origin rpy="1.571 0 -1.571" xyz="0 0 0"/>
    <parent link="base"/>
    <child link="link_0"/>
  </joint>

  <!-- Link 1: Door board/panel (rotates on hinge) -->
  <link name="link_1">
    <visual name="surf-board">
      <geometry>
        <mesh filename="texture_dae/board.dae" scale="0.846 0.846 0.846"/>
      </geometry>
    </visual>
    <collision>...</collision>
  </link>

  <!-- Joint 1: Revolute hinge (door opening) -->
  <joint name="joint_1" type="revolute">
    <origin xyz="-0.323 0.846 0"/>
    <axis xyz="0 -1 0"/>                    <!-- Rotation axis -->
    <limit lower="0" upper="1.508"/>        <!-- 0° to 86° -->
    <parent link="link_0"/>
    <child link="link_1"/>
  </joint>

  <!-- Link 2: Handle (rotates for grip) -->
  <link name="link_2">
    <visual name="handle">
      <geometry>
        <mesh filename="texture_dae/9962014.dae" scale="1.007 1.007 1.007"/>
      </geometry>
    </visual>
    <collision>...</collision>
  </link>

  <!-- Joint 2: Revolute handle rotation -->
  <joint name="joint_2" type="revolute">
    <origin xyz="0.242 -0.136 0.026"/>      <!-- Handle position on door -->
    <axis xyz="0 0 1"/>                     <!-- Z-axis rotation -->
    <limit lower="0" upper="1.508"/>
    <parent link="link_1"/>
    <child link="link_2"/>
  </joint>
</robot>
```

### Complex Door with Separate Lock
```xml
<!-- ... (frame and board as above) ... -->

<!-- Link 3: Lock mechanism base -->
<link name="link_3">
  <visual name="lock">
    <geometry>
      <mesh filename="texture_dae/9960003-lock-right.dae" scale="1.1 1.1 1.1"/>
    </geometry>
  </visual>
</link>

<!-- Joint 3: Fixed lock position on door -->
<joint name="joint_3" type="fixed">
  <origin xyz="0.25 -0.14 0.03"/>
  <parent link="link_1"/>
  <child link="link_3"/>
</joint>

<!-- Link 2: Handle (child of lock) -->
<link name="link_2">
  <visual name="handle">
    <geometry>
      <mesh filename="texture_dae/9960003-handle-right.dae" scale="1.1 1.1 1.1"/>
    </geometry>
  </visual>
</link>

<!-- Joint 2: Handle rotates relative to lock -->
<joint name="joint_2" type="revolute">
  <origin xyz="0 0 0"/>
  <axis xyz="0 0 1"/>
  <parent link="link_3"/>
  <child link="link_2"/>
</joint>
```

## Asset Naming Patterns

### Body IDs (First 7 digits)
- `9965xxx` - Standard door bodies
- `9966xxx` - Alternate door styles
- `9961xxx` - Cabinet bodies
- `9967xxx` - Car door bodies
- `9969xxx` - Window frames

### Handle IDs (Last 7 digits)
- `9962xxx` - Round rotating handles
- `9960xxx` - Lever handles
- `9968xxx` - Car door handles
- `9969xxx` - Window handles
- `9961xxx` - Safe locks/dials

### Example Decoding
- `99650069962004` → Body `9965006` + Handle `9962004`
- `99670019968001` → Body `9967001` (car) + Handle `9968001` (car handle)
- `99611129961204` → Body `9961112` (safe) + Handle `9961204` (combination lock)

## Quick Statistics

```
Total Assets:     280
Total Categories: 15
File Types:       .urdf, .json, .dae, .jpg
Avg Asset Size:   ~5 MB
Total Dataset:    ~1.4 GB
```

## Access Patterns

### By Category
```python
categories = [
    "RoundDoor_ccw_pull", "RoundDoor_ccw_push", "RoundDoor_cw_pull", "RoundDoor_cw_push",
    "RoundDoor_direct_pull", "RoundDoor_direct_push",
    "LeverDoor_ccw_pull", "LeverDoor_ccw_push", "LeverDoor_cw_pull", "LeverDoor_cw_push",
    "LeverDoor_direct_pull", "LeverDoor_direct_push",
    "Cabinet_ccw_pull", "Cabinet_cw_pull", "Cabinet_direct_pull",
    "Fridge_ccw_pull", "Fridge_cw_pull", "Fridge_direct_pull",
    "Safe_ccw_pull", "Safe_cw_pull", "Safe_direct_pull",
    "Car_pull",
    "Window_ccw_pull", "Window_cw_pull", "Window_direct_pull",
]
for category in categories:
    assets = os.listdir(f"generated_datasets/{category}")
    print(f"{category}: {len(assets)} assets")
```

### By Asset Name
```python
asset_path = "generated_datasets/RoundDoor_ccw_pull/99660019962014"
urdf = f"{asset_path}/mobility.urdf"
bbox = f"{asset_path}/bounding_box.json"
handle_info = f"{asset_path}/handle_bounding.json"
meshes = f"{asset_path}/texture_dae/"
```

### Random Selection
```python
import random
import glob

# Get all URDF files
all_urdfs = glob.glob("generated_datasets/**/mobility.urdf", recursive=True)
random_asset = random.choice(all_urdfs)
print(f"Selected: {random_asset}")
```
