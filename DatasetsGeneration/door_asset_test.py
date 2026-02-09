"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


UniDoorManip Asset Visualization Tool
-------------------------------------
Visualize different types of door/cabinet/window assets from the generated datasets.
Usage: python door_asset_test.py --example_id <ID>
"""

import os
import math
import argparse
from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
from isaacgym import gymtorch
import json
import torch

# ==============================================================================
# Asset Configuration - Add your assets here
# ==============================================================================
ASSET_EXAMPLES = [
    {
        "id": 0,
        "name": "Round Door Handle",
        "category": "RoundDoor",
        "asset_name": "99660019962014",
        "description": "Door with round rotating handle"
    },
    {
        "id": 1,
        "name": "Lever Door Handle",
        "category": "LeverDoor",
        "asset_name": "99650069960003",
        "description": "Door with lever-style handle"
    },
    {
        "id": 2,
        "name": "Cabinet Door",
        "category": "Cabinet",
        "asset_name": "99613029962004",
        "description": "Kitchen/office cabinet door"
    },
    {
        "id": 3,
        "name": "Refrigerator Door",
        "category": "Fridge",
        "asset_name": "99614019960005",
        "description": "Refrigerator door"
    },
    {
        "id": 4,
        "name": "Safe Door",
        "category": "Safe",
        "asset_name": "99611129961204",
        "description": "Safe box with combination lock"
    },
    {
        "id": 5,
        "name": "Car Door",
        "category": "Car",
        "asset_name": "99670019968001",
        "description": "Vehicle door"
    },
    {
        "id": 6,
        "name": "Window",
        "category": "Window",
        "asset_name": "99690049969510",
        "description": "Sliding or rotating window"
    },
]

def print_available_examples():
    """Print all available asset examples"""
    print("\n" + "="*70)
    print("Available Asset Examples:")
    print("="*70)
    for asset in ASSET_EXAMPLES:
        print(f"  [{asset['id']}] {asset['name']:25s} - {asset['description']}")
    print("="*70)
    print(f"Usage: python door_asset_test.py --example_id <ID>")
    print("="*70 + "\n")

def get_asset_config(example_id):
    """Get asset configuration by example ID"""
    for asset in ASSET_EXAMPLES:
        if asset['id'] == example_id:
            return asset
    return None

# ==============================================================================
# Main Script
# ==============================================================================

# Parse custom arguments
custom_parser = argparse.ArgumentParser(description="UniDoorManip Asset Viewer")
custom_parser.add_argument('--example_id', type=int, default=0,
                          help='Asset example ID to visualize (use --list to see all)')
custom_parser.add_argument('--list', action='store_true',
                          help='List all available asset examples and exit')
custom_args, remaining_args = custom_parser.parse_known_args()

# Show list and exit if requested
if custom_args.list:
    print_available_examples()
    exit(0)

# Get asset configuration
asset_config = get_asset_config(custom_args.example_id)
if asset_config is None:
    print(f"\nError: Invalid example_id '{custom_args.example_id}'")
    print_available_examples()
    exit(1)

print("\n" + "="*70)
print(f"Loading Asset Example #{asset_config['id']}: {asset_config['name']}")
print(f"Category: {asset_config['category']}")
print(f"Description: {asset_config['description']}")
print("="*70 + "\n")

# Initialize gym
gym = gymapi.acquire_gym()

# Parse IsaacGym arguments
args = gymutil.parse_arguments(
    description="UniDoorManip Asset Visualization",
    custom_parameters=[{
        "name": "--example_id",
        "type": int,
        "default": 0,
        "help": "Asset example ID"
    }]
)

# Create simulator
sim_params = gymapi.SimParams()
sim_params.substeps = 2
sim_params.dt = 1.0 / 60.0
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.physx.use_gpu = True
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(0, 0, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# Create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError('*** Failed to create viewer')

# Add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up
plane_params.distance = 0
plane_params.static_friction = 12
plane_params.dynamic_friction = 1
plane_params.restitution = 0
gym.add_ground(sim, plane_params)

# Set up environment grid
num_envs = 1
spacing = 1.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, 0.0, spacing)

# Construct asset paths based on selected example
asset_root = f"../generated_datasets/{asset_config['category']}"
asset_file = f"{asset_config['asset_name']}/mobility.urdf"
bounding_box_path = f"../generated_datasets/{asset_config['category']}/{asset_config['asset_name']}/bounding_box.json"

# Check if asset exists
if not os.path.exists(os.path.join(asset_root, asset_file)):
    print(f"\nError: Asset not found at {os.path.join(asset_root, asset_file)}")
    print("Please make sure you have generated the datasets first.")
    exit(1)

# Load asset
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.disable_gravity = True
asset_options.collapse_fixed_joints = True
asset_options.use_mesh_materials = True
asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
asset_options.override_com = True
asset_options.override_inertia = True
asset_options.vhacd_enabled = True
asset_options.vhacd_params = gymapi.VhacdParams()
asset_options.vhacd_params.resolution = 1024

print(f"Loading asset '{asset_file}' from '{asset_root}'")
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

if asset is None:
    print(f"*** Failed to load asset")
    exit(1)

# Get asset bounding box for positioning
with open(bounding_box_path, "r") as f:
    bounding_box = json.load(f)
    min_b = bounding_box["min"]

# Set initial pose - position asset above ground
initial_pose = gymapi.Transform()
initial_pose.p = gymapi.Vec3(0.0, 0.0, -min_b[2] + 0.1)
initial_pose.r = gymapi.Quat(0, 0, 1, 0)  # Rotate 180 degrees to face camera

# Create environment and actor
env = gym.create_env(sim, env_lower, env_upper, 2)
actor = gym.create_actor(env, asset, initial_pose, asset_config['name'], 0, 1)

# Configure DOF properties
props = gym.get_actor_dof_properties(env, actor)
num_dofs = len(props)

if num_dofs > 0:
    # Set all DOFs to position control mode
    props["driveMode"].fill(gymapi.DOF_MODE_POS)
    props["stiffness"].fill(5000.0)
    props["damping"].fill(100.0)
    gym.set_actor_dof_properties(env, actor, props)

    dof_dict = gym.get_actor_dof_dict(env, actor)
    print(f"\nDegrees of Freedom: {dof_dict}")
else:
    print("\nNo articulated joints found in this asset")

# Set camera view
cam_pos = gymapi.Vec3(3, 0, 2)
cam_target = gymapi.Vec3(0, 0, 1)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

print("\n" + "="*70)
print("Visualization Controls:")
print("  - Mouse: Rotate/Pan/Zoom camera")
print("  - ESC: Exit")
print("="*70 + "\n")

# Simulation loop
frame_count = 0
while not gym.query_viewer_has_closed(viewer):
    # Step physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # Optional: Animate joints (uncomment to enable)
    # if num_dofs > 0 and frame_count % 60 == 0:
    #     # Slowly open/close joints
    #     angle = math.sin(frame_count * 0.01) * 0.5 + 0.5  # Range [0, 1]
    #     for dof_name, dof_idx in dof_dict.items():
    #         gym.set_dof_target_position(env, gym.find_actor_dof_handle(env, actor, dof_name), angle)

    # Update graphics
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

    frame_count += 1

print('Done')

# Cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
