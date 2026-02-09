"""
Visualize Assets from UniDoorManip Dataset

This script selects the first N assets (sorted by ID) from each category and renders them using Isaac Gym.
The rendered images are saved to visualization/render_image/{category}_{asset_id}.jpg

Categories:
    - Cabinet
    - Car
    - Fridge
    - LeverDoor
    - RoundDoor
    - Safe
    - Window
"""

import os
import argparse

# isaacgym MUST be imported before torch
import isaacgym
from isaacgym import gymapi

import numpy as np
from PIL import Image

# Dataset root directory
DATASET_ROOT = os.path.join(os.path.dirname(__file__), '..')

# Categories to visualize (using capitalized folder names)
CATEGORIES = {
    'Cabinet': 'Cabinet',
    'Car': 'Car',
    'Fridge': 'Fridge',
    'LeverDoor': 'LeverDoor',
    'RoundDoor': 'RoundDoor',
    'Safe': 'Safe',
    'Window': 'Window'
}

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'render_image')


def get_available_assets(category_folder):
    """Get list of available asset IDs for a given category.

    Args:
        category_folder: Category folder name (e.g., 'LeverDoor', 'Cabinet')

    Returns:
        List of available asset ID strings
    """
    category_path = os.path.join(DATASET_ROOT, category_folder)

    if not os.path.exists(category_path):
        return []

    assets = []
    for name in os.listdir(category_path):
        asset_path = os.path.join(category_path, name)
        urdf_path = os.path.join(asset_path, 'mobility.urdf')
        if os.path.isdir(asset_path) and os.path.exists(urdf_path):
            assets.append(name)

    return sorted(assets)


def load_bounding_box(category_folder, asset_id):
    """Load bounding box for an asset.

    Returns:
        tuple: (min_coords, max_coords, center, size) or None if not found
    """
    import json
    bbox_path = os.path.join(DATASET_ROOT, category_folder, asset_id, 'bounding_box.json')

    if not os.path.exists(bbox_path):
        return None

    with open(bbox_path, 'r') as f:
        bbox = json.load(f)

    min_coords = np.array([bbox['min'][0], bbox['min'][1], bbox['min'][2]])
    max_coords = np.array([bbox['max'][0], bbox['max'][1], bbox['max'][2]])
    center = (min_coords + max_coords) / 2.0
    size = max_coords - min_coords

    return min_coords, max_coords, center, size


def visualize_asset(category_folder, asset_id, output_path, headless=True):
    """Visualize a single asset and save the rendered image.

    Args:
        category_folder: Category folder name (e.g., 'LeverDoor', 'Cabinet')
        asset_id: Asset ID string
        output_path: Path to save the rendered image
        headless: If False, open Isaac Gym GUI for user interaction
    """
    print(f"  Rendering {category_folder}/{asset_id}...")

    # Load bounding box to determine asset size
    bbox_info = load_bounding_box(category_folder, asset_id)

    # Calculate lift amount before creating actor
    lift_amount = 0.0
    if bbox_info is not None:
        min_coords, max_coords, center, size = bbox_info
        # Calculate how much to lift
        if min_coords[2] < 0:
            # Asset extends below origin, lift it up
            lift_amount = abs(min_coords[2]) + 0.05
        else:
            # Asset is above origin, adjust to reasonable height
            if size[2] > 1.0:  # Tall object (door)
                lift_amount = 0.0
            else:  # Short object (drawer, safe)
                lift_amount = 0.8  # Table height

    # Initialize gym
    gym = gymapi.acquire_gym()

    # Create sim
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2

    # PhysX-specific params
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.001
    sim_params.physx.friction_offset_threshold = 0.001
    sim_params.physx.friction_correlation_distance = 0.0005
    sim_params.physx.num_threads = 4
    sim_params.physx.use_gpu = True

    # Set compute and graphics device based on headless mode
    compute_device = 0
    graphics_device = 0  # Always use GPU 0 for graphics to enable camera sensors

    sim = gym.create_sim(compute_device, graphics_device, gymapi.SIM_PHYSX, sim_params)

    if sim is None:
        print("    ERROR: Failed to create sim")
        return

    # Create ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

    # Create environment
    env_spacing = 2.0
    env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
    env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
    env = gym.create_env(sim, env_lower, env_upper, 1)

    # Note: Skipping Franka robot to avoid URDF package path resolution issues
    # and to focus on visualizing the assets themselves

    # Load door/object asset
    asset_path = os.path.join(DATASET_ROOT, category_folder, asset_id)
    asset_file = 'mobility.urdf'

    asset_options = gymapi.AssetOptions()
    asset_options.flip_visual_attachments = False
    asset_options.fix_base_link = True
    asset_options.collapse_fixed_joints = False
    asset_options.disable_gravity = True  # Disable gravity to prevent doors from sagging
    asset_options.thickness = 0.001
    asset_options.angular_damping = 0.01
    asset_options.linear_damping = 0.01
    asset_options.max_angular_velocity = 100.0
    asset_options.density = 400.0
    asset_options.armature = 0.01
    asset_options.use_mesh_materials = True
    asset_options.vhacd_enabled = True
    asset_options.vhacd_params.resolution = 300000
    asset_options.vhacd_params.max_convex_hulls = 10
    asset_options.vhacd_params.max_num_vertices_per_ch = 64
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

    door_asset = gym.load_asset(sim, asset_path, asset_file, asset_options)

    # Door pose - adjust height based on pre-calculated lift amount
    door_pose = gymapi.Transform()
    door_pose.p = gymapi.Vec3(0.0, 0.0, lift_amount)
    door_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    door_actor = gym.create_actor(env, door_asset, door_pose, "door", 0, 1, 0)

    # Calculate camera position based on asset size and adjusted height
    if bbox_info is not None:
        min_coords, max_coords, center, size = bbox_info
        # Calculate the maximum dimension to determine camera distance
        max_dim = max(size[0], size[1], size[2])
        # Position camera at distance = max_dimension (no multiplier)
        distance = max_dim * 0.3

        # Adjust camera center to account for lifted object position
        cam_center = center.copy()
        cam_center[2] += lift_amount  # Add the lift amount to z-coordinate
    else:
        # Fallback to default values
        distance = 1.0
        cam_center = np.array([0.0, 0.0, 0.55])

    # Setup camera (3x original size: 768x768)
    camera_props = gymapi.CameraProperties()
    camera_props.width = 768
    camera_props.height = 768
    camera_props.enable_tensors = True

    camera_handle = gym.create_camera_sensor(env, camera_props)

    # Prepare simulation
    gym.prepare_sim(sim)

    # Create viewer if not headless (must be after prepare_sim)
    viewer = None
    if not headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            print("    ERROR: Failed to create viewer")
            headless = True  # Fall back to headless mode
        else:
            # Set viewer camera to same position as render camera
            angle_factor = (0.9, 0.3, 0.6)
            cam_pos = gymapi.Vec3(
                cam_center[0] + distance * angle_factor[0],
                cam_center[1] + distance * angle_factor[1],
                cam_center[2] + distance * angle_factor[2]
            )
            cam_target = gymapi.Vec3(cam_center[0], cam_center[1], cam_center[2])
            gym.viewer_camera_look_at(viewer, env, cam_pos, cam_target)

            print("    Interactive GUI opened. Close the window to continue and save the image...")

    # Get DOF information and open the door/handle to show them clearly
    num_dofs = gym.get_actor_dof_count(env, door_actor)

    # First run a few simulation steps to stabilize
    for _ in range(10):
        gym.simulate(sim)
        gym.fetch_results(sim, True)

    if num_dofs > 0:
        # Get DOF properties
        dof_props = gym.get_actor_dof_properties(env, door_actor)

        # Get the DOF handles for this actor
        actor_handle = door_actor

        # Set DOF states directly - this is the key!
        dof_state_dict = gym.get_actor_dof_states(env, actor_handle, gymapi.STATE_ALL)

        # Modify DOF positions - open significantly to show handles clearly
        for i in range(num_dofs):
            if i == 0:
                # Main door - open it wider (0.7 radians ≈ 40 degrees)
                upper_limit = dof_props['upper'][i]
                # Use 60% of upper limit or 0.7 radians, whichever is smaller
                open_amount = min(0.7, abs(upper_limit) * 0.6) if abs(upper_limit) > 0.01 else 0.5
                dof_state_dict['pos'][i] = open_amount
            elif i == 1:
                # Handle - rotate it significantly
                upper_limit = dof_props['upper'][i]
                # Use 70% of upper limit or 0.5 radians
                rotate_amount = min(0.5, abs(upper_limit) * 0.7) if abs(upper_limit) > 0.01 else 0.35
                dof_state_dict['pos'][i] = rotate_amount

            # Set velocity to zero
            dof_state_dict['vel'][i] = 0.0

        # Apply the DOF states
        gym.set_actor_dof_states(env, actor_handle, dof_state_dict, gymapi.STATE_ALL)

    # Run more simulation steps to let physics settle with the new DOF positions
    for _ in range(60):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)

    # If viewer exists, run interactive loop until user closes the window
    if viewer is not None:
        while not gym.query_viewer_has_closed(viewer):
            # Step the simulation
            gym.simulate(sim)
            gym.fetch_results(sim, True)

            # Step rendering
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, False)

            # Wait for dt to elapse in real time to sync viewer with simulation rate
            gym.sync_frame_time(sim)

        # GUI closed by user - capture the final camera position
        print("    GUI closed by user, capturing camera position...")

        # Get the viewer camera transform
        cam_transform = gym.get_viewer_camera_transform(viewer, env)

        # Extract position and target from the camera transform
        # The viewer camera stores position and a look-at target
        final_cam_pos = cam_transform.p

        # Calculate the look direction and target point
        # We'll use the camera's position and the original center as reference
        print(f"    Final camera position: ({final_cam_pos.x:.2f}, {final_cam_pos.y:.2f}, {final_cam_pos.z:.2f})")

        # Update angle_factor based on user's final camera position
        angle_factor = (
            final_cam_pos.x - cam_center[0],
            final_cam_pos.y - cam_center[1],
            final_cam_pos.z - cam_center[2]
        )
        print(f"    Angle factor: ({angle_factor[0]:.2f}, {angle_factor[1]:.2f}, {angle_factor[2]:.2f})")
        
        # Normalize by distance to get the direction
        current_distance = np.sqrt(angle_factor[0]**2 + angle_factor[1]**2 + angle_factor[2]**2)
        if current_distance > 0.01:
            angle_factor = (
                angle_factor[0] / current_distance,
                angle_factor[1] / current_distance,
                angle_factor[2] / current_distance
            )
            distance = current_distance

        print(f"    Using distance: {distance:.2f}m")
    else:
        # Default angle for headless mode
        angle_factor = (-3.32, 0.00, 0.50)

    # Render image from the final camera position
    cam_pos = gymapi.Vec3(
        cam_center[0] + distance * angle_factor[0],
        cam_center[1] + distance * angle_factor[1],
        cam_center[2] + distance * angle_factor[2]
    )
    cam_target = gymapi.Vec3(cam_center[0], cam_center[1], cam_center[2])
    gym.set_camera_location(camera_handle, env, cam_pos, cam_target)

    # Render the view
    gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)

    # Get camera image
    color_image = gym.get_camera_image(sim, env, camera_handle, gymapi.IMAGE_COLOR)

    # Convert to numpy array and reshape
    color_image = color_image.reshape(camera_props.height, camera_props.width, 4)

    # Convert RGBA to RGB
    rgb_image = color_image[:, :, :3]

    # Save image (no view suffix, just the asset name)
    img = Image.fromarray(rgb_image.astype(np.uint8))
    img.save(output_path)
    print(f"    Saved to {output_path}")

    # Cleanup
    if viewer is not None:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


def main(num_samples=5, seed=None, headless=True, categories=None):
    """Main function to visualize assets from each category.

    Args:
        num_samples: Number of assets to visualize per category (first N sorted by ID)
        seed: Not used (kept for backwards compatibility)
        headless: If False, open Isaac Gym GUI for user interaction
        categories: List of category names to visualize. If None, visualize all categories.
    """

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Determine which categories to visualize
    if categories is None:
        # Visualize all categories
        categories_to_viz = CATEGORIES
    else:
        # Filter to only the specified categories (case-insensitive)
        categories_lower = [c.lower() for c in categories]
        categories_to_viz = {
            name: folder for name, folder in CATEGORIES.items()
            if name.lower() in categories_lower
        }

        # Check for invalid categories
        valid_names = [name.lower() for name in CATEGORIES.keys()]
        invalid = [c for c in categories_lower if c not in valid_names]
        if invalid:
            print(f"WARNING: Invalid categories will be skipped: {invalid}")
            print(f"Valid categories are: {', '.join(CATEGORIES.keys())}")

        if not categories_to_viz:
            print("ERROR: No valid categories specified!")
            print(f"Valid categories are: {', '.join(CATEGORIES.keys())}")
            return

    print(f"Visualizing first {num_samples} assets (sorted by ID) from each category...")
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Mode: {'Interactive GUI' if not headless else 'Headless'}")
    print(f"Categories: {', '.join(categories_to_viz.keys())}")
    print()

    for category_name, category_folder in categories_to_viz.items():
        print(f"Category: {category_name.upper()}")

        # Get all available assets for this category (already sorted)
        available_assets = get_available_assets(category_folder)

        if len(available_assets) == 0:
            print(f"  WARNING: No assets found for {category_name}, skipping...")
            continue

        # Take first num_samples assets (already sorted by ID)
        num_to_sample = min(num_samples, len(available_assets))
        sampled_assets = available_assets[:num_to_sample]

        print(f"  Found {len(available_assets)} assets, taking first {num_to_sample}")

        # Visualize each sampled asset
        for asset_id in sampled_assets:
            output_filename = f"{category_folder}_{asset_id}.jpg"
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            try:
                visualize_asset(category_folder, asset_id, output_path, headless=headless)
            except Exception as e:
                print(f"    ERROR rendering {category_folder}/{asset_id}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print()

    print("Visualization complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize assets from UniDoorManip dataset")
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of assets to visualize per category (takes first N sorted by ID, default: 5)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Not used (kept for backwards compatibility)')
    parser.add_argument('--no-headless', action='store_true',
                        help='Open Isaac Gym GUI for interactive visualization (default: headless mode)')
    parser.add_argument('--category', '--categories', nargs='+', type=str, default=None,
                        help='List of categories to visualize (space-separated). '
                             'Valid: Cabinet Car Fridge LeverDoor RoundDoor Safe Window. '
                             'If not specified, all categories will be visualized. '
                             'Example: --category Cabinet Car LeverDoor')

    args = parser.parse_args()

    main(num_samples=args.num_samples, seed=args.seed, headless=not args.no_headless, categories=args.category)
