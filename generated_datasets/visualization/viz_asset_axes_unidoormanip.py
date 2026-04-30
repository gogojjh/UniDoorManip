"""
Visualize a UniDoorManip door asset in IsaacGym with RGB coordinate axes (headless).
Asset loading replicates the exact procedure used during expert data collection
(base_env.py + franka_slider_door.py).

Renders to image file in visualization/viz_asset_axes/<task>_<asset_id>.png

Usage:
    conda activate 3d_dp
    python viz_asset_axes_unidoormanip.py --dataset LeverDoor_ccw_pull --asset_id 99650069960003
    python viz_asset_axes_unidoormanip.py --all  # render first asset of every category
"""

import argparse
import json
import os
import numpy as np
from isaacgym import gymapi, gymutil

# ─── Config ───────────────────────────────────────────────────────────────────
ASSET_ROOT = '/Titan/code/robohike_ws/src/UniDoorManip/generated_datasets'
OUTPUT_DIR = os.path.join(ASSET_ROOT, 'visualization', 'viz_asset_axes')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Task-specific Z offsets (matches franka_slider_door.py)
TASK_Z_OFFSET = {
    'Cabinet': 0.20,
    'Safe': 0.50,
    'Window': 0.50,
}


def get_task_z_offset(dataset_path: str) -> float:
    for key, val in TASK_Z_OFFSET.items():
        if dataset_path.startswith(key):
            return val
    return 0.0


def get_all_categories() -> list:
    """Return list of (dataset_path, asset_id) for first asset in each category."""
    categories = []
    for name in sorted(os.listdir(ASSET_ROOT)):
        cat_dir = os.path.join(ASSET_ROOT, name)
        if not os.path.isdir(cat_dir) or name == 'visualization':
            continue
        assets = sorted(os.listdir(cat_dir))
        for a in assets:
            if os.path.isfile(os.path.join(cat_dir, a, 'mobility.urdf')):
                categories.append((name, a))
                break
    return categories


def render_asset(dataset_path: str, asset_id: str) -> None:
    """Render a single asset with axes and save to viz_asset_axes/."""
    urdf_path = f'{dataset_path}/{asset_id}/mobility.urdf'
    bbox_path = f'{ASSET_ROOT}/{dataset_path}/{asset_id}/bounding_box.json'
    output_path = os.path.join(OUTPUT_DIR, f'{dataset_path}_{asset_id}.png')

    # ─── Initialize gym ──────────────────────────────────────────────────────
    gym = gymapi.acquire_gym()

    sim_params = gymapi.SimParams()
    sim_params.substeps = 2
    sim_params.dt = 1.0 / 60.0
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.physx.use_gpu = True
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0
    sim_params.use_gpu_pipeline = False

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    assert sim is not None, "Failed to create sim"

    # Ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    plane_params.static_friction = 0.1
    plane_params.dynamic_friction = 0.1
    gym.add_ground(sim, plane_params)

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
    asset_options.vhacd_params.resolution = 2048

    print(f"Loading asset: {urdf_path}")
    door_asset = gym.load_asset(sim, ASSET_ROOT, urdf_path, asset_options)

    # Initial pose
    with open(bbox_path, 'r') as f:
        bbox = json.load(f)

    initial_pose = gymapi.Transform()
    base_z = -bbox["min"][2] + 0.1 + get_task_z_offset(dataset_path)
    initial_pose.p = gymapi.Vec3(0.0, 0.0, base_z)
    initial_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

    # Create env and actor
    spacing = 2.0
    env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)
    env = gym.create_env(sim, env_lower, env_upper, 1)
    door_actor = gym.create_actor(env, door_asset, initial_pose, 'door', 0, 1)

    # Camera sensor
    cam_props = gymapi.CameraProperties()
    cam_props.width = 1280
    cam_props.height = 960
    cam_props.enable_tensors = False
    cam_handle = gym.create_camera_sensor(env, cam_props)

    cam_pos = gymapi.Vec3(1.8, -0.9, 1.7)
    cam_target = gymapi.Vec3(0.0, 0.0, base_z * 0.6)
    gym.set_camera_location(cam_handle, env, cam_pos, cam_target)

    # Prepare sim
    gym.prepare_sim(sim)
    body_states = gym.get_actor_rigid_body_states(env, door_actor, gymapi.STATE_POS)
    num_bodies = len(body_states)
    body_names = gym.get_actor_rigid_body_names(env, door_actor)
    print(f"  Rigid bodies ({num_bodies}): {body_names}")

    # Step once
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # Create viewer for line drawing
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    viewer_available = viewer is not None

    if viewer_available:
        gym.clear_lines(viewer)
        # World axes at origin
        _draw_axes(gym, viewer, env,
                   np.array([0.0, 0.0, 0.01], dtype=np.float64),
                   (0.0, 0.0, 0.0, 1.0), 0.5)
        # Per-body local axes
        body_states = gym.get_actor_rigid_body_states(env, door_actor, gymapi.STATE_POS)
        for i in range(num_bodies):
            pos = body_states['pose']['p'][i]
            rot = body_states['pose']['r'][i]
            origin = np.array([pos[0], pos[1], pos[2]], dtype=np.float64)
            quat = (rot[0], rot[1], rot[2], rot[3])
            _draw_axes(gym, viewer, env, origin, quat, 0.2)

    # Render
    gym.step_graphics(sim)
    if viewer_available:
        gym.draw_viewer(viewer, sim, True)
    gym.render_all_camera_sensors(sim)

    # Save image
    image = gym.get_camera_image(sim, env, cam_handle, gymapi.IMAGE_COLOR)
    image = image.reshape(cam_props.height, cam_props.width, 4)[:, :, :3]

    from PIL import Image
    img = Image.fromarray(image)
    img.save(output_path)
    print(f"  Saved: {output_path}")

    # Cleanup
    if viewer_available:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


# ─── Axis drawing helpers ─────────────────────────────────────────────────────

def _quat_rotate(q, v):
    qx, qy, qz, qw = q
    t = 2.0 * np.cross([qx, qy, qz], v)
    return v + qw * t + np.cross([qx, qy, qz], t)


def _draw_axes(gym, viewer, env_ptr, origin, quat, length):
    """Draw RGB axes. Thickness: 5x5 grid of parallel lines."""
    axes = np.eye(3) * length
    colors_rgb = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    offset_dist = 0.003

    grid_range = [-2, -1, 0, 1, 2]
    offsets_grid = [(a, b) for a in grid_range for b in grid_range]

    for i in range(3):
        direction = _quat_rotate(quat, axes[i])
        end = origin + direction

        perp1 = _quat_rotate(quat, np.eye(3)[(i + 1) % 3]) * offset_dist
        perp2 = _quat_rotate(quat, np.eye(3)[(i + 2) % 3]) * offset_dist

        num_lines = len(offsets_grid)
        verts = np.empty((num_lines * 6,), dtype=np.float32)
        colors = np.tile(colors_rgb[i], num_lines)

        for idx, (a, b) in enumerate(offsets_grid):
            off = perp1 * a + perp2 * b
            s = (origin + off).astype(np.float32)
            e = (end + off).astype(np.float32)
            verts[idx * 6: idx * 6 + 3] = s
            verts[idx * 6 + 3: idx * 6 + 6] = e

        gym.add_lines(viewer, env_ptr, num_lines, verts, colors)


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize UniDoorManip asset axes")
    parser.add_argument('--dataset', type=str, default='LeverDoor_ccw_pull')
    parser.add_argument('--asset_id', type=str, default='99650069960003')
    parser.add_argument('--all', action='store_true', help='Render first asset of every category')
    # gymutil adds its own args, so use parse_known_args
    known_args, _ = parser.parse_known_args()

    if known_args.all:
        categories = get_all_categories()
        print(f"Rendering {len(categories)} categories...")
        for dataset_path, asset_id in categories:
            print(f"\n[{dataset_path}/{asset_id}]")
            render_asset(dataset_path, asset_id)
    else:
        render_asset(known_args.dataset, known_args.asset_id)

    print("\nAll done.")
