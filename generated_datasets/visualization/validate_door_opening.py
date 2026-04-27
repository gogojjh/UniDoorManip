"""Validate door opening by sweeping hinge joint angle in Isaac Gym.

Loads each door asset, sets the hinge joint to several angles across its limit
range, and renders an image at each angle. This verifies that pull doors open
toward the viewer and push doors open away.

Output: visualization/render_image/validate_{category}_{asset_id}_{angle_deg}.jpg

Usage:
    # Validate first 3 LeverDoor_ccw_pull and LeverDoor_cw_push assets
    python validate_door_opening.py --category LeverDoor_ccw_pull LeverDoor_cw_push --num-samples 3

    # Validate a specific asset
    python validate_door_opening.py --category LeverDoor_ccw_pull --asset-id 99650069960003

    # Validate all LeverDoor* with interactive GUI
    python validate_door_opening.py --category LeverDoor_ccw_pull LeverDoor_ccw_push --no-headless
"""

import argparse
import json
import os
from typing import List, Optional

import isaacgym
from isaacgym import gymapi

import numpy as np
from PIL import Image


DATASET_ROOT = os.path.join(os.path.dirname(__file__), '..')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'render_image')

VALID_CATEGORIES = [
    'Cabinet_ccw_pull', 'Cabinet_cw_pull',
    'Car',
    'Fridge_ccw_pull', 'Fridge_cw_pull',
    'LeverDoor_ccw_pull', 'LeverDoor_ccw_push', 'LeverDoor_cw_pull', 'LeverDoor_cw_push',
    'RoundDoor_ccw_pull', 'RoundDoor_ccw_push', 'RoundDoor_cw_pull', 'RoundDoor_cw_push',
    'Safe',
    'Window_ccw_pull', 'Window_cw_pull',
]

ANGLE_STEPS = [0.0, 0.33, 0.66, 1.0]  # fractions of the joint range


def get_available_assets(category: str) -> List[str]:
    cat_path = os.path.join(DATASET_ROOT, category)
    if not os.path.exists(cat_path):
        return []
    return sorted(
        name for name in os.listdir(cat_path)
        if os.path.isdir(os.path.join(cat_path, name))
        and os.path.exists(os.path.join(cat_path, name, 'mobility.urdf'))
    )


def load_bounding_box(category: str, asset_id: str):
    bbox_path = os.path.join(DATASET_ROOT, category, asset_id, 'bounding_box.json')
    if not os.path.exists(bbox_path):
        return None
    with open(bbox_path) as f:
        bbox = json.load(f)
    mn = np.array(bbox['min'])
    mx = np.array(bbox['max'])
    return mn, mx, (mn + mx) / 2.0, mx - mn


def validate_asset(
    category: str,
    asset_id: str,
    headless: bool = True,
    sweep_handle: bool = False,
) -> None:
    """Render a door at several hinge angles and save images."""
    print(f"\n  Validating {category}/{asset_id} ...")

    bbox_info = load_bounding_box(category, asset_id)
    lift = 0.0
    if bbox_info is not None:
        mn, mx, center, size = bbox_info
        if mn[2] < 0:
            lift = abs(mn[2]) + 0.05
        elif size[2] <= 1.0:
            lift = 0.8

    # --- Isaac Gym setup (mirrors viz_assets.py) ---
    gym = gymapi.acquire_gym()

    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.001
    sim_params.physx.friction_offset_threshold = 0.001
    sim_params.physx.friction_correlation_distance = 0.0005
    sim_params.physx.num_threads = 4
    sim_params.physx.use_gpu = True

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        print("    ERROR: Failed to create sim")
        return

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

    env = gym.create_env(
        sim,
        gymapi.Vec3(-2.0, -2.0, 0.0),
        gymapi.Vec3(2.0, 2.0, 2.0),
        1,
    )

    asset_path = os.path.join(DATASET_ROOT, category, asset_id)
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.collapse_fixed_joints = False
    asset_options.disable_gravity = True
    asset_options.thickness = 0.001
    asset_options.use_mesh_materials = True
    asset_options.vhacd_enabled = True
    asset_options.vhacd_params.resolution = 300000
    asset_options.vhacd_params.max_convex_hulls = 10
    asset_options.vhacd_params.max_num_vertices_per_ch = 64
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

    door_asset = gym.load_asset(sim, asset_path, 'mobility.urdf', asset_options)

    door_pose = gymapi.Transform()
    door_pose.p = gymapi.Vec3(0.0, 0.0, lift)
    door_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    door_actor = gym.create_actor(env, door_asset, door_pose, "door", 0, 1, 0)

    # Camera
    if bbox_info is not None:
        max_dim = max(size)
        distance = max_dim * 0.3
        cam_center = center.copy()
        cam_center[2] += lift
    else:
        distance = 1.0
        cam_center = np.array([0.0, 0.0, 0.55])

    camera_props = gymapi.CameraProperties()
    camera_props.width = 768
    camera_props.height = 768
    camera_props.enable_tensors = True
    camera_handle = gym.create_camera_sensor(env, camera_props)

    angle_factor = (-3.32, 0.00, 0.50)
    cam_pos = gymapi.Vec3(
        cam_center[0] + distance * angle_factor[0],
        cam_center[1] + distance * angle_factor[1],
        cam_center[2] + distance * angle_factor[2],
    )
    cam_target = gymapi.Vec3(cam_center[0], cam_center[1], cam_center[2])
    gym.set_camera_location(camera_handle, env, cam_pos, cam_target)

    gym.prepare_sim(sim)

    # Viewer for interactive mode
    viewer = None
    if not headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is not None:
            gym.viewer_camera_look_at(viewer, env, cam_pos, cam_target)

    # DOF info
    num_dofs = gym.get_actor_dof_count(env, door_actor)
    if num_dofs == 0:
        print("    WARNING: No DOFs found, skipping.")
        gym.destroy_sim(sim)
        return

    dof_props = gym.get_actor_dof_properties(env, door_actor)
    hinge_lower = float(dof_props['lower'][0])
    hinge_upper = float(dof_props['upper'][0])
    print(f"    Hinge limits: [{np.degrees(hinge_lower):.1f}, {np.degrees(hinge_upper):.1f}] deg")

    handle_lower, handle_upper = 0.0, 0.0
    if num_dofs > 1:
        handle_lower = float(dof_props['lower'][1])
        handle_upper = float(dof_props['upper'][1])
        print(f"    Handle limits: [{np.degrees(handle_lower):.1f}, {np.degrees(handle_upper):.1f}] deg")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Build list of (hinge_frac, handle_frac, label) to render
    if sweep_handle and num_dofs > 1:
        render_steps = []
        for frac in ANGLE_STEPS:
            render_steps.append((0.0, frac, "handle"))   # door closed, sweep handle
        for frac in ANGLE_STEPS[1:]:
            render_steps.append((frac, 1.0, "hinge"))    # handle fully open, sweep hinge
    else:
        render_steps = [(frac, 0.0, "hinge") for frac in ANGLE_STEPS]

    for hinge_frac, handle_frac, sweep_label in render_steps:
        hinge_angle = hinge_lower + hinge_frac * (hinge_upper - hinge_lower)
        handle_angle = handle_lower + handle_frac * (handle_upper - handle_lower)
        hinge_deg = np.degrees(hinge_angle)
        handle_deg = np.degrees(handle_angle)

        # Set joint angles
        dof_states = gym.get_actor_dof_states(env, door_actor, gymapi.STATE_ALL)
        dof_states['pos'][0] = hinge_angle
        dof_states['vel'][0] = 0.0
        if num_dofs > 1:
            dof_states['pos'][1] = handle_angle
            dof_states['vel'][1] = 0.0
        gym.set_actor_dof_states(env, door_actor, dof_states, gymapi.STATE_ALL)

        # Settle physics
        for _ in range(30):
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)

        # Interactive: let user inspect
        if viewer is not None:
            print(f"    hinge={hinge_deg:+.1f} handle={handle_deg:+.1f} — close viewer to continue...")
            while not gym.query_viewer_has_closed(viewer):
                gym.simulate(sim)
                gym.fetch_results(sim, True)
                gym.step_graphics(sim)
                gym.draw_viewer(viewer, sim, False)
                gym.sync_frame_time(sim)
            gym.destroy_viewer(viewer)
            viewer = gym.create_viewer(sim, gymapi.CameraProperties())
            if viewer is not None:
                gym.viewer_camera_look_at(viewer, env, cam_pos, cam_target)

        # Render
        gym.render_all_camera_sensors(sim)
        color_image = gym.get_camera_image(sim, env, camera_handle, gymapi.IMAGE_COLOR)
        color_image = color_image.reshape(camera_props.height, camera_props.width, 4)[:, :, :3]

        fname = f"validate_{category}_{asset_id}_{sweep_label}_h{hinge_deg:+.0f}_hdl{handle_deg:+.0f}.jpg"
        out_path = os.path.join(OUTPUT_DIR, fname)
        Image.fromarray(color_image.astype(np.uint8)).save(out_path)
        print(f"    Saved: {fname}")

    if viewer is not None:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate door opening by sweeping hinge joint angle in Isaac Gym."
    )
    parser.add_argument(
        '--category', nargs='+', type=str, required=True,
        help=f"Categories to validate. Valid: {', '.join(VALID_CATEGORIES)}",
    )
    parser.add_argument(
        '--asset-id', type=str, default=None,
        help="Validate a single asset ID (requires exactly one --category).",
    )
    parser.add_argument(
        '--num-samples', type=int, default=3,
        help="Number of assets per category (first N sorted by ID). Default: 3.",
    )
    parser.add_argument(
        '--no-headless', action='store_true',
        help="Open Isaac Gym GUI for interactive inspection at each angle.",
    )
    parser.add_argument(
        '--sweep-handle', action='store_true',
        help="Also sweep handle joint (joint_2) angles in addition to hinge.",
    )
    args = parser.parse_args()

    for cat in args.category:
        print(f"\nCategory: {cat}")
        if args.asset_id:
            assets = [args.asset_id]
        else:
            assets = get_available_assets(cat)[:args.num_samples]

        if not assets:
            print(f"  No assets found for {cat}, skipping.")
            continue

        print(f"  Validating {len(assets)} asset(s)...")
        for aid in assets:
            try:
                validate_asset(cat, aid, headless=not args.no_headless,
                               sweep_handle=args.sweep_handle)
            except Exception as e:
                print(f"    ERROR {cat}/{aid}: {e}")
                import traceback
                traceback.print_exc()

    print("\nValidation complete!")


if __name__ == "__main__":
    main()
