"""Utility: compute joint world poses for articulated objects.

Provides two public entry points:
- build_joint_world_pose()           : low-level math (z-axis = motion axis)
- load_joint_kinematics_from_json()  : parse kinematics.json + compute world poses
- load_kinematics_params()           : parse only (no world-pose computation)
- compute_joint_world_poses()        : compute from pre-parsed params
"""
import json
from typing import Dict
import numpy as np
from scipy.spatial.transform import Rotation

_JOINT_TYPE = {'revolute': 0, 'prismatic': 1}


def build_joint_world_pose(
    local_pos: np.ndarray,
    local_axis: np.ndarray,
    link_pos: np.ndarray,
    link_quat_xyzw: np.ndarray,
) -> np.ndarray:
    """Compute 7D [tx, ty, tz, qw, qx, qy, qz] world pose for a joint.

    Canonical frame (DAMP convention): z-axis = joint motion axis.

    Args:
        local_pos:      (3,) joint origin in link frame
        local_axis:     (3,) joint axis in link frame (will be normalised)
        link_pos:       (3,) link world position
        link_quat_xyzw: (4,) link world quaternion [x, y, z, w]
    Returns:
        (7,) float32 [tx, ty, tz, qw, qx, qy, qz]
    """
    R_link = Rotation.from_quat(link_quat_xyzw).as_matrix()
    world_pos = R_link @ local_pos + link_pos
    z = R_link @ local_axis
    z /= np.linalg.norm(z) + 1e-8

    # Gram-Schmidt: build right-handed frame with z = motion axis
    fallback = np.array([1., 0., 0.]) if abs(z[0]) < 0.9 else np.array([0., 1., 0.])
    x = fallback - np.dot(fallback, z) * z
    x /= np.linalg.norm(x) + 1e-8
    y = np.cross(z, x)
    rot_mat = np.column_stack([x, y, z])

    q_xyzw = Rotation.from_matrix(rot_mat).as_quat()
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)
    return np.concatenate([world_pos.astype(np.float32), q_wxyz])


def load_kinematics_params(kin_path: str) -> Dict:
    """Parse kinematics.json and return raw local params (no world-pose computation).

    Returns dict with keys:
        hinge_local_pos, hinge_local_axis, tau_hinge, theta_hinge_min, theta_hinge_max,
        handle_local_pos, handle_local_axis, tau_handle, theta_handle_min, theta_handle_max
    """
    with open(kin_path) as f:
        kd = json.load(f)
    hd = kd['hinge']
    hnd = kd['handle']
    handle_axis_key = 'rotation_axis' if 'rotation_axis' in hnd else 'axis'
    return {
        'hinge_local_pos':   np.array(hd['position'], dtype=np.float64),
        'hinge_local_axis':  np.array(hd.get('axis', hd.get('rotation_axis')), dtype=np.float64),
        'tau_hinge':         _JOINT_TYPE.get(hd['joint_type'], 0),
        'theta_hinge_min':   float(hd.get('lower_limit_rad', 0.0)),
        'theta_hinge_max':   float(hd.get('upper_limit_rad', 1.5)),
        'handle_local_pos':  np.array(
            hnd.get('position_on_door', hnd.get('position')), dtype=np.float64),
        'handle_local_axis': np.array(hnd[handle_axis_key], dtype=np.float64),
        'tau_handle':        _JOINT_TYPE.get(hnd['joint_type'], 0),
        'theta_handle_min':  float(hnd.get('lower_limit_rad', 0.0)),
        'theta_handle_max':  float(hnd.get('upper_limit_rad', 1.5)),
    }


def compute_joint_world_poses(
    params: Dict, door_pos: np.ndarray, door_quat_xyzw: np.ndarray
) -> Dict:
    """Compute world poses for hinge and handle joints from pre-parsed params.

    Args:
        params:         dict returned by load_kinematics_params()
        door_pos:       (3,) door world position
        door_quat_xyzw: (4,) door world quaternion [x, y, z, w]
    Returns:
        dict with T_world_joint_hinge (7,), T_world_joint_handle (7,),
        tau_hinge, tau_handle, theta_*_min, theta_*_max
    """
    T_hinge = build_joint_world_pose(
        params['hinge_local_pos'], params['hinge_local_axis'],
        door_pos, door_quat_xyzw,
    )
    T_handle = build_joint_world_pose(
        params['handle_local_pos'], params['handle_local_axis'],
        door_pos, door_quat_xyzw,
    )
    return {
        'T_world_joint_hinge':  T_hinge,
        'T_world_joint_handle': T_handle,
        'tau_hinge':           params['tau_hinge'],
        'tau_handle':          params['tau_handle'],
        'theta_hinge_min':     params['theta_hinge_min'],
        'theta_hinge_max':     params['theta_hinge_max'],
        'theta_handle_min':    params['theta_handle_min'],
        'theta_handle_max':    params['theta_handle_max'],
    }


def load_joint_kinematics_from_json(
    kin_path: str, door_pos: np.ndarray, door_quat_xyzw: np.ndarray
) -> Dict:
    """Parse kinematics.json and compute world poses for hinge and handle joints.

    One-shot convenience wrapper around load_kinematics_params + compute_joint_world_poses.

    Args:
        kin_path:       path to kinematics.json
        door_pos:       (3,) door world position
        door_quat_xyzw: (4,) door world quaternion [x, y, z, w]
    Returns:
        dict with T_world_joint_hinge (7,), T_world_joint_handle (7,), tau_*, theta_*
    """
    return compute_joint_world_poses(load_kinematics_params(kin_path), door_pos, door_quat_xyzw)
