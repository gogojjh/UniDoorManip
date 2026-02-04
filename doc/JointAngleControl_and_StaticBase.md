# Joint Angle Control and Static Base Configuration

## 1. How Joint Angles Are Determined

### 1.1 Robot Model Loading (`base_env.py:309-320`)

The system loads different URDF files based on the `mobile` configuration:

```python
if self.mobile:
    asset_file = "franka_description/robots/franka_panda_slider.urdf"  # 11 DOFs
else:
    asset_file = "franka_description/robots/franka_panda.urdf"        # 9 DOFs
```

**Mobile Robot (franka_panda_slider.urdf):**
- DOF 0-1: Base mobile joints (X, Y prismatic)
  - `z_free_base1`: X-axis translation, range [-1.0, 1.0]m
  - `z_free_base2`: Y-axis translation, range [-1.0, 1.0]m
- DOF 2-8: Arm joints (7 revolute joints)
- DOF 9-10: Gripper fingers (2 prismatic joints)
- **Total: 11 DOFs**

**Fixed Base Robot (franka_panda.urdf):**
- DOF 0-6: Arm joints (7 revolute joints)
- DOF 7-8: Gripper fingers (2 prismatic joints)
- **Total: 9 DOFs**

### 1.2 DOF State Tensor Structure (`base_env.py:615-621`)

```python
self.franka_dof_tensor = self.dof_state_tensor[:, :self.franka_num_dofs, :]
# Shape: [num_envs, franka_num_dofs, 2] where last dim is [position, velocity]
```

For mobile base (11 DOFs):
- `dof_state_tensor[:, 0, :]` - Base X position/velocity
- `dof_state_tensor[:, 1, :]` - Base Y position/velocity
- `dof_state_tensor[:, 2:9, :]` - Arm joint positions/velocities
- `dof_state_tensor[:, 9:11, :]` - Gripper positions/velocities

For fixed base (9 DOFs):
- `dof_state_tensor[:, 0:7, :]` - Arm joint positions/velocities
- `dof_state_tensor[:, 7:9, :]` - Gripper positions/velocities

### 1.3 Initial Joint Configuration (`base_env.py:367-372`)

```python
self.franka_num_dofs = self.gym.get_asset_dof_count(self.franka_asset)
default_dof_pos = np.zeros(self.franka_num_dofs, dtype=np.float32)
# Initialize arm joints to 30% of their range
default_dof_pos[:-2] = (franka_dof_lower_limits + franka_dof_upper_limits)[:-2] * 0.3
# Grippers open
default_dof_pos[-2:] = franka_dof_upper_limits[-2:]
```

## 2. Controller: Converting Position to Joint Angles

There are THREE control modes that convert end-effector pose to joint angles:

### 2.1 Inverse Kinematics (IK) Mode (`franka_slider_door.py:249-261`)

```python
if self.cfg["env"]["driveMode"] == "ik":
    joints = self.franka_num_dofs - 2  # Exclude gripper DOFs

    # Target end-effector position
    target_pos = actions[:, :3] * self.space_range + self.space_middle
    pos_err = target_pos - self.hand_rigid_body_tensor[:, :3]

    # Target end-effector rotation
    target_rot = actions[:, 3:7] / torch.sqrt((actions[:, 3:7]**2).sum(dim=-1)+1e-8).view(-1, 1)
    rot_err = orientation_error(target_rot, self.hand_rigid_body_tensor[:, 3:7])

    # 6D pose error [x, y, z, rx, ry, rz]
    dpose = torch.cat([pos_err, rot_err], -1).unsqueeze(-1)

    # Solve IK using Jacobian (damped least squares)
    delta = control_ik(self.jacobian_tensor[:, self.hand_rigid_body_index - 1, :, :-2],
                      self.device, dpose, self.num_envs)

    # Update joint positions
    self.pos_act[:, :joints] = dof_pos.squeeze(-1)[:, :joints] + delta
```

**IK Solver (`control_ik` function, lines 26-34):**
```python
def control_ik(j_eef, device, dpose, num_envs):
    damping = 0.05
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    # Damped least squares: Δθ = J^T(JJ^T + λI)^(-1) * Δx
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, -1)
    return u
```

### 2.2 Operational Space Control (OSC) Mode (`franka_slider_door.py:268-291`)

```python
else:  # osc
    joints = self.franka_num_dofs - 2

    # Position and orientation errors
    pos_err = actions[:,:3] - self.hand_rigid_body_tensor[:, :3]
    orn_err = orientation_error(actions[:,3:7], self.hand_rigid_body_tensor[:, 3:7])
    dpose = torch.cat([pos_err, orn_err], -1)

    # Compute operational space inertia matrix
    mm_inv = torch.inverse(self.mm)  # Mass matrix inverse
    m_eef_inv = self.j_eef @ mm_inv @ torch.transpose(self.j_eef, 1, 2)
    m_eef = torch.inverse(m_eef_inv)

    # Impedance control forces
    self.impedance_force = (self.kp * dpose) - (self.kv * (self.j_eef @ self.franka_dof_tensor[:, :joints, 1].unsqueeze(-1))).squeeze()

    # Task space control torque
    u = torch.transpose(self.j_eef, 1, 2) @ m_eef @ (self.kp * dpose.unsqueeze(-1) - self.kv * self.hand_rigid_body_tensor[:, 7:].unsqueeze(-1))

    # Null space control (maintain preferred posture)
    u_null = self.kd_null * -self.franka_dof_tensor[:,:,1].unsqueeze(-1) + self.kp_null * (
            (self.initial_dof_states[:,:self.franka_num_dofs, 0].view(self.num_envs,-1,1) - self.franka_dof_tensor[:,:,0].unsqueeze(-1) + np.pi) % (2 * np.pi) - np.pi)
    u_null = u_null[:,:joints]
    u_null = self.mm @ u_null

    # Project null space torque into null space of Jacobian
    u += (torch.eye(joints, device=self.device).unsqueeze(0) - torch.transpose(self.j_eef, 1, 2) @ j_eef_inv) @ u_null

    # Apply torques
    self.eff_act[:,:joints] = u.squeeze(-1)
```

**Key formula:** `τ = J^T * F_task + (I - J^T * J_inv) * τ_null`
- Primary task: Position/orientation control
- Secondary task: Maintain preferred joint configuration

### 2.3 Direct Position Mode (`franka_slider_door.py:262-267`)

```python
elif self.cfg["env"]["driveMode"] == "pos":
    joints = self.franka_num_dofs - 2
    # Direct joint angle control (no IK)
    self.pos_act[:, :joints] = self.pos_act[:, :joints] + actions[:, 0:joints] * self.dt * self.action_speed_scale
    self.pos_act[:, :joints] = tensor_clamp(
        self.pos_act[:, :joints], self.franka_dof_lower_limits_tensor[:joints],
        self.franka_dof_upper_limits_tensor[:joints])
```

## 3. How to Make the Base Static

You have **TWO methods** to make the base static:

### Method 1: Change Configuration (Recommended)

Edit the YAML configuration file (e.g., `cfg/franka_open_lever_door.yaml:26`):

```yaml
model:
  mobile: False  # Change from True to False
```

**What this does:**
- Loads `franka_panda.urdf` instead of `franka_panda_slider.urdf`
- Reduces DOFs from 11 to 9 (removes 2 base joints)
- Base is completely fixed (no X/Y translation)

### Method 2: Modify URDF File (Advanced)

If you want to keep the mobile model but lock the base, modify `franka_panda_slider.urdf`:

Change the base joints from `type="prismatic"` to `type="fixed"`:

```xml
<!-- Original -->
<joint name="z_free_base1" type="prismatic">
    <axis xyz="1 0 0"/>
    <limit effort="50" lower="-1.0" upper="1.0" velocity="1.0"/>
</joint>

<!-- Modified to static -->
<joint name="z_free_base1" type="fixed">
    <!-- No axis or limits needed for fixed joints -->
</joint>
```

Repeat for `z_free_base2`.

**Note:** This method is more complex and requires understanding URDF structure. Method 1 is simpler and recommended.

## 4. Controller Configuration

Set the control mode in the YAML file (`cfg/franka_open_lever_door.yaml:69`):

```yaml
env:
  driveMode: "osc"  # Options: "ik", "osc", "pos"
```

**Control Mode Comparison:**

| Mode | Description | Converts Pose→Joints | Best For |
|------|-------------|---------------------|----------|
| `ik` | Inverse Kinematics | ✅ Yes (Jacobian) | Position control tasks |
| `osc` | Operational Space Control | ✅ Yes (Impedance) | Contact-rich tasks (door opening) |
| `pos` | Direct Joint Control | ❌ No (direct angles) | Joint-level control |

## 5. Summary: Joint Angle Determination Flow

```
Configuration (mobile: True/False)
    ↓
Load URDF (11 or 9 DOFs)
    ↓
Initialize DOF State Tensor
    ↓
Actions (End-effector pose)
    ↓
Controller (IK/OSC/Pos)
    ↓
Joint Angles (pos_act or eff_act)
    ↓
IsaacGym Simulation
    ↓
Updated DOF State (dof_state_tensor)
```

## Example: Making Base Static

**Before (mobile):**
```bash
python run.py --task=FrankaSliderDoor --cfg_env=cfg/franka_open_lever_door.yaml
# Uses 11 DOFs (2 base + 7 arm + 2 gripper)
```

**After (static):**

1. Edit `cfg/franka_open_lever_door.yaml`:
```yaml
model:
  mobile: False
```

2. Run:
```bash
python run.py --task=FrankaSliderDoor --cfg_env=cfg/franka_open_lever_door.yaml
# Uses 9 DOFs (7 arm + 2 gripper), base is fixed
```

The base will now remain at its initial position defined in `_franka_init_pose()` and won't be affected by control commands.
