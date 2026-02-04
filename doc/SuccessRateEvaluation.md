# Success Rate Evaluation Analysis

## Overview
This document analyzes the code responsible for evaluating the success rate of door opening tasks in the UniDoorManip system.

## Key Files Involved

### 1. **base_env.py** (Primary Success Evaluation)
Location: `/Simulation/env/base_env.py`

This is the main file that implements success rate calculation and tracking.

## Success Evaluation Mechanism

### Success Metrics Tracked

The system tracks multiple success criteria defined in `base_env.py:1143-1176`:

#### 1. **Handle Success** (`base_env.py:1144`)
```python
self.open_handle_success = (torch.abs(self.door_handle_dof_tensor[:, 0] - self.door_lower_limits_tensor[:, 1]) > 0.1)
```
- **Criteria**: The handle has moved more than 0.1 radians from its lower limit
- **Purpose**: Checks if the lever handle has been successfully pressed down

#### 2. **Door Opening Threshold** (`base_env.py:1145`)
```python
self.open_door = (torch.abs(self.door_dof_tensor[:, 0]) > 0.1)
```
- **Criteria**: Door has opened more than 0.1 radians
- **Purpose**: Basic check for any door movement

#### 3. **15-Degree Success** (`base_env.py:1146`)
```python
self.open_door_success_15 = (torch.abs(self.door_dof_tensor[:, 0]) > math.pi / 180 * 15)
```
- **Criteria**: Door opened ≥ 15 degrees
- **Purpose**: Minimum success threshold

#### 4. **30-Degree Success** (`base_env.py:1147`)
```python
self.open_door_success_30 = (torch.abs(self.door_dof_tensor[:, 0]) > math.pi / 180 * 30)
```
- **Criteria**: Door opened ≥ 30 degrees
- **Purpose**: Moderate success threshold

#### 5. **45-Degree Success** (`base_env.py:1148`)
```python
self.open_door_success_45 = (torch.abs(self.door_dof_tensor[:, 0]) > math.pi / 180 * 45)
```
- **Criteria**: Door opened ≥ 45 degrees
- **Purpose**: High success threshold

### Success Calculation Flow

#### Step 1: Update During Simulation (`base_env.py:741-742`)
```python
self.open_door_flag = ((self.door_handle_dof_tensor[:, 0] >= 0.75 *
                       (self.door_actor_dof_upper_limits_tensor[:, 1] -
                        self.door_actor_dof_lower_limits_tensor[:, 1]))
                      ).unsqueeze(1).repeat_interleave(3, dim=-1) | self.open_door_flag
self.cal_success()
```
- Called at every simulation step
- `open_door_flag`: Binary flag indicating when the handle has been pressed 75% of its range
- This flag is used in manipulation logic to switch from "pressing handle" to "pulling door" mode

#### Step 2: Calculate Success Rates (`base_env.py:1151-1155`)
```python
mean_open_handle_success = (self.open_handle_success * 1.0).mean()
mean_open_door = (self.open_door * 1.0).mean()
mean_success_15 = (self.open_door_success_15 * 1.0).mean()
mean_success_30 = (self.open_door_success_30 * 1.0).mean()
mean_success_45 = (self.open_door_success_45 * 1.0).mean()
```
- Calculates the mean success rate across all parallel environments
- Converts boolean tensors to floats (0.0 or 1.0) and computes mean
- Result is a percentage (0.0 to 1.0) of successful environments

#### Step 3: Track Best Performance (`base_env.py:1157-1166`)
```python
if(mean_open_door>self.d_5):
    self.d_5 = mean_open_door
if(mean_success_15>self.d_15):
    self.d_15 = mean_success_15
if(mean_success_30>self.d_30):
    self.d_30 = mean_success_30
if(mean_success_45>self.d_45):
    self.d_45 = mean_success_45
```
- Tracks the maximum success rate achieved during the episode
- Variables `d_5`, `d_15`, `d_30`, `d_45` initialized to 0 at `base_env.py:122-125`

#### Step 4: Display Results (`base_env.py:1167-1172`)
```python
print("mean open handle success", mean_open_handle_success)
print("mean open door", mean_open_door)
print("mean 15 success", mean_success_15)
print("mean 30 success", mean_success_30)
print("mean 45 success", mean_success_45)
```

## Data Collection Success Criteria

### Collection Flag (`base_env.py:797-802`)
When collecting expert demonstration data, success is determined by gripper state:

```python
finger_dist = self.lfinger_dof_tensor[:,0] + self.rfinger_dof_tensor[:,0]

if self.task == "rounddoor":
    collect_flag = (finger_dist >= 0.05) & (finger_dist < 0.07)
else:
    collect_flag = (finger_dist >= 0.01) & (finger_dist < 0.04)
```

**Criteria**:
- **Round door**: Gripper distance must be between 0.05 and 0.07 (wider grip)
- **Other doors**: Gripper distance must be between 0.01 and 0.04 (tighter grip)
- **Purpose**: Ensures the gripper maintained proper grasp throughout the task

### Data Collection Timing (`base_env.py:727-728, 793`)
```python
if (self.progress_buf[0] > self.cfg["env"]["start_index"]) & (self.progress_buf[0]%10==1):
    # Collect data every 10 steps

if(self.progress_buf[0] >= end_index):
    # Save data and evaluate success
```
- Starts collecting from `start_index` (default: 80)
- Collects every 10 steps
- Evaluates and saves at `end_index` (default: 230)

## Handle State Tracking

### Handle Down Detection (`base_env.py:771`)
```python
self.init_state_index = torch.where(
    (self.open_door_flag[:,0] == True) & (self.init_state_index[:] == 0),
    (self.progress_buf - self.start_index - 1)/10,
    self.init_state_index
)
```
- Records the timestep when the handle was first pressed down sufficiently
- Used for analyzing the manipulation timeline
- Saved in collected data as the "index" field

## Related Files

### Manipulation Control (`open_lever_door.py:22-51`)
The manipulation logic uses `open_door_flag` to switch strategies:

```python
for i in range(18):
    handle_q = self.env.door_handle_rigid_body_tensor[:, 3:7]
    rotate_dir = quat_axis(handle_q, axis=1)  # down direction
    open_dir = quat_axis(handle_q, axis=2)    # pull direction

    # Switch between pressing handle and pulling door
    pred_p = torch.where(self.env.open_door_flag,
                        cur_p + open_dir * step_size,    # Pull door
                        cur_p - rotate_dir * step_size)  # Press handle
```

### Controller Entry (`gt_pose.py:12-17`)
```python
def run(self, eval=False):
    goal_hand_pose = self.env.adjust_hand_pose
    self.manipulation.plan_pathway_gt_multi_dt(goal_hand_pose, eval)
```
- Entry point for running manipulation tasks
- `eval` parameter can enable evaluation mode

## Configuration

### Key Parameters (`franka_open_lever_door.yaml`)
- `model_test`: Enable/disable model testing mode (line 43)
- `start_index`: When to start collecting data (default: 80)
- `end_index`: When to stop and evaluate (default: 230)
- `maxEpisodeLength`: Maximum episode length (default: 192)

## Usage in Training/Testing

### During Expert Data Collection
1. Run simulation with expert controller
2. Every step: Update `open_door_flag` and call `cal_success()`
3. Print current success rates to console
4. At end: Filter successful trajectories using `collect_flag`
5. Save only successful demonstrations

### During Policy Evaluation
1. Load trained policy model
2. Run policy in simulation
3. Monitor success metrics in real-time via console output
4. Track maximum achieved success rates (`d_15`, `d_30`, `d_45`)

## Summary

The success evaluation system provides:
- **Multi-level thresholds**: 15°, 30°, 45° for granular evaluation
- **Real-time monitoring**: Updates every simulation step
- **Best performance tracking**: Records maximum success rates
- **Quality control**: Filters training data based on gripper state
- **Parallel evaluation**: Simultaneously evaluates across all environments (typically 100)

The primary metric for "task success" is typically the **30-degree threshold**, which represents a door opened wide enough for practical use.
