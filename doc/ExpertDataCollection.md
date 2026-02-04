# 数据收集流程详解

## 1. 启动流程

当你运行 `bash franka_slider_open_lever_door_collect_data.sh` 时，脚本会执行以下命令：

```bash
python run.py \
  --task=FrankaSliderDoor \
  --controller=GtPoseController \
  --manipulation=OpenLeverDoorManipulation \
  --collect_data=True \
  --collect_data_path="lever_handle_one/2" \
  --num_envs=100
```

## 2. 初始化阶段 (run.py)

代码执行顺序：
1. 创建环境 (parse_env) - 创建 FrankaSliderDoor 环境
2. 创建操作器 (parse_manipulation) - 创建 OpenLeverDoorManipulation 操作器
3. 创建控制器 (parse_controller) - 创建 GtPoseController 控制器
4. 运行控制器 (controller.run())

## 3. 专家数据生成 - 关键部分

### 3.1 生成目标抓取位置 (franka_slider_door.py:79)

`get_adjust_hand_pose()` 函数负责生成机械臂的目标位置：

- **基础位置**：从门把手的刚体位置获取 (door_handle_rigid_body_tensor)
- **旋转调整**：将末端执行器旋转到合适的抓取姿态（绕Y轴旋转90度）
- **位置随机化**（如果不是测试模式）：
  - 在把手Y方向添加随机偏移（±handle_range/5）
  - 添加小幅度随机旋转（±10度）
- **最终位置**：在把手前方留出夹爪长度的空间

### 3.2 执行专家轨迹 (open_lever_door.py:22)

`plan_pathway_gt_multi_dt()` 函数定义了三阶段操作：

#### 阶段1：移动到把手位置 (50步)
```python
for i in range(50):
    self.env.step(pose)  # pose是目标抓取位置
```

#### 阶段2：闭合夹爪抓取 (30步)
```python
pose[:, 0] -= gripper_length + 0.012  # X方向前移
for i in range(30):
    self.env.step(pose)
```

#### 阶段3：下压并开门 (18次，每次10步 = 180步)
```python
for i in range(18):
    # 计算运动方向
    rotate_dir = quat_axis(handle_q, axis=1)  # 下压方向
    open_dir = quat_axis(handle_q, axis=2)    # 开门方向

    # 根据把手状态切换运动模式
    if 把手已下压到位:
        pred_p = cur_p + open_dir * 0.06      # 拉门
    else:
        pred_p = cur_p - rotate_dir * 0.06    # 下压把手

    for j in range(10):
        self.env.step(pred_pose)
```

## 4. 数据收集过程 (base_env.py:726-866)

### 4.1 触发条件 (base_env.py:727)

在每个 `step()` 中，当满足条件时收集数据：
```python
if (progress_buf > start_index) and (progress_buf % 10 == 1):
    collect_data()
```
- 从第80步开始收集（默认start_index=80）
- 每10步收集一次数据

### 4.2 收集的数据内容 (base_env.py:751-862)

每次收集包括：

**1. 点云数据** (collectPC=True)
  - 从深度相机生成点云
  - 使用最远点采样下采样到4096个点
  - 相对于环境原点的坐标

**2. 力传感器数据** (collectForce=True)
  - 6维力/力矩传感器读数
  - 归一化处理

**3. 本体感知信息** (32维)
  - 机械臂关节位置 (11维，归一化到[-1,1])
  - 机械臂关节速度 (11维)
  - 机械臂基座位置 (3维)
  - 末端执行器位置 (3维)
  - 末端执行器四元数 (4维)

**4. 动作数据**
  - 相对位移：action = 目标位置 - 当前末端位置
  - 旋转矩阵：从四元数转换的3x3旋转矩阵

**5. 状态数据**
  - 门和把手的关节位置和速度（8维）
  - 使用差分：记录每步的变化量

**6. 其他信息**
  - 阻抗控制力
  - 全局目标位置
  - 把手下压完成的时间索引

### 4.3 数据保存 (base_env.py:793-866)

到达 end_index（230步）时：
- 筛选成功的环境：finger_dist ∈ [0.01, 0.04]（夹爪保持合适的闭合状态）
- 将所有收集的数据堆叠成张量
- 保存为 .pt 文件到 `../logs/_seed{}/lever_handle_one/2/data_{seed}.pt`

## 5. 数据格式

最终保存的字典包含：
```python
{
    "pc": [num_envs, num_steps, 4096, 3],           # 点云
    "hand_sensor_force": [num_envs, num_steps, 3],  # 力传感器
    "impedance_force": [num_envs, num_steps, 6],    # 阻抗力
    "proprioception_info": [num_envs, num_steps, 32], # 本体感知
    "action": [num_envs, num_steps, 3],             # 相对动作
    "rotate_matrix": [num_envs, num_steps, 3, 3],   # 旋转矩阵
    "state": [num_envs, num_steps, 8],              # 门状态差分
    "goal_global_tensor": [num_envs, 3],            # 全局目标
    "index": [num_envs],                            # 把手下压完成索引
    "collect_flag": [num_envs],                     # 成功标志
}
```

## 6. 关键配置参数 (franka_open_lever_door.yaml)

- `collectData: False` - 需要通过命令行传入 `--collect_data=True`
- `start_index: 80` - 从第80步开始收集
- `end_index: 230` - 到第230步停止
- `PointDownSampleNum: 4096` - 点云采样数量
- `num_envs: 100` - 100个并行环境同时收集数据

这样每次运行可以并行收集100个轨迹的数据，极大提高了数据收集效率！

## 总结

专家数据收集流程：
1. **启动阶段**：初始化100个并行环境
2. **专家轨迹**：执行预定义的三阶段操作（接近→抓取→开门）
3. **数据采集**：从第80步开始，每10步收集一次多模态数据
4. **数据保存**：在第230步筛选成功的轨迹并保存

这种并行化的数据收集方式使得系统能够高效地生成大量高质量的专家示范数据，用于后续的模仿学习训练。
