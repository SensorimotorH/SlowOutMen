# 位姿变换方法

## pose_transform_mujoco

只能在MuJoCo环境中运行，无法用于ROS2环境，因为无法通过MuJoCo获取物体位姿信息

### 核心功能类: PoseTransform

`PoseTransform`类提供了在MuJoCo环境中进行坐标变换和逆运动学计算的功能。

#### 初始化方法

- `__init__(model, data)`: 初始化坐标变换器，传入MuJoCo的模型和数据对象。

#### 位姿计算方法

- `get_relative_pose(model, data, body1_name, body2_name)`: 计算从body1到body2的相对位姿变换，返回4x4的变换矩阵。
- `get_T_OC(object_name, camera_name)`: 计算物体相对于相机的位姿变换 T_OC。
- `get_T_CB(camera_name, base_name)`: 计算相机相对于机械臂基座的位姿变换 T_CB。
- `get_T_OB(object_name, base_name, camera_name)`: 计算物体相对于机械臂基座的位姿变换 T_OB，可选通过相机中转计算。

#### 逆运动学方法

- `get_arm_joint_pose(pose3d, action, arm, q_ref, action_rot)`: 根据物体3D坐标计算机械臂关节角度。
  - `pose3d`: 物体在机械臂基座坐标系下的3D坐标[x, y, z]
  - `action`: 动作名称("pick", "carry", "look")或相对于机械臂基座坐标系旋转矩阵
  - `arm`: 机械臂选择("l"左臂, "r"右臂)
  - `q_ref`: 参考关节角度(可选)
  - `action_rot`: 额外的旋转矩阵(可选)

- `get_arm_joint_pose_full(pose6d, arm, q_ref, action_rot)`: 根据完整的6DOF位姿计算机械臂关节角度。
  - 注意: 此方法可能会有奇异解，谨慎使用!

- `get_arm_joint_pose_from_object(object_name, base_name, camera_name, action, arm, q_ref, action_rot)`: 根据物体名称直接计算机械臂关节角度。
  - 结合了位姿获取和逆运动学计算的便捷方法

#### 特别说明

提供了三种`action_rot`的选择：

- 默认值：`action_rot: Optional[np.ndarray] = np.eye(3)`
- None：`action_rot = Rotation.from_euler('zyx', [0, -1.1, 0]).as_matrix()`
- 手动传入`action_rot`参数

建议使用前两种，第一种（单位矩阵）会让夹爪直上直下抓取，第二种（绕y旋转矩阵）会让夹爪有一定角度抓取

### 测试内容说明

main函数使用`GraspAppleTask`作为测试环境，测试包含以下内容：

1. **相对位姿计算测试**:

   - 计算物体相对于相机的位姿(T_OC)
   - 计算相机相对于机械臂基座的位姿(T_CB)
   - 通过相机中转计算物体相对于机械臂基座的位姿(T_OB)
   - 直接计算物体相对于机械臂基座的位姿(T_OB_direct)

2. **逆运动学计算测试**:

   - 使用预设动作计算关节角度并保存渲染图像
   - 使用完整6D位姿计算关节角度并保存渲染图像
   - 直接从物体名称计算关节角度并保存渲染图像

这些测试展示了如何使用PoseTransform类的各种方法，从获取相对位姿到计算机械臂关节角度的完整流程。测试结果会保存为三张图像，分别展示不同方法计算出的机械臂姿态。

![test_1.png](./assets/test_1.png)

![test_2.png](./assets/test_2.png)

![test_3.png](./assets/test_3.png)

## pose_transform_ros2

在ROS2环境中实现位姿变换，需要通过在MuJoCo环境中获取变换矩阵进行标定
