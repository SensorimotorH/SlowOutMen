## 通过ROS2通信实现MMK2控制的简单API

### 1. 简介

#### 1.1 MMK2_Controller

参考MMK2_mujoco_node的Subscribed topics，封装了以下功能：

- 控制mmk2底盘移动
- 控制mmk2头部移动
- 控制mmk2左臂移动
- 控制mmk2右臂移动
- 控制mmk2升降移动

```yaml
Subscribed topics:
 * /mmk2/cmd_vel [geometry_msgs/msg/Twist] 1 subscriber
 * /mmk2/head_forward_position_controller/commands [std_msgs/msg/Float64MultiArray] 1 subscriber
 * /mmk2/left_arm_forward_position_controller/commands [std_msgs/msg/Float64MultiArray] 1 subscriber
 * /mmk2/right_arm_forward_position_controller/commands [std_msgs/msg/Float64MultiArray] 1 subscriber
 * /mmk2/spine_forward_position_controller/commands [std_msgs/msg/Float64MultiArray] 1 subscriber
```

#### 1.2 MMK2_Receiver

参考MMK2_mujoco_node的Published topics，封装了以下功能：

- 接收头部相机的深度图像
- 接收头部相机的RGB图像
- 接收左臂相机的RGB图像
- 接收右臂相机的RGB图像
- 接收mmk2的全部关节参数（dim=19）

```yaml
Publishers:
  /mmk2/camera/head_camera/aligned_depth_to_color/image_raw: sensor_msgs/msg/Image
  /mmk2/camera/head_camera/color/image_raw: sensor_msgs/msg/Image
  /mmk2/camera/left_camera/color/image_raw: sensor_msgs/msg/Image
  /mmk2/camera/right_camera/color/image_raw: sensor_msgs/msg/Image
  /mmk2/joint_states: sensor_msgs/msg/JointState
```

### 2. 使用方法

#### 2.1 初始化控制器和接收器

首先需要安装s2r2025：

```bash
cd ./SIM2REAL-2025
pip install -e .
```

然后就可以在环境中的任意Python代码中导入相关模块：

```python
from s2r2025.simple_api.ros2_mmk2_api import MMK2_Controller, MMK2_Receiver
import rclpy
```

初始化ROS2和创建控制器与接收器：

```python
# 初始化ROS2节点
rclpy.init()

# 创建接收器（先创建接收器）
receiver = MMK2_Receiver()

# 创建控制器（需要传入接收器来获取关节状态）
controller = MMK2_Controller(receiver)
```

#### 2.2 底盘控制

```python
# 底盘移动速度控制
controller.set_move_forward_vel(0.5)  # 以0.5m/s前进
controller.set_move_backward_vel(0.3)  # 以0.3m/s后退
controller.stop_all()  # 停止移动

# 底盘移动位置控制
controller.set_move_forward_pos(1.0, 0.4)  # 前进1米，速度为0.4m/s
controller.set_move_backward_pos(0.5, 0.3)  # 后退0.5米，速度为0.3m/s

# 底盘转向控制
controller.set_turn_left_angle(90)  # 向左转90度
controller.set_turn_right_angle(45)  # 向右转45度
```

#### 2.3 头部控制

```python
# 头部俯仰角度控制
controller.set_head_pitch_angle(0.5)  # 设置头部俯仰角度为0.5rad

# 头部偏航角度控制
controller.set_head_yaw_angle(0.3)  # 设置头部偏航角度为0.3rad

# 同时控制头部俯仰和偏航
controller.set_head_position(0.3, 0.5)  # 设置头部偏航为0.3rad，俯仰为0.5rad
```

#### 2.4 升降台控制

```python
# 设置升降台高度
controller.set_head_slide_pos(0.5)  # 设置升降台高度为0.5m
```

#### 2.5 手臂控制

```python
# 设置左臂位置
left_arm_positions = [0.0, -0.166, 0.032, 0.0, 1.571, 2.223, 0.0]
controller.set_left_arm_pos(left_arm_positions)

# 设置右臂位置
right_arm_positions = [0.0, -0.166, 0.032, 0.0, -1.571, -2.223, 0.0]
controller.set_right_arm_pos(right_arm_positions)
```

#### 2.6 夹爪控制

```python
# 左臂夹爪控制
controller.set_left_arm_gripper_open()  # 完全打开左臂夹爪
controller.set_left_arm_gripper_close()  # 完全关闭左臂夹爪
controller.set_left_arm_gripper_open(0.5)  # 设置左臂夹爪开度为0.5

# 右臂夹爪控制
controller.set_right_arm_gripper_open()  # 完全打开右臂夹爪
controller.set_right_arm_gripper_close()  # 完全关闭右臂夹爪
controller.set_right_arm_gripper_close(0.8)  # 设置右臂夹爪关闭程度为0.8
```

#### 2.7 获取传感器数据

```python
# 获取头部相机RGB图像
head_rgb = receiver.get_head_rgb()
if head_rgb is not None:
    # 处理RGB图像数据
    image_height, image_width, _ = head_rgb.shape
    print(f"头部RGB图像尺寸: {image_width}x{image_height}")

# 获取头部相机深度图像
head_depth = receiver.get_head_depth()
if head_depth is not None:
    # 处理深度图像数据
    depth_height, depth_width = head_depth.shape
    print(f"头部深度图像尺寸: {depth_width}x{depth_height}")

# 获取左臂相机图像
left_rgb = receiver.get_left_arm_rgb()

# 获取右臂相机图像
right_rgb = receiver.get_right_arm_rgb()

# 获取关节状态
joint_states = receiver.get_joint_states()
if joint_states is not None:
    positions = joint_states["positions"]
    velocities = joint_states["velocities"]
    efforts = joint_states["efforts"]
    # 处理关节数据
```

#### 2.8 完整使用示例

```python
import time
import rclpy
from s2r2025.simple_api.ros2_mmk2_api import MMK2_Controller, MMK2_Receiver

# 初始化ROS2
rclpy.init()

try:
    # 初始化接收器
    receiver = MMK2_Receiver()
    time.sleep(1)  # 等待接收器获取初始数据
    
    # 初始化控制器
    controller = MMK2_Controller(receiver)
    
    # 执行任务示例：前进、抬头、打开夹爪
    controller.set_move_forward_pos(0.5, 0.3)  # 前进0.5米
    controller.set_head_pitch_angle(0.5)  # 抬头
    controller.set_left_arm_gripper_open()  # 打开左臂夹爪
    
    # 获取相机图像
    rgb_image = receiver.get_head_rgb()
    
    # 执行更多操作...

except Exception as e:
    print(f"发生错误: {e}")
    
finally:
    # 确保安全停止
    controller.stop_all()
    rclpy.shutdown()
```

* 请在使用完毕后调用 controller.stop_all() 和 rclpy.shutdown() 以确保机器人安全停止并正确关闭ROS2节点。
* 测试时如果遇到问题，在群里及时交流。