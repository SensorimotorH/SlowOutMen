import time
import numpy as np
import rclpy
import mujoco
import threading
from typing import List, Optional, Dict
from scipy.spatial.transform import Rotation

from SlowOutMen.Communication.ros2_communication import TopicPublisher, TopicSubscriber


# 控制参数宏定义
DEFAULT_LINEAR_VEL = 0.5   # 默认线速度 (m/s)
DEFAULT_ANGULAR_VEL = 0.2  # 默认角速度 (rad/s)
MAX_LINEAR_VEL = 1.0       # 最大线速度 (m/s)
MAX_ANGULAR_VEL = 0.5      # 最大角速度 (rad/s)

# 头部角度范围
HEAD_YAW_MIN = -0.5        # 头部偏航最小值
HEAD_YAW_MAX = 0.5         # 头部偏航最大值
HEAD_PITCH_MIN = -0.16     # 头部俯仰最小值
HEAD_PITCH_MAX = 1.18      # 头部俯仰最大值

# 升降台高度范围
SPINE_HEIGHT_MIN = -0.04   # 升降台最低高度
SPINE_HEIGHT_MAX = 0.87    # 升降台最高高度

# 夹爪开合范围
GRIPPER_OPEN = 0.0         # 夹爪完全打开
GRIPPER_CLOSE = 1.0        # 夹爪完全关闭

# 控制常量
SECOND_PER_DEGREE = 0.025   # 每度旋转所需的秒数
SECOND_PER_METER = 0.5     # 每米移动所需的秒数

# 默认臂位置（用于初始化,如果无法获得当前关节信息）
DEFAULT_LEFT_ARM_POSITION = [-0.00, -0.17, 0.03, -0.00, 1.57, 2.22, 0.00]
DEFAULT_RIGHT_ARM_POSITION = [0.00, -0.17, 0.03, 0.00, -1.57, -2.22, 0.00]
DEFAULT_HEAD_POSITION = [0.0, 0.0]
DEFAULT_SPINE_HEIGHT = 0.0

# 机器人基座初始方向（世界坐标四元数）
POSITIVE_DERECTION = [0.707, 0,  0, -0.707]

# 添加平滑控制相关参数
CONTROL_RATE = 24.0  # 控制频率 (Hz)
DELTA_T = 1.0 / CONTROL_RATE  # 控制周期 (s)
DEFAULT_MOVE_RATIO = 0.3  # 默认移动比例
MAX_JOINT_SPEED = 0.5        # 最大关节速度 (rad/s)
MAX_JOINT_ACCELERATION = 0.2  # 最大关节加速度 (rad/s²)
MIN_MOVEMENT_TIME = 0.5      # 最小运动时间 (s)


class MMK2_Receiver:
    """
    MMK2机器人传感器数据接收器
    提供获取摄像头图像、关节状态等功能
    """

    def __init__(self, save_dir: Optional[str] = None, topics: List[str] = None, debug: bool = False):
        """
        初始化接收器
        :param save_dir: 数据保存目录,默认为None时会自动生成
        """
        self.sub = None
        self.data_dir = None

        try:
            self.sub = TopicSubscriber(
                save_dir=save_dir, topics=topics, debug=debug)
            self.data_dir = self.sub.save_dir
            self._record_initial_joint_states()
            self._stop_flag = threading.Event()
            self.sub_thread = threading.Thread(
                target=self._run_subscriber, daemon=False)
            self.sub_thread.start()
        except Exception as e:
            print(f"初始化接收器失败: {e}")
            raise

    def _run_subscriber(self):
        """
        运行 ROS2 订阅回调的独立线程
        """
        while not self._stop_flag.is_set() and rclpy.ok():
            rclpy.spin_once(self.sub)

    def _record_initial_joint_states(self):
        """
        记录初始关节状态
        """
        retry_count = 0
        max_retries = 5

        while retry_count < max_retries:
            joint_states = self.get_joint_states()
            if joint_states is not None and 'positions' in joint_states:
                self.initial_joint_states = joint_states.copy()
                print("成功记录初始关节状态")
                return True

            print(f"尝试获取初始关节状态 ({retry_count+1}/{max_retries})...")
            time.sleep(0.5)
            retry_count += 1

        print("警告：无法获取初始关节状态")
        return False

    def get_initial_joint_states(self) -> Optional[Dict]:
        """
        获取初始关节状态
        :return: 包含初始关节位置的字典,若无数据则返回None
        """
        return self.initial_joint_states

    def get_clock(self) -> Optional[str]:
        """
        获取系统时钟信息
        :return: 时钟时间戳字符串,若无数据则返回None
        """
        if self.sub and 'clock' in self.sub.ros_info:
            return self.sub.ros_info['clock']
        return None

    def get_head_rgb_camera_info(self) -> Optional[Dict]:
        """
        获取头部RGB相机信息
        :return: 相机参数信息对象,若无数据则返回None
        """
        if self.sub and 'head_camera_info' in self.sub.ros_info:
            return self.sub.ros_info['head_camera_info']
        return None

    def get_head_depth_camera_info(self) -> Optional[Dict]:
        """
        获取头部深度相机信息
        :return: 相机参数信息对象,若无数据则返回None
        """
        if self.sub and 'head_depth_camera_info' in self.sub.ros_info:
            return self.sub.ros_info['head_depth_camera_info']
        return None

    def get_left_arm_camera_info(self) -> Optional[Dict]:
        """
        获取左臂相机信息
        :return: 相机参数信息对象,若无数据则返回None
        """
        if self.sub and 'left_camera_info' in self.sub.ros_info:
            return self.sub.ros_info['left_camera_info']
        return None

    def get_right_arm_camera_info(self) -> Optional[Dict]:
        """
        获取右臂相机信息
        :return: 相机参数信息对象,若无数据则返回None
        """
        if self.sub and 'right_camera_info' in self.sub.ros_info:
            return self.sub.ros_info['right_camera_info']
        return None

    def get_odom(self) -> Optional[Dict]:
        """
        获取里程计信息
        :return: 包含位置、方向和速度信息的字典,若无数据则返回None
        """
        if self.sub and 'odom' in self.sub.ros_info:
            return self.sub.ros_info['odom']
        return None

    def get_taskinfo(self) -> Optional[str]:
        """
        获取任务信息
        :return: 任务信息字符串,若无数据则返回None
        """
        if self.sub and 'taskinfo' in self.sub.ros_info:
            return self.sub.ros_info['taskinfo']
        return None

    def get_gameinfo(self) -> Optional[str]:
        """
        获取游戏信息
        :return: 游戏信息字符串,若无数据则返回None
        """
        if self.sub and 'gameinfo' in self.sub.ros_info:
            return self.sub.ros_info['gameinfo']
        return None

    def get_head_depth(self) -> Optional[np.ndarray]:
        """
        获取头部深度图像
        :return: 深度图像数组,若无数据则返回None
        """
        if self.sub and 'head_depth' in self.sub.ros_info:
            return self.sub.ros_info['head_depth']
        return None

    def get_head_rgb(self) -> Optional[np.ndarray]:
        """
        获取头部RGB图像
        :return: RGB图像数组,若无数据则返回None
        """
        if self.sub and 'head_rgb' in self.sub.ros_info:
            return self.sub.ros_info['head_rgb']
        return None

    def get_left_arm_rgb(self) -> Optional[np.ndarray]:
        """
        获取左臂摄像头RGB图像
        :return: RGB图像数组,若无数据则返回None
        """
        if self.sub and 'left_rgb' in self.sub.ros_info:
            return self.sub.ros_info['left_rgb']
        return None

    def get_right_arm_rgb(self) -> Optional[np.ndarray]:
        """
        获取右臂摄像头RGB图像
        :return: RGB图像数组,若无数据则返回None
        """
        if self.sub and 'right_rgb' in self.sub.ros_info:
            return self.sub.ros_info['right_rgb']
        return None

    def get_joint_states(self) -> Optional[Dict]:
        """
        获取关节状态
        :return: 包含关节位置、速度、力矩的字典,若无数据则返回None
        """
        return self.sub.ros_info['joint_states']

    def get_base_position(self) -> Optional[Dict]:
        """
        获取底盘位置
        :return: 包含底盘位置和方向的字典,若无数据则返回None
        """
        if self.get_joint_states() is not None:
            base_x = self.get_odom()['position']['x']
            base_y = self.get_odom()['position']['y']
            base_z = self.get_odom()['position']['z']
            base_rot = self.get_odom()['orientation']
            base_rot_list = [base_rot['w'], base_rot['x'],
                             base_rot['y'], base_rot['z']]
            base_angle = 90.0 + np.rad2deg(Rotation.from_quat(
                [base_rot['x'], base_rot["y"], base_rot["z"], base_rot["w"]]).as_euler('zyx')[0])
            base_angle = (base_angle + 180) % 360 - 180
            return {
                'position': [base_x, base_y, base_z],
                'orientation_list': base_rot_list,
                'orientation': base_rot,
                'x': base_x,
                'y': base_y,
                'z': base_z,
                'angle': base_angle
            }
        return None

    def get_arm_position(self) -> Optional[Dict]:
        """
        获取机械臂位置
        :return: 包含机械臂位置的字典,包含夹爪,若无数据则返回None
        """
        if self.get_joint_states() is not None:
            left_arm = self.get_joint_states()['positions'][3:10]
            right_arm = self.get_joint_states()['positions'][10:17]
            return {
                'left_arm': left_arm,
                'right_arm': right_arm
            }
        return None

    def get_head_position(self) -> Optional[Dict]:
        """
        获取头部位置
        :return: 包含头部位置的字典,若无数据则返回None
        """
        if self.get_joint_states() is not None:
            head_position = self.get_joint_states()['positions'][0:3]
            return {
                'slide': head_position[0],
                'pitch': head_position[1],
                'yaw': head_position[2]
            }
        return None

    def stop(self):
        """
        停止订阅线程并记录数据
        """
        self.sub.record_data()
        self._stop_flag.set()
        if self.sub_thread.is_alive():
            self.sub_thread.join()


class MMK2_Controller:
    """
    MMK2机器人控制器
    提供底盘移动、头部、手臂、夹爪、升降台等控制功能
    """

    def __init__(self, receiver: MMK2_Receiver = None, save_dir: Optional[str] = None, topics: List[str] = None, debug: bool = False) -> None:
        """
        初始化控制器,创建发布者
        :param receiver: 可选的接收器对象,用于获取当前关节状态
        """
        print(f"初始化控制器，接收器状态：{receiver is not None}")
        self.pub = TopicPublisher(save_dir=save_dir, debug=debug)

        # 检查接收器
        if receiver is None:
            raise ValueError("接收器对象不能为空")

        if not isinstance(receiver, MMK2_Receiver):
            raise ValueError(f"接收器对象类型错误: {type(receiver)}")

        self.receiver = receiver  # 用于获取当前关节状态
        self.left_arm_position = DEFAULT_LEFT_ARM_POSITION.copy()
        self.right_arm_position = DEFAULT_RIGHT_ARM_POSITION.copy()

        # 保存初始状态
        self.initial_left_arm_position = np.zeros(7)
        self.initial_right_arm_position = np.zeros(7)
        self.initial_head_position = np.zeros(2)
        self.initial_spine_height = 0.0

        # 尝试从接收器获取初始状态
        self._load_initial_states()

        if not self._update_joint_states():
            print("警告：无法获取初始关节状态，使用默认值")

        # 添加平滑控制相关变量
        self.delta_t = DELTA_T
        self.current_control = {
            'head': np.zeros(2),
            'spine': np.zeros(1),
            'left_arm': np.zeros(7),
            'right_arm': np.zeros(7)
        }
        self.target_control = self.current_control.copy()
        self.joint_move_ratio = DEFAULT_MOVE_RATIO

    def _load_initial_states(self) -> bool:
        """
        从接收器加载初始状态
        """
        try:
            initial_states = self.receiver.get_initial_joint_states()
            if initial_states and 'positions' in initial_states:
                positions = initial_states["positions"]
                if len(positions) >= 17:
                    # 包含夹爪
                    self.initial_left_arm_position = positions[3:10].copy()
                    # 包含夹爪
                    self.initial_right_arm_position = positions[10:17].copy()
                    # 如果头部和升降台位置也在关节状态中，也可以记录
                    if len(positions) >= 3:
                        self.initial_head_position = [
                            positions[0], positions[1]]  # 头部yaw和pitch
                        self.initial_spine_height = positions[2]  # 升降台高度

                    print("成功加载初始关节状态")
                    return True

            print("无法从接收器获取完整的初始状态，将使用默认值")
            return False
        except Exception as e:
            print(f"加载初始状态失败: {e}")
            return False

    def reset_all(self) -> bool:
        """
        将机器人重置到初始位置
        """
        print("正在将机器人重置到初始位置...")

        try:
            # 先停止所有运动
            self.stop_all()
            time.sleep(0.5)

            # 重置头部位置
            self.set_head_position(
                DEFAULT_HEAD_POSITION[0], DEFAULT_HEAD_POSITION[1])
            time.sleep(0.5)

            # 重置升降台高度
            self.set_head_slide_position(DEFAULT_SPINE_HEIGHT)
            time.sleep(0.5)

            # 重置左臂位置
            self.set_left_arm_position(DEFAULT_LEFT_ARM_POSITION)
            time.sleep(0.5)

            # 重置右臂位置
            self.set_right_arm_position(DEFAULT_RIGHT_ARM_POSITION)
            time.sleep(0.5)

            print("机器人已重置到初始位置")
            return True

        except Exception as e:
            print(f"重置机器人失败: {e}")
            return False

    def reset_head(self) -> bool:
        """
        将机器人头部位姿重置到初始位置
        """
        print("正在将机器人头部位姿重置到初始位置...")

        try:
            # 先停止所有运动
            self.stop_all()
            time.sleep(0.5)

            # 重置头部位置
            self.set_head_position(
                DEFAULT_HEAD_POSITION[0], DEFAULT_HEAD_POSITION[1])
            time.sleep(0.5)

            print("头部位姿已重置")
            return True

        except Exception as e:
            print(f"头部位姿重置失败: {e}")
            return False

    def reset_slide(self) -> bool:
        """
        将机器人头部高度重置到初始位置
        """
        print("正在将机器人头部高度重置到初始位置...")

        try:
            # 先停止所有运动
            self.stop_all()
            time.sleep(0.5)

            # 重置升降台高度
            self.set_head_slide_position(DEFAULT_SPINE_HEIGHT)
            time.sleep(0.5)

            print("机器人头部高度已重置到初始高度")
            return True

        except Exception as e:
            print(f"头部高度重置失败: {e}")
            return False

    def reset_arms(self) -> bool:
        """
        将机器人机械臂重置到初始位置
        """
        print("正在将机器人机械臂重置到初始位置...")

        try:
            # 先停止所有运动
            self.stop_all()
            time.sleep(0.5)

            # 重置左臂位置
            self.set_left_arm_position(DEFAULT_LEFT_ARM_POSITION)
            time.sleep(0.5)

            # 重置右臂位置
            self.set_right_arm_position(DEFAULT_RIGHT_ARM_POSITION)
            time.sleep(0.5)

            print("机械臂已重置到初始位置")
            return True

        except Exception as e:
            print(f"重置机械臂失败: {e}")
            return False

    def _update_joint_states(self) -> bool:
        """
        从接收器更新关节状态
        :return: 更新是否成功
        """
        if self.receiver is None:
            print("警告：接收器对象为空")
            return False

        try:
            joint_states = self.receiver.get_joint_states()
            if joint_states is None:
                print("警告：无法获取关节状态")
                return False

            # 解析关节状态,提取手臂相关关节位置
            positions = joint_states["positions"]
            if len(positions) >= 17:  # 确保有足够的关节数据
                # 假设左臂和右臂关节各占6个位置,加上夹爪
                self.left_arm_position = positions[3:10]
                self.right_arm_position = positions[10:17]
                return True
            return False
        except Exception as e:
            print(f"更新关节状态失败: {e}")
            return False

    def _flush_print(self, message: str) -> None:
        """实时刷新打印信息"""
        print(message, end='', flush=True)

    def _smooth_control_step(self, current: float, target: float, step: float, acceleration_factor: float = 0.2) -> float:
        """
        改进的平滑控制函数，增加加速度限制
        :param current: 当前值
        :param target: 目标值 
        :param step: 步长
        :param acceleration_factor: 加速度因子，控制初始加速度
        :return: 下一个值
        """
        # 计算当前位置与目标位置的差值
        diff = target - current

        # 使用sigmoid函数实现平滑加速
        progress = abs(diff) / (abs(diff) + acceleration_factor)
        actual_step = step * progress

        # 限制实际步长
        if abs(diff) < actual_step:
            return target
        elif diff > 0:
            return current + actual_step
        else:
            return current - actual_step

    def _smooth_control_update(self):
        """
        更新平滑控制
        """
        # 头部控制平滑
        for i in range(2):
            self.current_control['head'][i] = self._smooth_control_step(
                self.current_control['head'][i],
                self.target_control['head'][i],
                self.joint_move_ratio * self.delta_t
            )

        # 升降台控制平滑
        self.current_control['spine'][0] = self._smooth_control_step(
            self.current_control['spine'][0],
            self.target_control['spine'][0],
            self.joint_move_ratio * self.delta_t
        )

        # 左臂控制平滑
        for i in range(7):
            self.current_control['left_arm'][i] = self._smooth_control_step(
                self.current_control['left_arm'][i],
                self.target_control['left_arm'][i],
                self.joint_move_ratio * self.delta_t
            )

        # 右臂控制平滑
        for i in range(7):
            self.current_control['right_arm'][i] = self._smooth_control_step(
                self.current_control['right_arm'][i],
                self.target_control['right_arm'][i],
                self.joint_move_ratio * self.delta_t
            )

    def _move_base(self, target_pos: float = 0.0,
                   target_angle: float = 0.0,
                   pos_tolerance: float = 0.01,
                   angle_tolerance: float = 0.1,
                   default_linear_vel: float = DEFAULT_LINEAR_VEL,
                   default_angular_vel: float = DEFAULT_ANGULAR_VEL,
                   pos_threshold: float = 0.2,
                   angle_threshold: float = 5.0,
                   stability_time: float = 1.0,
                   execute_time_threshold: float = 60.0,
                   flush_print: bool = False) -> None:
        """
        底盘闭环控制方法，支持同时进行前进和转向控制

        Args:
            target_pos: 目标前进距离(米)，正数前进，负数后退
            target_angle: 目标转向角度(弧度)，正数左转，负数右转
            pos_tolerance: 位置误差容限(米)
            angle_tolerance: 角度误差容限(角度)
            default_linear_vel: 最大线速度(m/s)
            default_angular_vel: 最大角速度(rad/s)
            pos_threshold: 位置误差小于该值时，才使用误差进行速度控制，否则用最大速度
            angle_threshold: 角度误差小于该值时，才使用误差进行速度控制，否则用最大角速度
            stability_time: 稳定时间(秒)，在此时间内位置和角度误差都小于容限才算到达目标
            execute_time_threshold: 执行时间阈值
            flush_print: 是否实时打印进度
        """
        # 获取初始状态
        initial_odom = self.receiver.get_odom()

        initial_pos = np.array([
            initial_odom['position']['x'],
            initial_odom['position']['y'],
            initial_odom['position']['z']
        ])
        initial_quat = np.array([
            initial_odom['orientation']['w'],
            initial_odom['orientation']['x'],
            initial_odom['orientation']['y'],
            initial_odom['orientation']['z']
        ])

        R_initial = np.zeros(9)
        mujoco.mju_quat2Mat(R_initial, initial_quat)
        R_initial = R_initial.reshape(3, 3)
        forward_direction = R_initial[:, 0]

        # 计算目标状态
        target_projection = target_pos
        if target_angle != 0:
            angle_quat = np.array(
                [np.cos(target_angle/2), 0, 0, np.sin(target_angle/2)])
            target_quat = np.zeros(4)
            mujoco.mju_mulQuat(target_quat, initial_quat, angle_quat)
        else:
            target_quat = initial_quat.copy()

        start_stable_time = None
        start_execute_time = time.monotonic()
        try:
            while time.monotonic() - start_execute_time < execute_time_threshold:
                # 获取当前状态
                current_odom = self.receiver.get_odom()
                current_pos = np.array([
                    current_odom['position']['x'],
                    current_odom['position']['y'],
                    current_odom['position']['z']
                ])
                current_quat = np.array([
                    current_odom['orientation']['w'],
                    current_odom['orientation']['x'],
                    current_odom['orientation']['y'],
                    current_odom['orientation']['z']
                ])

                # 计算位置误差
                displacement = current_pos - initial_pos
                current_projection = displacement.dot(forward_direction)
                pos_error = target_projection - current_projection

                # 计算角度误差
                error_quat = np.zeros(4)
                inv_current_quat = np.array([current_quat[0], -current_quat[1],
                                             -current_quat[2], -current_quat[3]])
                mujoco.mju_mulQuat(error_quat, target_quat, inv_current_quat)
                angle_error = 2 * np.arccos(np.clip(error_quat[0], -1.0, 1.0))
                if error_quat[3] < 0:
                    angle_error = -angle_error

                # 检查是否到达目标
                pos_reached = abs(pos_error) < pos_tolerance
                angle_reached = abs(angle_error) < angle_tolerance
                current_stable = pos_reached and angle_reached

                # 退出条件判断
                current_time = time.monotonic()
                if current_stable:
                    if start_stable_time is None:
                        start_stable_time = current_time
                    elif current_time - start_stable_time >= stability_time:
                        self.stop_all()
                        break
                else:
                    start_stable_time = None

                # 计算控制量
                linear_vel = (np.clip(pos_error, -default_linear_vel, default_linear_vel)
                              if abs(pos_error) < pos_threshold else np.sign(pos_error) * default_linear_vel)
                angular_vel = (np.clip(angle_error, -default_angular_vel, default_angular_vel)
                               if abs(angle_error) < angle_threshold else np.sign(angle_error) * default_angular_vel)

                # 发布速度指令
                if not pos_reached:
                    linear_vel = linear_vel
                else:
                    linear_vel = 0

                if not angle_reached:
                    angular_vel = angular_vel
                else:
                    angular_vel = 0
                self.pub.publish_cmd_vel(linear_vel, angular_vel)

                if flush_print:
                    self._flush_print(
                        f"\rMovement Progress | "
                        f"Position: {(target_pos-pos_error):.5f}/{target_pos:.5f}m | "
                        f"Angle: {np.rad2deg(target_angle - angle_error):.3f}\u00b0/{np.rad2deg(target_angle):.3f}\u00b0 | "
                        f"Linear_vel: {linear_vel:.3f}m/s | Angular_vel: {angular_vel:.3f}rad/s"
                        f"pos_reached:{pos_reached} | angle_reached:{angle_reached}"
                    )
                time.sleep(DELTA_T)

        finally:
            self.stop_all()
            if flush_print:
                print("\nMovement completed")

    def _publish_spine_position(self, spine_value):
        """
        发布升降台位置
        :param spine_value: 包含一个浮点数的列表
        """
        try:
            if not isinstance(spine_value, (list, np.ndarray)):
                spine_value = [float(spine_value)]
            elif isinstance(spine_value, np.ndarray):
                spine_value = spine_value.tolist()

            self.pub.publish_spine_position(spine_value)
        except Exception as e:
            print(f"Failed to publish spine position: {str(e)}")

    def set_move_forward_position(self, relative_position: float,
                                  pos_tolerance: float = 0.01,
                                  flush_print: bool = False) -> None:
        """
        前进指定距离（带反馈控制）
        :param relative_position: 前进距离(m)
        :param pos_tolerance: 位置容差(m)
        :param flush_print: 实时打印进度
        """
        if relative_position < 0:
            self.set_move_backward_position(
                target_pos=-relative_position,
                pos_tolerance=pos_tolerance,
                flush_print=flush_print
            )

        self._move_base(
            target_pos=relative_position,
            pos_tolerance=pos_tolerance,
            flush_print=flush_print
        )

    def set_move_backward_position(self, relative_position: float,
                                   pos_tolerance: float = 0.01,
                                   flush_print: bool = False) -> None:
        """
        后退指定距离（带反馈控制）
        :param relative_position: 后退距离(m)
        :param pos_tolerance: 位置容差(m)
        :param flush_print: 实时打印进度
        """
        if relative_position < 0:
            self.set_move_forward_position(
                target_pos=-relative_position,
                pos_tolerance=pos_tolerance,
                flush_print=flush_print
            )

        self._move_base(
            target_pos=-relative_position,
            pos_tolerance=pos_tolerance,
            flush_print=flush_print,
        )

    def set_turn_left_angle(self, angle: float,
                            angle_tolerance: float = 0.1,
                            flush_print: bool = False) -> None:
        """
        左转指定角度（带反馈控制）
        :param angle: 旋转角度(度)
        :param angle_tolerance: 角度容差(度)
        :param flush_print: 实时打印进度
        """
        self._move_base(
            target_angle=np.deg2rad(angle),
            angle_tolerance=np.deg2rad(angle_tolerance),
            flush_print=flush_print
        )

    def set_turn_right_angle(self, angle: float,
                             angle_tolerance: float = 0.1,
                             flush_print: bool = False) -> None:
        """
        右转指定角度（带反馈控制）
        :param angle: 旋转角度(度)
        :param angle_tolerance: 角度容差(度)
        :param flush_print: 实时打印进度
        """
        self._move_base(
            target_angle=-np.deg2rad(angle),
            angle_tolerance=np.deg2rad(angle_tolerance),
            flush_print=flush_print
        )

    def set_head_yaw_angle(self, position: float, execute_time_threshold: float = 5.0) -> None:
        """
        平滑设置头部偏航角度
        :param position: 偏航角度,范围约为[-0.5, 0.5]
        """
        try:
            # 限制角度范围
            position = float(max(HEAD_YAW_MIN, min(position, HEAD_YAW_MAX)))

            # 保持pitch角度不变,只改变yaw
            current_pitch = self.current_control['head'][1]
            self.target_control['head'] = np.array(
                [position, current_pitch], dtype=np.float64)

            # 记录开始时间
            start_time = time.monotonic()

            # 连续更新直到达到目标位置或超时
            while time.monotonic() - start_time < execute_time_threshold:
                # 执行平滑更新
                self._smooth_control_update()

                # 发布新的位置指令
                head_positions = self.current_control['head'].tolist()
                self.pub.publish_head_position(head_positions)

                # 检查是否达到目标位置
                if np.allclose(self.current_control['head'][0],
                               self.target_control['head'][0],
                               atol=0.01):
                    break

                # 控制更新频率
                time.sleep(1/CONTROL_RATE)

        except Exception as e:
            print(f"平滑设置头部偏航角度失败: {str(e)}")
        finally:
            # 确保最后一次发布目标位置
            self.pub.publish_head_position(
                self.target_control['head'].tolist())

    def set_head_pitch_angle(self, position: float, execute_time_threshold: float = 5.0) -> None:
        """
        平滑设置头部俯仰角度
        :param position: 俯仰角度,范围约为[-0.16, 1.18]
        """
        try:
            # 限制角度范围
            position = float(
                max(HEAD_PITCH_MIN, min(position, HEAD_PITCH_MAX)))

            # 保持yaw角度不变,只改变pitch
            current_yaw = self.current_control['head'][0]
            self.target_control['head'] = np.array(
                [current_yaw, position], dtype=np.float64)

            # 记录开始时间
            start_time = time.monotonic()

            # 连续更新直到达到目标位置或超时
            while time.monotonic() - start_time < execute_time_threshold:
                # 执行平滑更新
                self._smooth_control_update()

                # 发布新的位置指令
                head_positions = self.current_control['head'].tolist()
                self.pub.publish_head_position(head_positions)

                # 检查是否达到目标位置
                if np.allclose(self.current_control['head'][1],
                               self.target_control['head'][1],
                               atol=0.01):
                    break

                # 控制更新频率
                time.sleep(1/CONTROL_RATE)

        except Exception as e:
            print(f"平滑设置头部俯仰角度失败: {str(e)}")
        finally:
            # 确保最后一次发布目标位置
            self.pub.publish_head_position(
                self.target_control['head'].tolist())

    def set_head_position(self, pitch: float, yaw: float, execute_time_threshold: float = 5.0) -> None:
        """
        平滑设置头部偏航和俯仰角度
        :param yaw: 偏航角度,范围约为[-0.5, 0.5]
        :param pitch: 俯仰角度,范围约为[-0.16, 1.18]
        """
        try:
            # 限制角度范围
            yaw = float(max(HEAD_YAW_MIN, min(yaw, HEAD_YAW_MAX)))
            pitch = float(max(HEAD_PITCH_MIN, min(pitch, HEAD_PITCH_MAX)))

            # 设置目标位置
            self.target_control['head'] = np.array(
                [yaw, pitch], dtype=np.float64)

            # 记录开始时间
            start_time = time.monotonic()

            # 连续更新直到达到目标位置或超时
            while time.monotonic() - start_time < execute_time_threshold:
                # 执行平滑更新
                self._smooth_control_update()

                # 发布新的位置指令
                head_positions = self.current_control['head'].tolist()
                self.pub.publish_head_position(head_positions)

                # 检查是否达到目标位置
                if np.allclose(self.current_control['head'],
                               self.target_control['head'],
                               atol=0.01):
                    break

                # 控制更新频率
                time.sleep(1/CONTROL_RATE)

        except Exception as e:
            print(f"平滑设置头部位置失败: {str(e)}")
        finally:
            # 确保最后一次发布目标位置
            self.pub.publish_head_position(
                self.target_control['head'].tolist())

    def set_head_slide_position(self, position: float, execute_time_threshold: float = 5.0) -> None:
        """
        平滑设置升降台高度
        :param position: 高度位置,范围为[-0.04, 0.87]
        """
        try:
            # 限制高度范围
            position = float(
                max(SPINE_HEIGHT_MIN, min(position, SPINE_HEIGHT_MAX)))

            # 设置目标位置
            self.target_control['spine'] = np.array(
                [position], dtype=np.float64)

            # 记录开始时间
            start_time = time.monotonic()

            # 连续更新直到达到目标位置或超时
            while time.monotonic() - start_time < execute_time_threshold:
                # 执行平滑更新
                self._smooth_control_update()

                # 发布新的位置指令
                spine_value = [float(self.current_control['spine'][0])]
                self._publish_spine_position(spine_value)

                # 检查是否达到目标位置
                if np.allclose(self.current_control['spine'],
                               self.target_control['spine'],
                               atol=0.01):
                    break

                # 控制更新频率
                time.sleep(1/CONTROL_RATE)

        except Exception as e:
            print(f"平滑设置升降台高度失败: {str(e)}")
        finally:
            # 确保最后一次发布目标位置
            self._publish_spine_position(
                [float(self.target_control['spine'][0])])

    def set_head_slide_position_smooth(self, position: float, execute_time_smooth: float = 10.0, execute_time_step: float = 0.1) -> None:
        """
        平滑设置升降台高度
        :param position: 高度位置,范围为[-0.04, 0.87]
        :param execute_time_smooth: 平滑执行时间(秒)
        :param execute_time_step: 平滑执行时间步长(秒)
        """
        current_slide_pose = self.receiver.get_head_position()['slide']
        target_slide_pose = position
        for i in range(int(execute_time_smooth / execute_time_step)):
            self.set_head_slide_position((1 - i / int(execute_time_smooth / execute_time_step)) * current_slide_pose + (
                i / int(execute_time_smooth / execute_time_step)) * target_slide_pose)
            time.sleep(execute_time_step)

    def set_left_arm_position(self, positions: List[float], execute_time_threshold: float = 5.0,
                              start_ratio: float = 0.1, check_threshold: float = 0.5) -> None:
        """
        平滑设置左臂关节位置，增加缓启动和位置检查
        :param positions: 目标关节位置列表
        :param execute_time_threshold: 运动持续时间(秒)
        :param start_ratio: 初始运动比例，控制启动速度
        :param check_threshold: 位置检查阈值
        """
        self._update_joint_states()
        # 补充夹爪位置
        if len(positions) == 6:
            positions.append(self.left_arm_position[6])
        
        positions = [float(i) for i in positions]
            

        try:
            # 检查目标位置与当前位置的差异是否过大
            position_diff = np.linalg.norm(
                np.array(positions) - np.array(self.left_arm_position))
            if position_diff > check_threshold:
                print(f"警告: 左臂运动幅度较大 (差异: {position_diff:.3f})")

            # 设置初始目标为当前位置到最终目标位置的一小段距离
            current_position = np.array(self.left_arm_position)
            target_position = np.array(positions)

            # 缓启动阶段目标
            initial_target = current_position + \
                (target_position - current_position) * start_ratio
            self.target_control['left_arm'] = initial_target

            # 记录开始时间
            start_time = time.monotonic()

            # 缓启动阶段
            while time.monotonic() - start_time < execute_time_threshold * start_ratio:
                self._smooth_control_update()

                # 发布新的位置指令
                arm_positions = self.current_control['left_arm'].tolist()
                self.pub.publish_left_arm_position(arm_positions)

                time.sleep(1/CONTROL_RATE)

            # 设置最终目标位置
            self.target_control['left_arm'] = target_position

            # 主运动阶段
            while time.monotonic() - start_time < execute_time_threshold:
                self._smooth_control_update()

                arm_positions = self.current_control['left_arm'].tolist()
                self.pub.publish_left_arm_position(arm_positions)

                # 检查是否达到目标位置
                if np.allclose(self.current_control['left_arm'], target_position, atol=0.01):
                    break

                time.sleep(1/CONTROL_RATE)

        except Exception as e:
            print(f"设置左臂位置失败: {str(e)}")
        finally:
            # 确保最终位置正确
            self.pub.publish_left_arm_position(positions)

    def set_right_arm_position(self, positions: List[float], execute_time_threshold: float = 5.0,
                               start_ratio: float = 0.1, check_threshold: float = 0.5) -> None:
        """
        平滑设置右臂关节位置，增加缓启动和位置检查
        :param positions: 目标关节位置列表
        :param execute_time_threshold: 运动持续时间(秒)
        :param start_ratio: 初始运动比例，控制启动速度
        :param check_threshold: 位置检查阈值
        """
        self._update_joint_states()
        # 补充夹爪位置
        if len(positions) == 6:
            positions.append(self.right_arm_position[6])
            
        positions = [float(i) for i in positions]

        try:
            # 检查目标位置与当前位置的差异是否过大
            position_diff = np.linalg.norm(
                np.array(positions) - np.array(self.right_arm_position))
            if position_diff > check_threshold:
                print(f"警告: 右臂运动幅度较大 (差异: {position_diff:.3f})")

            # 设置初始目标为当前位置到最终目标位置的一小段距离
            current_position = np.array(self.right_arm_position)
            target_position = np.array(positions)

            # 缓启动阶段目标
            initial_target = current_position + \
                (target_position - current_position) * start_ratio
            self.target_control['right_arm'] = initial_target

            # 记录开始时间
            start_time = time.monotonic()

            # 缓启动阶段
            while time.monotonic() - start_time < execute_time_threshold * start_ratio:
                self._smooth_control_update()

                # 发布新的位置指令
                arm_positions = self.current_control['right_arm'].tolist()
                self.pub.publish_right_arm_position(arm_positions)

                time.sleep(1/CONTROL_RATE)

            # 设置最终目标位置
            self.target_control['right_arm'] = target_position

            # 主运动阶段
            while time.monotonic() - start_time < execute_time_threshold:
                self._smooth_control_update()

                arm_positions = self.current_control['right_arm'].tolist()
                self.pub.publish_right_arm_position(arm_positions)

                # 检查是否达到目标位置
                if np.allclose(self.current_control['right_arm'], target_position, atol=0.01):
                    break

                time.sleep(1/CONTROL_RATE)

        except Exception as e:
            print(f"设置右臂位置失败: {str(e)}")
        finally:
            # 确保最终位置正确
            self.pub.publish_right_arm_position(positions)

    def set_arm_position(self, lft_positions: List[float], rgt_positions: List[float], execute_time_threshold: float = 5.0,
                         start_ratio: float = 0.1, check_threshold: float = 0.5) -> None:
        """
        改进的双臂关节位置控制，增加缓启动和位置检查
        :param lft_positions: 左臂目标关节位置列表
        :param rgt_positions: 右臂目标关节位置列表
        :param execute_time_threshold: 运动持续时间(秒)
        :param start_ratio: 初始运动比例，控制启动速度
        :param check_threshold: 位置检查阈值
        """
        self._update_joint_states()

        # 补充夹爪位置
        if len(lft_positions) == 6:
            lft_positions.append(self.left_arm_position[6])
        if len(rgt_positions) == 6:
            rgt_positions.append(self.right_arm_position[6])
            
        lft_positions = [float(i) for i in lft_positions]
        rgt_positions = [float(i) for i in rgt_positions]

        try:
            # 检查目标位置与当前位置的差异是否过大
            lft_diff = np.linalg.norm(
                np.array(lft_positions) - np.array(self.left_arm_position))
            rgt_diff = np.linalg.norm(
                np.array(rgt_positions) - np.array(self.right_arm_position))

            if lft_diff > check_threshold or rgt_diff > check_threshold:
                print(f"警告: 运动幅度较大 (左臂: {lft_diff:.3f}, 右臂: {rgt_diff:.3f})")

            # 设置初始目标为当前位置到最终目标位置的一小段距离
            current_lft = np.array(self.left_arm_position)
            current_rgt = np.array(self.right_arm_position)
            target_lft = np.array(lft_positions)
            target_rgt = np.array(rgt_positions)

            # 缓启动阶段
            initial_target_lft = current_lft + \
                (target_lft - current_lft) * start_ratio
            initial_target_rgt = current_rgt + \
                (target_rgt - current_rgt) * start_ratio

            # 设置目标位置
            self.target_control['left_arm'] = initial_target_lft
            self.target_control['right_arm'] = initial_target_rgt

            # 记录开始时间
            start_time = time.monotonic()

            # 缓启动阶段
            while time.monotonic() - start_time < execute_time_threshold * start_ratio:
                self._smooth_control_update()

                # 发布新的位置指令
                rgt_arm_positions = self.current_control['right_arm'].tolist()
                self.pub.publish_right_arm_position(rgt_arm_positions)
                lft_arm_positions = self.current_control['left_arm'].tolist()
                self.pub.publish_left_arm_position(lft_arm_positions)

                time.sleep(1/CONTROL_RATE)

            # 设置最终目标位置
            self.target_control['left_arm'] = target_lft
            self.target_control['right_arm'] = target_rgt

            # 主运动阶段
            while time.monotonic() - start_time < execute_time_threshold:
                self._smooth_control_update()

                rgt_arm_positions = self.current_control['right_arm'].tolist()
                self.pub.publish_right_arm_position(rgt_arm_positions)
                lft_arm_positions = self.current_control['left_arm'].tolist()
                self.pub.publish_left_arm_position(lft_arm_positions)

                if np.allclose(self.current_control['right_arm'], target_rgt, atol=0.01) and \
                        np.allclose(self.current_control['left_arm'], target_lft, atol=0.01):
                    break

                time.sleep(1/CONTROL_RATE)

        except Exception as e:
            print(f"设置机械臂位置失败: {str(e)}")
        finally:
            # 确保最终位置正确
            self.pub.publish_left_arm_position(lft_positions)
            self.pub.publish_right_arm_position(rgt_positions)

    def set_left_arm_gripper(self, position: float = GRIPPER_OPEN) -> None:
        """
        打开左手夹爪,保持其他关节位置不变
        :param position: 夹爪开度,0.0为完全打开
        """
        position = max(GRIPPER_OPEN, min(position, GRIPPER_CLOSE))

        # 尝试获取当前关节状态
        self._update_joint_states()

        # 创建一个新的关节位置列表,只修改夹爪位置
        gripper_cmd = self.left_arm_position.copy()
        gripper_cmd[6] = position  # 修改夹爪位置

        self.left_arm_position = gripper_cmd.copy()  # 更新内部状态
        self.pub.publish_left_arm_position(gripper_cmd)

    def set_right_arm_gripper(self, position: float = GRIPPER_OPEN) -> None:
        """
        打开右手夹爪,保持其他关节位置不变
        :param position: 夹爪开度,0.0为完全打开
        """
        position = max(GRIPPER_OPEN, min(position, GRIPPER_CLOSE))

        # 尝试获取当前关节状态
        self._update_joint_states()

        # 创建一个新的关节位置列表,只修改夹爪位置
        gripper_cmd = self.right_arm_position.copy()
        gripper_cmd[6] = position  # 修改夹爪位置

        self.right_arm_position = gripper_cmd.copy()  # 更新内部状态
        self.pub.publish_right_arm_position(gripper_cmd)

    def stop_all(self) -> None:
        """
        停止所有运动
        """
        self.pub.publish_cmd_vel(0.0, 0.0)


def main() -> None:
    rclpy.init()
    receiver = None
    controller = None
    try:
        # 只创建一次接收器
        print("初始化MMK2接收器...")
        receiver = MMK2_Receiver()
        time.sleep(1)

        # 验证接收器状态
        if receiver is None:
            raise RuntimeError("接收器初始化失败")

        print(f"接收器数据目录: {receiver.data_dir}")

        # 创建控制器
        print("\n初始化MMK2控制器...")
        controller = MMK2_Controller(receiver)

        while rclpy.ok():

            controller.set_move_backward_position(0.4)
            controller.set_head_pitch_angle(1.5)
            controller.set_head_yaw_angle(1.5)

            controller.set_arm_position(
                rgt_positions=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                lft_positions=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            )
            controller.set_arm_position(
                rgt_positions=[0., 0., 0., 0., 0., 0., 0.],
                lft_positions=[0., 0., 0., 0., 0., 0., 0.]
            )

            controller.stop_all()

    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 清理资源
        if controller:
            try:
                controller.stop_all()
                print("已停止所有运动")
            except Exception as e:
                print(f"停止运动失败: {e}")

        # 关闭节点
        if receiver and receiver.sub:
            try:
                receiver.sub.destroy_node()
                print("已关闭接收器节点")
            except Exception as e:
                print(f"关闭节点失败: {e}")

        rclpy.shutdown()


if __name__ == "__main__":
    main()
