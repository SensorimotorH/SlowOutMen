import numpy as np
import time
import sys
import mujoco
from typing import List, Union, Any
from SlowOutMen.Communication.mujoco_mmk2_receiver_api import MMK2_Receiver

# 控制参数宏定义
DEFAULT_LINEAR_VEL = 0.5   # 默认线速度 (m/s)
DEFAULT_ANGULAR_VEL = 0.4  # 默认角速度 (rad/s)
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

# 默认速度
DEFAULT_HEAD_VEL = 0.5
DEFAULT_ARM_VEL = 0.5

# 关节和传感器索引映射
JOINT_INDICES = {
    # 底层控制关节索引
    'wheel_forward': 0,        # 底盘前进/后退
    'wheel_turn': 1,           # 底盘转向
    'slide': 2,                # 升降台
    'head_yaw': 3,             # 头部偏航
    'head_pitch': 4,           # 头部俯仰
    'left_arm': slice(5, 11),  # 左臂6个关节
    'left_gripper': 11,        # 左手夹爪
    'right_arm': slice(12, 18),    # 右臂6个关节
    'right_gripper': 18,       # 右手夹爪

    # 传感器数据索引
    'base_position': slice(57, 60),      # 基座位置 (x, y, z)
    'base_orientation': slice(60, 64),    # 基座朝向 (四元数)
    'base_linear_vel': slice(64, 67),     # 基座线速度
    'base_angular_vel': slice(67, 70),    # 基座角速度
    'base_acceleration': slice(70, 73),   # 基座加速度
    'left_arm_endpoint': slice(73, 76),   # 左臂末端位置
    'left_gripper_state': slice(76, 80),  # 左手夹爪状态
    'right_arm_endpoint': slice(80, 83),  # 右臂末端位置
    'right_gripper_state': slice(83, 87)  # 右手夹爪状态
}

AVAILABLE_PUBLICATION_TOPICS = [
    'cmd_vel',
    'head'
    'left_arm',
    'right_arm',
    'spine'
]


class MMK2_Controller:
    """
    在 MuJoCo 环境下的MMK2机器人控制器  
    提供底盘移动、头部、手臂、夹爪、升降台等控制功能  
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, receiver: MMK2_Receiver) -> None:
        """
        初始化控制器  

        Args:
            model (mujoco.MjModel): MuJoCo 模型对象
            data (mujoco.MjData): MuJoCo 数据对象
            receiver (MMK2_Receiver): MMK2接收器对象

        Raises:
            ValueError: 当接收器对象为空时抛出
        """
        self.model = model
        self.data = data

        if receiver is None:
            raise ValueError("接收器对象不能为空")

        self.receiver = receiver
        self.left_arm_position = np.array([0.0]*7)
        self.right_arm_position = np.array([0.0]*7)

        self.initial_left_arm_position = np.zeros(7)
        self.initial_right_arm_position = np.zeros(7)
        self.initial_head_position = np.zeros(2)
        self.initial_spine_height = 0.0

        self._load_initial_states()

        if not self._update_joint_states():
            print("警告:无法获取初始关节状态,使用默认值")

        self.njq = 28
        self.njctrl = 19
        self.arm_action_init_position = {
            "pick": {
                "l": np.array([0.223,  0.21, 1.07055]),
                "r": np.array([0.223, -0.21, 1.07055]),
            },
        }
        self.lft_arm_target_pose = self.arm_action_init_position["pick"]["l"].copy(
        )
        self.lft_end_euler = np.zeros(3)
        self.rgt_arm_target_pose = self.arm_action_init_position["pick"]["r"].copy(
        )
        self.rgt_end_euler = np.zeros(3)

        self.timestep = model.opt.timestep
        self.control_freq = 24
        self.steps_per_control = int(1.0 / (self.control_freq * self.timestep))
        self.decimation = 4

        self.use_pid = False
        self.pid_params = {
            'Kp': 100.0,  # 比例系数
            'Ki': 0.0,    # 积分系数
            'Kd': 10.0,   # 微分系数
            'max_ctrl': 1.0,  # 控制输出限幅
        }
        self.error_integrals = {}  # 用于存储积分误差

        self.control_ranges = self._update_control_ranges()
        self.default_action = np.zeros(19)
        self.stop_all()

    def _load_initial_states(self) -> bool:
        """
        从接收器加载初始状态
        :return: 更新是否成功 
        """
        try:
            initial_states = self.receiver.get_initial_joint_states()
            if initial_states and 'positions' in initial_states:
                positions = initial_states["positions"]
                if len(positions) >= 19:
                    self.initial_left_arm_position = np.array(positions[5:12])
                    self.initial_right_arm_position = np.array(
                        positions[12:19])
                    if len(positions) >= 3:
                        self.initial_head_position = np.array(positions[3:5])
                        self.initial_spine_height = positions[2]
                    print("成功加载初始关节状态")
                    return True

            print("无法从接收器获取完整的初始状态,将使用默认值")
            return False
        except Exception as e:
            print(f"加载初始状态失败: {e}")
            return False

    def _update_joint_states(self) -> bool:
        """
        从接收器更新关节状态  
        :return: 更新是否成功  
        """
        if self.receiver is None:
            print("警告:接收器对象为空")
            return False

        try:
            joint_states = self.receiver.get_joint_states()
            if joint_states is None:
                print("警告:无法获取关节状态")
                return False

            positions = joint_states["positions"]
            if len(positions) >= 19:
                self.left_arm_position = np.array(positions[5:12])
                self.right_arm_position = np.array(positions[12:19])
                return True
            return False
        except Exception as e:
            print(f"更新关节状态失败: {e}")
            return False

    def _update_control_ranges(self) -> dict[str, Any]:
        """
        根据模型更新控制范围
        :return: 控制范围字典
        """
        ctrl_range = self.model.actuator_ctrlrange[:self.njctrl]
        return {
            'LINEAR_VEL_RANGE': [-abs(ctrl_range[0][1]), abs(ctrl_range[0][1])],
            'ANGULAR_VEL_RANGE': [-abs(ctrl_range[1][1]), abs(ctrl_range[1][1])],
            'SPINE_HEIGHT_RANGE': [ctrl_range[2][0], ctrl_range[2][1]],
            'HEAD_YAW_RANGE': [ctrl_range[3][0], ctrl_range[3][1]],
            'HEAD_PITCH_RANGE': [ctrl_range[4][0], ctrl_range[4][1]],
            'ARM_JOINT_RANGES': ctrl_range[5:18],
            'GRIPPER_RANGE': [ctrl_range[18][0], ctrl_range[18][1]]
        }

    def _get_pid_control(self, error: np.ndarray, joint_index: Union[int, slice]) -> Any:
        """
        简化的控制器,支持PID和简单比例控制

        Args:
            error: 误差值
            joint_index: 关节索引

        Returns:
            控制输出值
        """
        if isinstance(joint_index, slice):
            joint_index = (joint_index.start,
                           joint_index.stop, joint_index.step)

        if joint_index not in self.error_integrals:
            self.error_integrals[joint_index] = 0.0

        self.error_integrals[joint_index] += error * self.timestep
        derivative = error / self.timestep

        control = (self.pid_params['Kp'] * error +
                   self.pid_params['Ki'] * self.error_integrals[joint_index] +
                   self.pid_params['Kd'] * derivative)

        return np.clip(control, -self.pid_params['max_ctrl'], self.pid_params['max_ctrl'])

    def _update_control(self, action: np.ndarray) -> None:
        """
        更新控制器的控制动作,核心控制逻辑,将控制动作应用到机器人模型中

        Args:
            action (np.ndarray): 控制动作数组(dim=19)
        """
        self.data.ctrl[:self.njctrl] = np.clip(
            action[:self.njctrl], self.model.actuator_ctrlrange[:self.njctrl, 0], self.model.actuator_ctrlrange[:self.njctrl, 1])

    def _step_simulation(self, action: List[float] = None, steps: int = None, render: bool = True) -> None:
        """
        执行仿真步骤,使用指定的控制动作

        Args:
            action (List[float], optional): 控制动作数组,默认为None
            steps (int, optional): 步数,默认为None,改写为self.decimation=4
            render (bool, optional): 是否渲染,默认为True
        """
        if steps is None:
            steps = self.decimation

        if action is None:
            action = self.default_action

        action = np.array(action)
        for _ in range(steps):
            self._update_control(action)
            mujoco.mj_step(self.model, self.data)
            if render:
                self.receiver.render()

    def _flush_print(self, content: str) -> None:
        """
        刷新打印输出

        Args:
            content (str): 输出内容
        """
        sys.stdout.write(content)
        sys.stdout.flush()

    def _flush_print_process(self, indices: Union[int, slice], current: np.ndarray, target: np.ndarray) -> None:
        """
        刷新打印输出,显示控制进度,只适用于_move_joint方法

        Args:
            indices (Union[int, slice]): 控制的关节索引
            current (np.ndarray): 当前关节位置
            target (np.ndarray): 目标关节位置
        """
        if np.ndim(current) == 0:
            current = np.round(current, 3)
        else:
            current = [round(x, 3) for x in np.array(current)]

        if np.ndim(target) == 0:
            target = np.round(target, 3)
        else:
            target = [round(x, 3) for x in np.array(target)]

        if isinstance(indices, int):
            self._flush_print(
                f"\rqpos[{indices}]: {current} -> {target}")
        else:
            self._flush_print(
                f"\rqpos[{indices.start}:{indices.stop}]: {current} -> {target}")

    def _get_current_pos(self, indices: Union[int, slice]) -> np.ndarray:
        """
        获取当前关节位置

        Args:
            indices (Union[int, slice]): 关节索引

        Returns:
            np.ndarray: 当前关节位置
        """
        return np.array(self.receiver.get_joint_states()['positions'][indices]).copy()

    def _get_current_action(self) -> np.ndarray:
        """
        获取当前控制动作

        Returns:
            np.ndarray: _description_
        """
        return np.array(self.receiver.get_joint_states()['positions']).copy()

    def _move_joint(self, joint_name: Union[str, int],
                    target: Union[float, np.ndarray],
                    tolerance: float = 1e-2,
                    flush_print: bool = True) -> None:
        """
        更新后的关节控制方法,使用action列表进行控制

        Args:
            joint_name (Union[str, int]): 关节名称或索引
            target (Union[float, np.ndarray]): 目标位置
            tolerance (float, optional): 误差容限,默认为1e-2
            flush_print (bool, optional): 是否实时打印进度,默认为True
        """
        if isinstance(joint_name, str):
            indices = JOINT_INDICES[joint_name]
        else:
            indices = joint_name

        if np.isscalar(target):
            target = np.array([target])

        current = self._get_current_pos(indices)
        action = self._get_current_action()

        if self.use_pid:
            while np.linalg.norm(target - current) > tolerance:
                error = target - current
                control_value = self._get_pid_control(error, indices)
                action[indices] = control_value
                self._step_simulation(action, 1)
                current = self._get_current_pos(indices)

                if flush_print:
                    self._flush_print_process(indices, current, target)

            if flush_print:
                print("")

        else:
            while np.linalg.norm(target - current) > tolerance:
                action[indices] = target
                self._step_simulation(action)
                current = self._get_current_pos(indices)

                if flush_print:
                    self._flush_print_process(indices, current, target)

            if flush_print:
                print("\nReached target position")

    def _move_base(self, target_pos: float = 0.0,
                   target_angle: float = 0.0,
                   pos_tolerance: float = 0.01,
                   angle_tolerance: float = 0.1,
                   max_linear_vel: float = MAX_LINEAR_VEL,
                   max_angular_vel: float = MAX_ANGULAR_VEL,
                   pos_threshold: float = 0.2,
                   angle_threshold: float = 5.0,
                   stability_time: float = 0.5,
                   flush_print: bool = True) -> None:
        """
        底盘控制方法,支持同时进行前进和转向控制

        Args:
            forward_target: 前进距离(米)
            turn_target: 转向角度(弧度)
            pos_tolerance: 位置的误差容限
            angle_tolerance: 角度的误差容限
            flush_print: 是否实时打印进度
        """
        # 获取初始状态
        angle_tolerance = np.deg2rad(angle_tolerance)
        initial_state = self.receiver.get_base_position_orientation()
        initial_pos = np.array(initial_state["position"])
        initial_quat = np.array(initial_state["orientation"])

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

        action = self._get_current_action()

        if self.use_pid:
            while True:
                current_state = self.receiver.get_base_position_orientation()
                current_pos = np.array(current_state["position"])
                current_quat = np.array(current_state["orientation"])

                # 计算位置误差
                displacement = current_pos - initial_pos
                current_projection = displacement.dot(forward_direction)
                pos_error = target_projection - current_projection

                # 计算角度误差
                error_quat = mujoco.mju_mulQuat(target_quat,
                                                [current_quat[0], -current_quat[1], -current_quat[2], -current_quat[3]])
                angle_error = 2 * np.arccos(np.clip(error_quat[0], -1.0, 1.0))
                if error_quat[3] < 0:
                    angle_error = -angle_error

                # 检查是否到达目标
                pos_reached = abs(pos_error) < pos_tolerance
                angle_reached = abs(angle_error) < angle_tolerance
                if (target_angle == 0 and pos_reached) or (pos_reached and angle_reached):
                    break

                # PID控制计算
                if not pos_reached:
                    action[JOINT_INDICES['wheel_forward']] = self._get_pid_control(
                        pos_error, 'wheel_forward')
                else:
                    action[JOINT_INDICES['wheel_forward']] = 0

                if not angle_reached:
                    action[JOINT_INDICES['wheel_turn']] = self._get_pid_control(
                        angle_error, 'wheel_turn')
                else:
                    action[JOINT_INDICES['wheel_turn']] = 0

                self._step_simulation(action, 1)

                if flush_print:
                    self._flush_print(
                        f"\rtarget_pos: {target_pos:.3f}m, target_angle: {target_angle}°, pos_error: {pos_error:.3f}m, angle_error: {np.rad2deg(angle_error):.1f}°")
        else:
            start_stable_time = None
            while True:
                current_state = self.receiver.get_base_position_orientation()
                current_pos = np.array(current_state["position"])
                current_quat = np.array(current_state["orientation"])

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
                        break
                else:
                    start_stable_time = None

                # 计算控制量
                linear_vel = (np.clip(pos_error, -max_linear_vel, max_linear_vel)
                              if abs(pos_error) < pos_threshold else np.sign(pos_error) * max_linear_vel)
                angular_vel = (np.clip(angle_error, -max_angular_vel, max_angular_vel)
                               if abs(angle_error) < angle_threshold else np.sign(angle_error) * max_angular_vel)

                if not pos_reached:
                    action[JOINT_INDICES['wheel_forward']
                           ] = linear_vel
                else:
                    action[JOINT_INDICES['wheel_forward']] = 0

                if not angle_reached:
                    action[JOINT_INDICES['wheel_turn']] = angular_vel
                else:
                    action[JOINT_INDICES['wheel_turn']] = 0

                self._step_simulation(action)

                if flush_print:
                    self._flush_print(
                        f"\rMovement Progress | "
                        f"Position: {current_projection:.3f}/{target_pos:.3f}m | "
                        f"Angle: {np.rad2deg(target_angle - angle_error):.3f}\u00b0/{np.rad2deg(target_angle):.3f}\u00b0"
                    )

        if flush_print:
            print("\nReached target position")

    def set_move_forward_pos(self, relative_position: float) -> None:
        """
        设置前进距离,使用专门的底盘控制方法

        Args:
            relative_position (float): 前进距离,单位为米
            velocity (float, optional): 前进速度,单位为米/秒. 默认为 DEFAULT_LINEAR_VEL
        """
        self._move_base(target_pos=relative_position)

    def set_move_backward_pos(self, relative_position: float) -> None:
        """
        设置后退距离,使用专门的底盘控制方法

        Args:
            relative_position (float): 后退距离,单位为米
            velocity (float, optional): 后退速度,单位为米/秒. 默认为 DEFAULT_LINEAR_VEL
        """
        self._move_base(target_pos=-relative_position)

    def set_turn_left_angle(self, angle: float) -> None:
        """
        向左转动指定角度,使用专门的底盘控制方法

        Args:
            angle (float): 转动角度,单位为度
        """
        self._move_base(target_angle=-np.deg2rad(angle))

    def set_turn_right_angle(self, angle: float) -> None:
        """
        向右转动指定角度,使用专门的底盘控制方法

        Args:
            angle (float): 转动角度,单位为度
        """
        self._move_base(target_angle=np.deg2rad(angle))

    def set_head_slide_pos(self, position: float) -> None:
        """
        设置升降台高度, 使用 qpos 和 qvel 反馈控制  
        :param position: 高度位置, 范围为 [SPINE_HEIGHT_MIN, SPINE_HEIGHT_MAX]  
        """
        position = max(SPINE_HEIGHT_MIN, min(position, SPINE_HEIGHT_MAX))
        self._move_joint('slide', position)

    def set_head_yaw_angle(self, position: float) -> None:
        """
        设置头部偏航角度,使用 qpos 和 qvel 反馈控制

        Args:
            position (float): 偏航角度,范围为 [HEAD_YAW_MIN, HEAD_YAW_MAX]
        """
        position = max(HEAD_YAW_MIN, min(position, HEAD_YAW_MAX))
        self._move_joint('head_yaw', position)

    def set_head_pitch_angle(self, position: float) -> None:
        """
        设置头部俯仰角度,使用 qpos 和 qvel 反馈控制

        Args:
            position (float): 俯仰角度,范围为 [HEAD_PITCH_MIN, HEAD_PITCH_MAX]
        """
        position = max(HEAD_PITCH_MIN, min(position, HEAD_PITCH_MAX))
        self._move_joint('head_pitch', position)

    def set_head_position(self, yaw: float, pitch: float) -> None:
        """
        同时设置头部偏航和俯仰角度,使用 qpos 和 qvel 反馈控制

        Args:
            yaw (float): 偏航角度,范围为 [HEAD_YAW_MIN, HEAD_YAW_MAX]
            pitch (float): 俯仰角度,范围为 [HEAD_PITCH_MIN, HEAD_PITCH_MAX]
        """
        self.set_head_yaw_angle(yaw)
        self.set_head_pitch_angle(pitch)

    def set_left_arm_pos(self, positions: List[float]) -> None:
        """
        设置左臂关节位置,使用 qpos 和 qvel 反馈控制

        Args:
            positions (List[float]): 包含6个关节角度或7个(最后一个为夹爪)的浮点数列表

        Raises:
            ValueError: 当输入列表长度不为6或7时抛出
        """
        self._update_joint_states()
        if len(positions) == 7:
            target = positions
            self.left_arm_position = np.array(positions)
        elif len(positions) == 6:
            target = self.data.qpos[5:12].copy()
            target[:6] = positions
            self.left_arm_position[:6] = positions
        else:
            raise ValueError("左臂位置应包含6个或7个值(6个关节+1个夹爪)")
        self._move_joint(
            'left_arm', target[:6])
        self._move_joint(
            'left_gripper', target[6])

    def set_left_arm_gripper_open(self, position: float = GRIPPER_OPEN) -> None:
        """
        打开左手夹爪,使用 qpos 和 qvel 反馈控制

        Args:
            position (float, optional): 夹爪开度,0.0为完全打开,1.0为完全关闭. 默认为 GRIPPER_OPEN
        """
        position = max(GRIPPER_OPEN, min(position, GRIPPER_CLOSE))
        self._update_joint_states()
        self.left_arm_position[6] = position
        self._move_joint('left_gripper', position)

    def set_left_arm_gripper_close(self, position: float = GRIPPER_CLOSE) -> None:
        """
        关闭左手夹爪, 使用 qpos 和 qvel 反馈控制, 保持其他关节位置不变  
        :param position: 夹爪开度, 1.0 为完全关闭  
        """
        position = max(GRIPPER_OPEN, min(position, GRIPPER_CLOSE))
        self._update_joint_states()
        self.left_arm_position[6] = position
        self._move_joint('left_gripper', position)

    def set_right_arm_pos(self, positions: List[float]) -> None:
        """
        设置右臂关节位置,使用 qpos 和 qvel 反馈控制

        Args:
            positions (List[float]): 包含6个关节角度或7个(最后一个为夹爪)的浮点数列表

        Raises:
            ValueError: 当输入列表长度不为6或7时抛出
        """
        self._update_joint_states()
        if len(positions) == 7:
            target = positions
            self.right_arm_position = np.array(positions)
        elif len(positions) == 6:
            target = self.data.qpos[12:19].copy()
            target[:6] = positions
            self.right_arm_position[:6] = positions
        else:
            raise ValueError("右臂位置应包含6个或7个值(6个关节+1个夹爪)")
        self._move_joint(
            'right_arm', target[:6])
        self._move_joint(
            'right_gripper', target[6])

    def set_right_arm_gripper_open(self, position: float = GRIPPER_OPEN) -> None:
        """
        打开右手夹爪,使用 qpos 和 qvel 反馈控制

        Args:
            position (float, optional): 夹爪开度,0.0为完全打开,1.0为完全关闭. 默认为 GRIPPER_OPEN
        """
        position = max(GRIPPER_OPEN, min(position, GRIPPER_CLOSE))
        self._update_joint_states()
        self.right_arm_position[6] = position
        self._move_joint('right_gripper', position)

    def set_right_arm_gripper_close(self, position: float = GRIPPER_CLOSE) -> None:
        """
        关闭右手夹爪, 使用 qpos 和 qvel 反馈控制, 保持其他关节位置不变  
        :param position: 夹爪开度, 1.0 为完全关闭  
        """
        position = max(GRIPPER_OPEN, min(position, GRIPPER_CLOSE))
        self._update_joint_states()
        self.right_arm_position[6] = position
        self._move_joint('right_gripper', position)

    def stop_all(self) -> None:
        """
        停止所有运动
        将所有受控关节的速度(qvel)置零,并执行一个控制周期的仿真
        """
        for i in range(self.njctrl):
            self.data.ctrl[i] = 0
            self.data.qvel[i] = 0
        action = np.zeros(self.njctrl)
        self._step_simulation(action)

    def reset(self) -> bool:
        """
        将机器人重置到初始位置

        Returns:
            bool: 重置是否成功
        """
        print("正在将机器人重置到初始位置...")

        try:
            # 先停止所有运动
            self.stop_all()
            time.sleep(1)

            # 重置头部位置
            self.set_head_position(
                DEFAULT_HEAD_POSITION[0], DEFAULT_HEAD_POSITION[1])

            # 重置升降台高度
            self.set_head_slide_pos(DEFAULT_SPINE_HEIGHT)

            # 重置左臂位置
            self.set_left_arm_pos(DEFAULT_LEFT_ARM_POSITION)

            # 重置右臂位置
            self.set_right_arm_pos(DEFAULT_RIGHT_ARM_POSITION)

            print("机器人已重置到初始位置")
            return True

        except Exception as e:
            print(f"重置机器人失败: {e}")
            return False


def main() -> None:
    pass


if __name__ == "__main__":
    main()
