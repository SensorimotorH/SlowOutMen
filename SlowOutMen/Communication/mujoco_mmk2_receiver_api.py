import time
import numpy as np
import mujoco
import mujoco.viewer
from typing import Optional, Dict, Any
from scipy.spatial.transform import Rotation
import os
import datetime
import cv2

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


AVAILABLE_SUBSCIPTION_TOPICS = [
    'clock',
    'head_rgb',
    'head_depth',
    'head_camera_info',
    'head_depth_camera_info',
    'left_rgb',
    'left_camera_info',
    'right_rgb',
    'right_camera_info',
    'odom',
    'joint_states',
    'taskinfo',
    'gameinfo'
]
ROBOT_LINK_LIST = [
    "agv_link", "slide_link", "head_yaw_link", "head_pitch_link",
    "lft_arm_base", "lft_arm_link1", "lft_arm_link2",
    "lft_arm_link3", "lft_arm_link4", "lft_arm_link5", "lft_arm_link6",
    "lft_finger_left_link", "lft_finger_right_link",
    "rgt_arm_base", "rgt_arm_link1", "rgt_arm_link2",
    "rgt_arm_link3", "rgt_arm_link4", "rgt_arm_link5", "rgt_arm_link6",
    "rgt_finger_left_link", "rgt_finger_right_link"
]


class MMK2_Receiver:
    """
    在 MuJoCo 环境下的MMK2机器人传感器数据接收器
    提供获取摄像头图像、关节状态等功能
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, save_dir: Optional[str] = None, debug: bool = False):
        """
        初始化接收器
        :param model: MuJoCo模型
        :param data: MuJoCo数据
        :param save_dir: 数据保存目录,默认为None时会自动生成
        :param debug: 是否启用调试模式
        """
        self.model = model
        self.data = data
        self.save_dir = save_dir
        self.debug = debug
        self.topics = AVAILABLE_SUBSCIPTION_TOPICS
        self.sub_imgs = {}
        self.sub_files = {}
        self.njq = 28
        self.njctrl = 19
        self.initial_joint_states = self._record_initial_joint_states()
        self.height = 480
        self.width = 640

        self.renderer = mujoco.Renderer(
            self.model, self.height, self.width)
        self.options = mujoco.MjvOption()

        self.viewer = None

        self.free_camera = mujoco.MjvCamera()
        self.free_camera.fixedcamid = -1
        self.free_camera.type = mujoco._enums.mjtCamera.mjCAMERA_FREE
        mujoco.mjv_defaultFreeCamera(self.model, self.free_camera)

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        # 设置保存目录
        if self.debug:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.head_rgb_list = []
            self.head_depth_list = []
            self.right_rgb_list = []
            self.left_rgb_list = []
            if save_dir is None:
                self.save_dir = f"/workspace/SlowOutMen/DebugOutput/recorded_data_{timestamp}"
                os.makedirs(self.save_dir, exist_ok=True)

                # 初始化文件和目录
                for topic_name in self.topics:
                    if topic_name.endswith('_rgb') or topic_name.endswith('_depth'):
                        self.sub_imgs[topic_name] = os.path.join(
                            self.save_dir, topic_name)
                        os.makedirs(self.sub_imgs[topic_name], exist_ok=True)
                    else:
                        ext = '.json' if topic_name.endswith(
                            '_camera_info') else '.txt'
                        self.sub_files[topic_name] = open(os.path.join(
                            self.save_dir, f'{topic_name}{ext}'), 'w')

        # 末端执行器初始位置
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

        # 初始化基座位置和方向
        self.base_position = np.zeros(3)
        self.base_orientation = np.array(
            [1.0, 0.0, 0.0, 0.0])  # 四元数 [w, x, y, z]

    def _record_initial_joint_states(self)-> dict[str, Any] | dict[str, Any]:
        """
        记录初始关节状态
        """
        try:
            # 获取传感器数据
            # 根据MMK2Base类的定义,传感器数据组织如下
            qpos = self.data.sensordata[:self.njctrl]

            # 获取基座位置和方向
            base_position = self.data.sensordata[3 *
                                                 self.njctrl:3 * self.njctrl + 3]
            base_orientation = self.data.sensordata[3 *
                                                    self.njctrl + 3:3 * self.njctrl + 7]

            # 轮子位置
            wheel_qpos = qpos[:2]

            # 升降台高度
            slide_qpos = qpos[2:3]

            # 头部位置
            head_qpos = qpos[3:5]

            # 左臂位置 (6个值)
            left_arm_qpos = qpos[5:11]

            # 左夹爪
            left_gripper_qpos = qpos[11:12]

            # 右臂位置 (6个值)
            right_arm_qpos = qpos[12:18]

            # 右夹爪
            right_gripper_qpos = qpos[18:19]

            # 为防止无法获取传感器数据,设置默认值
            if len(qpos) == 0:
                wheel_qpos = np.zeros(2)
                slide_qpos = np.array([DEFAULT_SPINE_HEIGHT])
                head_qpos = np.array(DEFAULT_HEAD_POSITION)
                left_arm_qpos = np.array(DEFAULT_LEFT_ARM_POSITION[:6])
                left_gripper_qpos = np.array([GRIPPER_OPEN])
                right_arm_qpos = np.array(DEFAULT_RIGHT_ARM_POSITION[:6])
                right_gripper_qpos = np.array([GRIPPER_OPEN])
                base_position = np.zeros(3)
                base_orientation = np.array(
                    [1.0, 0.0, 0.0, 0.0])

            return {
                "positions": qpos.tolist(),
                "wheel_positions": wheel_qpos.tolist(),
                "slide_height": slide_qpos.tolist(),
                "head_positions": head_qpos.tolist(),
                "left_arm_positions": left_arm_qpos.tolist(),
                "left_gripper_position": left_gripper_qpos.tolist(),
                "right_arm_positions": right_arm_qpos.tolist(),
                "right_gripper_position": right_gripper_qpos.tolist(),
                "base_position": base_position.tolist(),
                "base_orientation": base_orientation.tolist()
            }
        except Exception as e:
            print(f"记录初始关节状态失败: {e}")
            return {
                "positions": np.zeros(self.njctrl).tolist(),
                "wheel_positions": np.zeros(2).tolist(),
                "slide_height": np.array([DEFAULT_SPINE_HEIGHT]).tolist(),
                "head_positions": np.array(DEFAULT_HEAD_POSITION).tolist(),
                "left_arm_positions": np.array(DEFAULT_LEFT_ARM_POSITION[:6]).tolist(),
                "left_gripper_position": np.array([GRIPPER_OPEN]).tolist(),
                "right_arm_positions": np.array(DEFAULT_RIGHT_ARM_POSITION[:6]).tolist(),
                "right_gripper_position": np.array([GRIPPER_OPEN]).tolist(),
                "base_position": np.zeros(3).tolist(),
                "base_orientation": np.array([1.0, 0.0, 0.0, 0.0]).tolist()
            }

    def _get_camera_intrinsics(self, camera_name: str) -> Optional[np.ndarray]:
        """
        获取相机内参矩阵K
        :param camera_name: 相机名称
        :return: 3x3相机内参矩阵,若无法获取则返回None
        """
        try:
            camera_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

            fovy = self.model.cam_fovy[camera_id]
            width = self.width
            height = self.height

            cx = width / 2
            cy = height / 2
            fovx = 2 * np.arctan(np.tan(fovy / 2.) * width / height)
            fx = cx / np.tan(fovx / 2)
            fy = cy / np.tan(fovy / 2)

            K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0,  0,  1]])

            return K
        except Exception as e:
            print(f"获取相机内参失败: {e}")
            return None

    def get_initial_joint_states(self) -> Optional[Dict]:
        """
        获取初始关节位置状态
        :return: 包含初始关节位置的字典,若无数据则返回None
        """
        return self.initial_joint_states

    def get_base_position_orientation(self) -> Dict:
        """
        获取机器人基座位置和方向
        :return: 包含位置和方向的字典
        """
        try:
            self.get_joint_states()
            return {
                "position": self.base_position,
                "orientation": self.base_orientation
            }
        except Exception as e:
            print(f"获取基座位置和方向失败: {e}")
            return {
                "position": self.base_position,
                "orientation": self.base_orientation
            }

    def get_head_rgb_camera_info(self) -> Optional[Dict]:
        """
        获取头部RGB相机信息
        :return: 相机参数信息对象,若无数据则返回None
        """
        try:
            return {
                "width": self.width,
                "height": self.height,
                "k": self._get_camera_intrinsics("head_cam")
            }
        except Exception as e:
            print(f"获取头部RGB相机信息失败: {e}")
            return None

    def get_head_depth_camera_info(self) -> Optional[Dict]:
        """
        获取头部深度相机信息
        :return: 相机参数信息对象,若无数据则返回None
        """
        try:
            return {
                "width": self.width,
                "height": self.height,
                "k": self._get_camera_intrinsics("head_cam")
            }
        except Exception as e:
            print(f"获取头部深度相机信息失败: {e}")
            return None

    def get_left_arm_camera_info(self) -> Optional[Dict]:
        """
        获取左臂相机信息
        :return: 相机参数信息对象,若无数据则返回None
        """
        try:
            return {
                "width": self.width,
                "height": self.height,
                "k": self._get_camera_intrinsics("lft_handeye")
            }
        except Exception as e:
            print(f"获取左臂相机信息失败: {e}")
            return None

    def get_right_arm_camera_info(self) -> Optional[Dict]:
        """
        获取右臂相机信息
        :return: 相机参数信息对象,若无数据则返回None
        """
        try:
            return {
                "width": self.width,
                "height": self.height,
                "k": self._get_camera_intrinsics("rgt_handeye")
            }
        except Exception as e:
            print(f"获取右臂相机信息失败: {e}")
            return None

    def get_head_depth(self) -> Optional[np.ndarray]:
        """
        获取头部深度图像
        :return: 深度图像数组,若无数据则返回None
        """
        try:
            camera_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
            self.renderer.enable_depth_rendering()
            self.renderer.update_scene(self.data, camera_id, self.options)
            head_depth_img = self.renderer.render()
            head_depth_img = np.array(
                np.clip(head_depth_img*1e3, 0, 65535), dtype=np.uint16)

            if self.debug:
                timestamp = time.time()
                filename = f"{timestamp:.6f}.png"
                cv2.imwrite(os.path.join(
                    self.sub_imgs['head_depth'], filename), head_depth_img)
                print(f"保存头部深度图像: {filename}")

            return head_depth_img

        except Exception as e:
            print(f"获取头部深度图像失败: {e}")
            return None

    def get_head_depth_rgb(self, min_depth=0.0, max_depth=2.0) -> Optional[np.ndarray]:
        """
        获取头部深度图像
        :return: 深度图像数组,若无数据则返回None
        """
        try:
            camera_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
            self.renderer.enable_depth_rendering()
            self.renderer.update_scene(self.data, camera_id, self.options)
            head_depth_rgb_img = self.renderer.render()

            head_depth_rgb_img = np.clip(
                head_depth_rgb_img, min_depth, max_depth)
            head_depth_rgb_img = (
                head_depth_rgb_img * 255.0 / (max_depth - min_depth)).astype(np.uint8)
            head_depth_rgb_img = cv2.applyColorMap(
                head_depth_rgb_img, cv2.COLORMAP_JET)

            return head_depth_rgb_img

        except Exception as e:
            print(f"获取头部深度图像失败: {e}")
            return None

    def get_head_rgb(self) -> Optional[np.ndarray]:
        """
        获取头部RGB图像
        :return: RGB图像数组,若无数据则返回None
        """
        try:
            camera_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
            self.renderer.disable_depth_rendering()
            self.renderer.update_scene(self.data, camera_id, self.options)
            head_rgb_img = self.renderer.render()
            head_rgb_img = cv2.cvtColor(head_rgb_img, cv2.COLOR_RGB2BGR)

            if self.debug:
                timestamp = time.time()
                filename = f"{timestamp:.6f}.png"
                cv2.imwrite(os.path.join(
                    self.sub_imgs['head_rgb'], filename), head_rgb_img)
                print(f"保存头部RGB图像: {filename}")

            return head_rgb_img

        except Exception as e:
            print(f"获取头部RGB图像失败: {e}")
            return None

    def get_free_camera_rgb(self) -> Optional[np.ndarray]:
        """
        获取自由视角RGB图像
        :return: RGB图像数组,若无数据则返回None
        """
        try:
            self.renderer.update_scene(
                self.data, self.free_camera, self.options)
            self.renderer.disable_depth_rendering()
            self.renderer.update_scene(
                self.data, self.free_camera, self.options)
            free_camera_rgb_img = self.renderer.render()
            free_camera_rgb_img = cv2.cvtColor(
                free_camera_rgb_img, cv2.COLOR_RGB2BGR)

            return free_camera_rgb_img

        except Exception as e:
            print(f"获取自由视角RGB图像失败: {e}")
            return None

    def get_left_arm_rgb(self) -> Optional[np.ndarray]:
        """
        获取左臂摄像头RGB图像
        :return: RGB图像数组,若无数据则返回None
        """
        try:
            camera_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, "lft_handeye")
            self.renderer.disable_depth_rendering()
            self.renderer.update_scene(self.data, camera_id, self.options)
            left_arm_rgb_img = self.renderer.render()
            left_arm_rgb_img = cv2.cvtColor(
                left_arm_rgb_img, cv2.COLOR_RGB2BGR)

            if self.debug:
                timestamp = time.time()
                filename = f"{timestamp:.6f}.png"
                cv2.imwrite(os.path.join(
                    self.sub_imgs['left_rgb'], filename), left_arm_rgb_img)
                print(f"保存左臂RGB图像: {filename}")

            return left_arm_rgb_img

        except Exception as e:
            print(f"获取左臂RGB图像失败: {e}")
            return None

    def get_right_arm_rgb(self) -> Optional[np.ndarray]:
        """
        获取右臂摄像头RGB图像
        :return: RGB图像数组,若无数据则返回None
        """
        try:
            camera_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, "rgt_handeye")
            self.renderer.disable_depth_rendering()
            self.renderer.update_scene(self.data, camera_id, self.options)
            right_arm_rgb_img = self.renderer.render()
            right_arm_rgb_img = cv2.cvtColor(
                right_arm_rgb_img, cv2.COLOR_RGB2BGR)

            if self.debug:
                timestamp = time.time()
                filename = f"{timestamp:.6f}.png"
                cv2.imwrite(os.path.join(
                    self.sub_imgs['right_rgb'], filename), right_arm_rgb_img)
                print(f"保存右臂RGB图像: {filename}")

            return right_arm_rgb_img

        except Exception as e:
            print(f"获取右臂RGB图像失败: {e}")
            return None

    def get_joint_states(self) -> Optional[Dict]:
        """
        获取关节状态
        :return: 包含关节位置、速度、力矩的字典,若无数据则返回None
        """
        try:
            self.qpos = self.data.sensordata[:self.njctrl]
            self.qvel = self.data.sensordata[self.njctrl:2*self.njctrl]
            self.force = self.data.sensordata[2*self.njctrl:3*self.njctrl]

            # base
            self.base_position = self.data.sensordata[3 *
                                                      self.njctrl:3*self.njctrl+3]
            self.base_orientation = self.data.sensordata[3 *
                                                         self.njctrl+3:3*self.njctrl+7]
            self.base_linear_vel = self.data.sensordata[3 *
                                                        self.njctrl+7:3*self.njctrl+10]
            self.base_gyro = self.data.sensordata[3 *
                                                  self.njctrl+10:3*self.njctrl+13]
            self.base_acc = self.data.sensordata[3 *
                                                 self.njctrl+13:3*self.njctrl+16]

            # arm endpoint
            self.left_arm_endpoint = self.data.sensordata[3 *
                                                          self.njctrl+16:3*self.njctrl+19]
            self.left_arm_gripper = self.data.sensordata[3 *
                                                         self.njctrl+19:3*self.njctrl+23]
            self.right_arm_endpoint = self.data.sensordata[3 *
                                                           self.njctrl+23:3*self.njctrl+26]
            self.right_arm_gripper = self.data.sensordata[3 *
                                                          self.njctrl+26:3*self.njctrl+30]

            # wheel
            self.wheel_qpos = self.qpos[:2]
            self.wheel_qvel = self.qvel[:2]
            self.wheel_qctrl = self.force[:2]

            # slide
            self.slide_qpos = self.qpos[2:3]
            self.slide_qvel = self.qvel[2:3]
            self.slide_qctrl = self.force[2:3]

            # head
            self.head_qpos = self.qpos[3:5]
            self.head_qvel = self.qvel[3:5]
            self.head_qctrl = self.force[3:5]

            # left arm
            self.left_arm_qpos = self.qpos[5:11]
            self.left_arm_qvel = self.qvel[5:11]
            self.left_arm_qctrl = self.force[5:11]

            # left gripper
            self.left_gripper_qpos = self.qpos[11:12]
            self.left_gripper_qvel = self.qvel[11:12]
            self.left_gripper_ctrl = self.force[11:12]

            # right arm
            self.right_arm_qpos = self.qpos[12:18]
            self.right_arm_qvel = self.qvel[12:18]
            self.right_arm_qctrl = self.force[12:18]

            # right gripper
            self.right_gripper_qpos = self.qpos[18:19]
            self.right_gripper_qvel = self.qvel[18:19]
            self.right_gripper_ctrl = self.force[18:19]

            # 构建关节状态字典
            joint_states = {
                "positions": self.qpos.tolist(),
                "velocities": self.qvel.tolist(),
                "effort": self.force.tolist(),
            }

            if self.debug:
                timestamp = time.time()
                positions = ','.join(
                    [f"{p}" for p in joint_states['positions']])
                velocities = ','.join(
                    [f"{v}" for v in joint_states['velocities']])
                efforts = ','.join([f"{e}" for e in joint_states['effort']])
                self.sub_files['joint_states'].write(
                    f"{timestamp},{positions},{velocities},{efforts}\n")
                self.sub_files['joint_states'].flush()

            return joint_states
        except Exception as e:
            print(f"获取关节状态失败: {e}")
            return None

    def get_end_effector_pose(self, arm: str) -> Optional[Dict]:
        """
        获取末端执行器位姿
        :param arm: 'l'表示左臂, 'r'表示右臂
        :return: 包含位置和姿态的字典,若无数据则返回None
        """
        try:
            if arm == 'l':
                left_finger_site_name = "lft_finger_left_link"
                right_finger_site_name = "lft_finger_right_link"
            elif arm == 'r':
                left_finger_site_name = "rgt_finger_left_link"
                right_finger_site_name = "rgt_finger_right_link"
            else:
                raise ValueError("参数arm必须为'l'或'r'")

            left_finger_site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, left_finger_site_name)
            right_finger_site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, right_finger_site_name)

            # 计算末端位置
            left_finger_position = self.data.site_xpos[left_finger_site_id]
            right_finger_position = self.data.site_xpos[right_finger_site_id]
            position = (left_finger_position + right_finger_position) / 2

            # 计算末端旋转矩阵
            left_finger_mat = self.data.site_xmat[left_finger_site_id].reshape(
                3, 3)
            right_finger_mat = self.data.site_xmat[right_finger_site_id].reshape(
                3, 3)
            rotation_matrix = (left_finger_mat + right_finger_mat) / 2

            # 转换为欧拉角
            rotation = Rotation.from_matrix(rotation_matrix)
            euler = rotation.as_euler('zyx')

            return {
                "position": position,
                "rotation_matrix": rotation_matrix,
                "euler": euler
            }
        except Exception as e:
            print(f"获取末端执行器位姿失败: {e}")
            return None

    def get_object_pose(self, name: str) -> tuple | tuple[Any, np.ndarray[Any, Any]] | tuple[None, None]:
        try:
            position = self.data.body(name).xpos
            quat = self.data.body(name).xquat
            return position, quat
        except KeyError:
            try:
                position = self.data.geom(name).xpos
                quat = Rotation.from_matrix(self.data.geom(
                    name).xmat.reshape((3, 3))).as_quat()[[3, 0, 1, 2]]
                return position, quat
            except KeyError:
                print("Invalid object name: {}".format(name))
                return None, None

    def render(self, use_mujoco_viewer: bool = True, use_cv_viewer: bool = True, enable_depth: bool = False) -> None:
        if use_mujoco_viewer:
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(
                    self.model, self.data)
                self.viewer.sync()
            else:
                self.viewer.sync()

        if use_cv_viewer:
            cv2.namedWindow('Head View', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Head View', self.width, self.height)

            img = self.get_head_rgb()
            if img is not None and img.size > 0:
                cv2.imshow('Head View', img)

            if enable_depth:
                cv2.namedWindow('Depth View', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Depth View', self.width, self.height)

                depth = self.get_head_depth_rgb()
                if depth is not None and depth.size > 0:
                    cv2.imshow('Depth View', depth)

            cv2.waitKey(1)

    def __del__(self) -> None:
        if self.debug:
            for f in self.sub_files.values():
                f.close()


def main() -> None:
    pass


if __name__ == "__main__":
    main()
