from SlowOutMen.PoseTransform.utils.mmk2.mmk2_fk import MMK2FK
from SlowOutMen.PoseTransform.utils.mmk2.mmk2_fik import MMK2FIK
from SlowOutMen.Communication.ros2_mmk2_api import MMK2_Receiver

from scipy.spatial.transform import Rotation
import math
import rclpy
import numpy as np
from typing import List


class PoseTransform:
    def __init__(self, receiver: MMK2_Receiver):
        self.receiver = receiver
        self.mmk2_fk = MMK2FK()
        self.mmk2_fik = MMK2FIK()
        self.obs = {
            "time": None,
            "jq": [0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0.],
            "base_position": [0., 0., 0.],
            "base_orientation": [1., 0., 0., 0.],
        }

    def _update_pose_fk(self):
        """
        更新FK中当前的关节角度和末端执行器的位置
        """
        self.obs["base_position"] = self.receiver.get_base_position()[
            'position']
        self.obs["base_orientation"] = self.receiver.get_base_position()[
            'orientation_list']
        self.obs["jq"][:] = self.receiver.get_joint_states()['positions']
        self.sensor_slide_qpos = self.obs["jq"][:1]
        self.sensor_head_qpos = self.obs["jq"][1:3]
        self.sensor_lft_arm_qpos = self.obs["jq"][3:9]
        self.sensor_lft_gripper_qpos = self.obs["jq"][9:10]
        self.sensor_rgt_arm_qpos = self.obs["jq"][10:16]
        self.sensor_rgt_gripper_qpos = self.obs["jq"][16:17]

        self.mmk2_fk.set_base_pose(
            self.obs["base_position"], self.obs["base_orientation"])
        self.mmk2_fk.set_slide_joint(self.sensor_slide_qpos[0])
        self.mmk2_fk.set_head_joints(self.sensor_head_qpos)
        self.mmk2_fk.set_left_arm_joints(self.sensor_lft_arm_qpos)
        self.mmk2_fk.set_right_arm_joints(self.sensor_rgt_arm_qpos)

    def get_arm_joint_pose(self, target_position: np.ndarray, arm_action: str | np.ndarray, arm: str, q_ref: np.ndarray = None, action_rot: np.ndarray = np.eye(3)) -> np.ndarray | bool:
        # target_position: pose w.r.t base 3dim
        assert len(target_position) == 3, "target_position must be a 3D vector"
        if isinstance(arm_action, str):
            assert arm_action in [
                "pick", "carry", "look"], "arm_action must be 'pick', 'carry' or 'look'"
        assert arm in ["l", "r"], "arm must be 'l' or 'r'"
        assert q_ref is None or len(
            q_ref) == 6, "q_ref must be a 6D vector or None"
        head_slide_position = self.receiver.get_head_position()['slide']

        if q_ref is None:
            if arm == "l":
                q_ref = np.array(
                    self.receiver.get_joint_states()['positions'][3:9])
            elif arm == "r":
                q_ref = np.array(
                    self.receiver.get_joint_states()['positions'][10:16])
        try:
            rq = self.mmk2_fik.get_arm_joint_pose_wrt_base(
                target_position, arm_action, arm, head_slide_position, q_ref, action_rot)
            return rq

        except ValueError as e:
            print(
                f"Failed to solve IK {e} params: arm={arm}, target={target_position}, slide={head_slide_position:.2f}")
            return False

    def get_world_position_from_head_camera(self, position_in_camera: List[float] | np.ndarray) -> List[float] | np.ndarray:
        """
        获取相机坐标系下的点在世界坐标系下的坐标
        :param position_in_camera: 相机坐标系下的点的坐标 (x, y, z)
        :return: 相机坐标系下的点在世界坐标系下的坐标
        """
        position_in_base = self.transform_position_wrt_camera_to_base(
            position_in_camera)
        position_in_world = self.transform_position_wrt_base_to_world(
            position_in_base)
        return position_in_world

    def transform_position_wrt_camera_to_base(self, position_in_camera: List[float] | np.ndarray) -> List[float] | np.ndarray:
        """
        将相机坐标系下的点转换到MMK2基座坐标系下
        :param position_in_camera: 相机坐标系下的点的坐标 (x, y, z)
        :return: MMK2基座坐标系下的坐标 (x, y, z)
        """
        self._update_pose_fk()

        head_camera_world_position, head_camera_world_orientation = self.mmk2_fk.get_head_camera_world_pose()
        head_camera_transform_matrix = np.eye(4)
        head_camera_transform_matrix[:3, 3] = head_camera_world_position
        head_camera_transform_matrix[:3, :3] = Rotation.from_quat(
            head_camera_world_orientation[[1, 2, 3, 0]]).as_matrix()

        point3d = np.array(
            [position_in_camera[0], position_in_camera[1], position_in_camera[2], 1.0])
        world_position = head_camera_transform_matrix @ point3d

        # mmk2 w.r.t world
        current_world_position = self.obs["base_position"]   # [X, Y, Z]
        current_world_quat = self.obs["base_orientation"]  # [qw, qx, qy, qz]

        mmk2_world_pose = np.eye(4)
        mmk2_world_pose[:3, 3] = current_world_position
        mmk2_world_pose[:3, :3] = Rotation.from_quat(
            [current_world_quat[1], current_world_quat[2], current_world_quat[3], current_world_quat[0]]).as_matrix()

        position_in_base = (np.linalg.inv(
            mmk2_world_pose) @ world_position)[:3]

        return position_in_base

    def transform_position_wrt_base_to_world(self, position_in_base: List[float] | np.ndarray) -> List[float] | np.ndarray:
        """
        将物体从机器人基座坐标系转换到世界坐标系下的坐标。

        Parameters:
        - position_in_base: numpy.ndarray, 物体在机器人基座坐标系下的坐标 (x_base, y_base, z_base)
        Returns:
        - numpy.ndarray(float, float, float): 物体在世界坐标系下的坐标 (x_world, y_world, z_world)
        """
        x_mmk2_world_position = self.receiver.get_base_position()['x']
        y_mmk2_world_position = self.receiver.get_base_position()['y']
        z_mmk2_world_position = self.receiver.get_base_position()['z']
        rot = self.receiver.get_base_position()["orientation"]

        # 官方的正方向是面朝右柜子
        theta_official = np.rad2deg(Rotation.from_quat(
            [rot['x'], rot["y"], rot["z"], rot["w"]]).as_euler('zyx')[0])
        theta_official = (theta_official + 180) % 360 - 180

        x_base = position_in_base[0]
        y_base = position_in_base[1]
        z_base = position_in_base[2]
        theta_rad = math.radians(theta_official)

        # 应用旋转变换
        x_rot = x_base * math.cos(theta_rad) - y_base * math.sin(theta_rad)
        y_rot = x_base * math.sin(theta_rad) + y_base * math.cos(theta_rad)

        # 平移变换
        x_world = x_rot + x_mmk2_world_position
        y_world = y_rot + y_mmk2_world_position
        z_world = z_base + z_mmk2_world_position
        position_in_world = np.array([x_world, y_world, z_world])

        return position_in_world
    
    def transform_position_wrt_world_to_base(self, position_in_world:np.array) -> np.array:
        """
        将位置从世界坐标系转换到机器人基座坐标系的位置。

        Parameters:
        - position_in_world: numpy.array, 物体在机器人基座坐标系下的坐标 (x_local, y_local, z_local)
        Returns:
        - numpy.array(float, float): 物体在世界坐标系下的坐标 (x_world, y_world)
        """
        x_mmk2 = self.receiver.get_odom()['position']['x']
        y_mmk2 = self.receiver.get_odom()['position']['y']
        rot = self.receiver.get_odom()["orientation"]
        # 官方的正方向是面朝右柜子
        theta_official =np.rad2deg(Rotation.from_quat(
            [rot['x'], rot["y"], rot["z"], rot["w"]]).as_euler('zyx')[0])
        theta_official = (theta_official + 180) % 360 - 180
        
        x_world = position_in_world[0]
        y_world = position_in_world[1]
        z_world = position_in_world[2]
        
        # 首先进行平移变换（相对于机器人基座位置）
        x_trans = x_world - x_mmk2
        y_trans = y_world - y_mmk2

        # 然后进行旋转变换（使用相反的角度）
        theta_rad = math.radians(-theta_official)  # 注意这里使用负角度来进行逆变换
        
        # 应用旋转变换
        x_local = x_trans * math.cos(theta_rad) - y_trans * math.sin(theta_rad)
        y_local = x_trans * math.sin(theta_rad) + y_trans * math.cos(theta_rad)
        
        # 返回基座坐标系下的位置
        position_in_base = np.array([x_local, y_local, z_world])
        return position_in_base


def main():
    rclpy.init()
    receiver = MMK2_Receiver()
    posetran = PoseTransform(receiver)
    pose_base = posetran.transform_position_wrt_camera_to_base(
        [0.012, -0.005, 0.553])
    print(pose_base)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
