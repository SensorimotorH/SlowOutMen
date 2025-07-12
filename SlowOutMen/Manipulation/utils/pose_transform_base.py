from DISCOVERSE.discoverse.mmk2.mmk2_fk import MMK2FK
from SlowOutMen.Communication.ros2_mmk2_api import MMK2_Receiver
from scipy.spatial.transform import Rotation

import math
import rclpy
import numpy as np


class PoseTransform:
    def __init__(self, receiver: MMK2_Receiver):
        self.receiver = receiver
        self.mmk2_fk = MMK2FK()
        self.obs = {
            "time": None,
            "jq": [0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0.],
            "base_position": [0., 0., 0.],
            "base_orientation": [1., 0., 0., 0.],
        }

    def update_pose_fk(self):
        """
        更新FK中当前的关节角度和末端执行器的位置
        """
        position_x = self.receiver.get_odom()['position']['x']
        position_y = self.receiver.get_odom()['position']['y']
        position_z = self.receiver.get_odom()['position']['z']
        self.obs["base_position"] = [position_x, position_y, position_z]
        orientation_x = self.receiver.get_odom()['orientation']['x']
        orientation_y = self.receiver.get_odom()['orientation']['y']
        orientation_z = self.receiver.get_odom()['orientation']['z']
        orientation_w = self.receiver.get_odom()['orientation']['w']
        self.obs["base_orientation"] = [orientation_w, orientation_x,
                                        orientation_y, orientation_z]
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

    def pos_cam_to_base(self, pos_cam):
        """
        将相机坐标系下的点转换到MMK2基座坐标系下
        :param pos_cam: 相机坐标系下的点
        :return: MMK2基座坐标系下的点
        """
        self.update_pose_fk()
        
        cam_head_trans, cam_head_ori = self.mmk2_fk.get_head_camera_pose()
        tmat_cam_head = np.eye(4)
        tmat_cam_head[:3, 3] = cam_head_trans
        tmat_cam_head[:3, :3] = Rotation.from_quat(
            cam_head_ori[[1, 2, 3, 0]]).as_matrix()

        point3d = np.array([pos_cam[0], pos_cam[1], pos_cam[2], 1.0])
        posi_world = tmat_cam_head @ point3d

        # mmk2 w.r.t world
        current_pos = self.obs["base_position"]   # [X, Y, Z]
        current_quat = self.obs["base_orientation"]  # [qw, qx, qy, qz]

        tmat_mmk2 = np.eye(4)
        tmat_mmk2[:3, 3] = current_pos
        tmat_mmk2[:3, :3] = Rotation.from_quat(
            [current_quat[1], current_quat[2], current_quat[3], current_quat[0]]).as_matrix()

        posi_local = (np.linalg.inv(tmat_mmk2) @ posi_world)[:3]

        return posi_local
    
    def get_end_pose_base(self, arm: str):
        """
        得到末端执行器在基座坐标系下位置
        
        Parameters:
        - arm: str, 末端执行器的类型，'left' 或 'right'
        Returns:
        - numpy.array(float, float, float): 末端执行器在基座坐标系下的位置 (x_base, y_base, z_base)
        """
        self.update_pose_fk()
        if arm == 'left':
            end_pos, _ = self.mmk2_fk.get_left_endeffector_pose()
        elif arm == 'right':
            end_pos, _ = self.mmk2_fk.get_right_endeffector_pose()
        else:
            raise ValueError("arm参数必须是'left'或'right'")
        
        # 将末端执行器的世界坐标转换为基座坐标系下的坐标
        # 创建末端执行器在世界坐标系中的齐次坐标
        pos_world = np.append(end_pos, 1.0)
        
        # 创建机器人基座在世界坐标系中的变换矩阵
        tmat_mmk2 = np.eye(4)
        tmat_mmk2[:3, 3] = self.obs["base_position"]
        tmat_mmk2[:3, :3] = Rotation.from_quat([
            self.obs["base_orientation"][1],  # qx
            self.obs["base_orientation"][2],  # qy
            self.obs["base_orientation"][3],  # qz
            self.obs["base_orientation"][0]   # qw
        ]).as_matrix()
        
        # 转换到基座坐标系
        pos_base = (np.linalg.inv(tmat_mmk2) @ pos_world)[:3]
        
        return pos_base

    def pos_base_to_world(self, object_local:np.array) -> np.array:
        """
        将物体从机器人基座坐标系转换到世界坐标系下的坐标。

        Parameters:
        - object_local: numpy.array, 物体在机器人基座坐标系下的坐标 (x_local, y_local, z_local)
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

        x_local = object_local[0]
        y_local = object_local[1]
        z_local = object_local[2]
        

        theta_rad = math.radians(theta_official)

        # 应用旋转变换
        x_rot = x_local * math.cos(theta_rad) - y_local * math.sin(theta_rad)
        y_rot = x_local * math.sin(theta_rad) + y_local * math.cos(theta_rad)

        # 平移变换
        x_world = x_rot + x_mmk2
        y_world = y_rot + y_mmk2
        pos_world = np.array([x_world, y_world, z_local])
        return pos_world
    
    def pos_world_to_base(self, position_world:np.array) -> np.array:
        """
        将位置从世界坐标系转换到机器人基座坐标系的位置。

        Parameters:
        - position_world: numpy.array, 物体在机器人基座坐标系下的坐标 (x_local, y_local, z_local)
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
        
        x_world = position_world[0]
        y_world = position_world[1]
        z_world = position_world[2]
        
        # 首先进行平移变换（相对于机器人基座位置）
        x_trans = x_world - x_mmk2
        y_trans = y_world - y_mmk2

        # 然后进行旋转变换（使用相反的角度）
        theta_rad = math.radians(-theta_official)  # 注意这里使用负角度来进行逆变换
        
        # 应用旋转变换
        x_local = x_trans * math.cos(theta_rad) - y_trans * math.sin(theta_rad)
        y_local = x_trans * math.sin(theta_rad) + y_trans * math.cos(theta_rad)
        
        # 返回基座坐标系下的位置
        pos_local = np.array([x_local, y_local, z_world])
        return pos_local
        
        
        
def main():
    rclpy.init()
    receiver = MMK2_Receiver()
    posetran = PoseTransform(receiver)
    pose_base = posetran.pos_cam_to_base([0.012, -0.005, 0.553])
    print(pose_base)
    rclpy.shutdown()
        
if __name__ == "__main__":
    main()
    