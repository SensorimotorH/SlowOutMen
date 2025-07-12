from SlowOutMen.VisionUnderstanding.VisiontoPoindcloud.detect_and_segment import yolo_process
from SlowOutMen.PoseTransform.utils.mmk2.mmk2_fk import MMK2FK

from typing import List
import numpy as np
from scipy.spatial.transform import Rotation


class GetBasePositionFromCamera():
    def __init__(self, sim_node):
        self.mmk2_fk = MMK2FK()
        self.obs = sim_node.obs
    
    def _update_pose_fk(self):
        """
        更新FK中当前的关节角度和末端执行器的位置
        """
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
    
    def get_base_position_from_camera(self, object_name):
        object_position_wrt_camera = self.get_Toc_from_yolo(object_name)
        object_position_wrt_base = self.transform_position_wrt_camera_to_base(object_position_wrt_camera)
        return object_position_wrt_base
        
    def get_Toc_from_yolo(self, object_name: str) -> List[float]:
        fx = 575.29
        fy = 575.29
        cx = 320
        cy = 240
        rgb, depth = self._get_rgb_and_depth()
        yolo_results = yolo_process(rgb)
        target_box = [item["box"]
                      for item in yolo_results if item["class_name"] == object_name]
        x0, y0, x1, y1 = map(int, target_box[0])

        x = int((x0 + x1) / 2)
        y = int((y0 + y1) / 2)
        Z = round(depth[y, x] / 1000, 3)
        X = round((x - cx) * Z / fx, 3)
        Y = round((y - cy) * Z / fy, 3)
        vector = [X, Y, Z]
        print(
            f"binding box of {object_name} is: {np.round(np.array(target_box[0]), 3)}")
        print(f"{object_name} position in camera is: {vector}")
        return vector
    
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
        