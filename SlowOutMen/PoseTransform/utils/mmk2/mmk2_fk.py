import os
import mujoco
from scipy.spatial.transform import Rotation
import numpy as np
from typing import List, Tuple

from discoverse import DISCOVERSE_ROOT_DIR


class MMK2FK:
    def __init__(self, mjcf_path: str = None) -> None:
        if mjcf_path is None:
            mjcf_path = os.path.join(
                DISCOVERSE_ROOT_DIR, "models/mjcf/mmk2_floor.xml")
        self.mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.mj_data = mujoco.MjData(self.mj_model)

    def _forward_kinematics(self) -> None:
        """
        Perform forward kinematics to update the model state.
        """
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.pos_modifidied = False

    def _get_site_transform(self, mj_data: mujoco.MjData, site_name: str) -> np.ndarray:
        """
        Get the transformation matrix of a site in the model.
        Params:
            mj_data: MuJoCo data object
            site_name: Name of the site in the model
        Returns:
            tmat: 4x4 transformation matrix of the site
        """
        tmat = np.eye(4)
        tmat[:3, :3] = mj_data.site(site_name).xmat.reshape((3, 3))
        tmat[:3, 3] = mj_data.site(site_name).xpos
        return tmat

    def _get_body_transform(self, mj_data: mujoco.MjData, body_name: str) -> np.ndarray:
        """
        Get the transformation matrix of a body in the model.
        Params:
            mj_data: MuJoCo data object
            body_name: Name of the body in the model
        Returns:
            tmat: 4x4 transformation matrix of the body
        """
        tmat = np.eye(4)
        tmat[:3, :3] = Rotation.from_quat(mj_data.body(
            body_name).xquat[[1, 2, 3, 0]]).as_matrix()
        tmat[:3, 3] = mj_data.body(body_name).xpos
        return tmat

    def _get_camera_transform(self, mj_data: mujoco.MjData, camera_name: str) -> np.ndarray:
        """
        Get the transformation matrix of a camera in the model.
        Params:
            mj_data: MuJoCo data object
            camera_name: Name of the camera in the model
        Returns:
            tmat: 4x4 transformation matrix of the camera
        """
        tmat = np.eye(4)
        tmat[:3, :3] = mj_data.camera(camera_name).xmat.reshape((3, 3))
        tmat[:3, 3] = mj_data.camera(camera_name).xpos
        return tmat

    def _get_camera_k(self, fovy: float, width: int, height: int) -> np.ndarray:
        cx = width / 2
        cy = height / 2
        fovx = 2 * np.arctan(np.tan(fovy / 2.) * width / height)
        fx = cx / np.tan(fovx / 2)
        fy = cy / np.tan(fovy / 2)
        return np.array([[fx, 0, cx],
                        [0, fy, cy],
                        [0,  0,  1]])

    def set_base_pose(self, position: List, orientation: List) -> None:
        """
        Set the base pose of the robot.
        Params:
            position: 3D position of the base <xyz>
            orientation: 4D quaternion orientation of the base <wxyz>
        """
        self.mj_data.qpos[:3] = position
        self.mj_data.qpos[3:7] = orientation
        self.pos_modifidied = True

    def set_slide_joint(self, joint_angle: float) -> None:
        """
        Set the joint angle of the slide joint.
        Params:
            joint_angle: joint angle
        """
        self.mj_data.qpos[9] = joint_angle
        self.pos_modifidied = True

    def set_head_joints(self, joint_angles: List) -> None:
        """
        Set the joint angles of the head.
        Params:
            joint_angles: list of joint angles
        """
        assert len(joint_angles) == 2, "Head joints should have 2 joint angles."
        self.mj_data.qpos[10:12] = joint_angles
        self.pos_modifidied = True

    def set_left_arm_joints(self, joint_angles: List) -> None:
        """
        Set the joint angles of the left arm.
        Params:
            joint_angles: list of joint angles
        """
        assert len(
            joint_angles) == 6, "Left arm joints should have 6 joint angles"
        self.mj_data.qpos[12:18] = joint_angles
        self.pos_modifidied = True

    def set_right_arm_joints(self, joint_angles: List) -> None:
        """
        Set the joint angles of the right arm.
        Params:
            joint_angles: list of joint angles
        """
        assert len(
            joint_angles) == 6, "Right arm joints should have 6 joint angles"
        self.mj_data.qpos[20:26] = joint_angles
        self.pos_modifidied = True

    def get_head_camera_world_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the pose of the head camera w.r.t world.
        Returns:
            position: 3D position of the head camera <xyz>
            orientation: 4D quaternion orientation of the head camera <wxyz>
        """
        if self.pos_modifidied:
            self._forward_kinematics()
        head_camera_transform_matrix = self._get_site_transform(self.mj_data, "headeye")
        position = head_camera_transform_matrix[:3, 3]
        orientation = Rotation.from_matrix(
            head_camera_transform_matrix[:3, :3]).as_quat()[[3, 0, 1, 2]]
        return position, orientation

    def get_left_arm_camera_world_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the pose of the left camera w.r.t world.
        Returns:
            position: 3D position of the left camera <xyz>
            orientation: 4D quaternion orientation of the left camera <wxyz>"
        """
        if self.pos_modifidied:
            self._forward_kinematics()
        left_arm_camera_transform_matrix = self._get_site_transform(self.mj_data, "left_cam")
        position = left_arm_camera_transform_matrix[:3, 3]
        orientation = Rotation.from_matrix(
            left_arm_camera_transform_matrix[:3, :3]).as_quat()[[3, 0, 1, 2]]
        return position, orientation

    def get_right_arm_camera_world_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the pose of the right camera w.r.t world.
        Returns:
            position: 3D position of the right camera <xyz>
            orientation: 4D quaternion orientation of the right camera <wxyz>
        """
        if self.pos_modifidied:
            self._forward_kinematics()
        right_arm__camera_transform_matrix = self._get_site_transform(self.mj_data, "right_cam")
        position = right_arm__camera_transform_matrix[:3, 3]
        orientation = Rotation.from_matrix(
            right_arm__camera_transform_matrix[:3, :3]).as_quat()[[3, 0, 1, 2]]
        return position, orientation

    def get_left_endeffector_world_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the pose of the left end effector w.r.t world.
        Returns:
            position: 3D position of the left end effector <xyz>
            orientation: 4D quaternion orientation of the left end effector <wxyz>
        """
        if self.pos_modifidied:
            self._forward_kinematics()
        left_endeffector_transform_matrix = self._get_site_transform(self.mj_data, "lft_endpoint")
        position = left_endeffector_transform_matrix[:3, 3]
        orientation = Rotation.from_matrix(
            left_endeffector_transform_matrix[:3, :3]).as_quat()[[3, 0, 1, 2]]
        return position, orientation

    def get_right_endeffector_world_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the pose of the right end effector w.r.t world.
        Returns:
            position: 3D position of the right end effector <xyz>
            orientation: 4D quaternion orientation of the right end effector <wxyz>
        """
        if self.pos_modifidied:
            self._forward_kinematics()
        right_endeffector_transform_matrix = self._get_site_transform(self.mj_data, "rgt_endpoint")
        position = right_endeffector_transform_matrix[:3, 3]
        orientation = Rotation.from_matrix(
            right_endeffector_transform_matrix[:3, :3]).as_quat()[[3, 0, 1, 2]]
        return position, orientation


if __name__ == "__main__":
    fk = MMK2FK()

    fk.set_base_pose([0, 0, 0], [1, 0, 0, 0])
    fk.set_slide_joint(0)
    fk.set_head_joints([0, 0])
    fk.set_left_arm_joints([0, 0, 0, 0, 0, 0])
    fk.set_right_arm_joints([0, 0, 0, 0, 0, 0])

    print("Head camera pose: ", fk.get_head_camera_world_pose())
    print("Left camera pose: ", fk.get_left_arm_camera_world_pose())
    print("Right camera pose: ", fk.get_right_arm_camera_world_pose())

    print("Left end effector pose: ", fk.get_left_endeffector_world_pose())
    print("Right end effector pose: ", fk.get_right_endeffector_world_pose())
