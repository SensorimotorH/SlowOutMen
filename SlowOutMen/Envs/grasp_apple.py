from scipy.spatial.transform import Rotation
from discoverse.mmk2.mmk2_fik import MMK2FIK
import mujoco.viewer
import mujoco
import cv2
from SlowOutMen.PoseTransform.pose_transform import get_relative_pose
from scipy.spatial.transform import Rotation as R
import numpy as np
import os
from typing import Optional, List

SIM2REAL_ASSERT_DIR = "DISCOVERSE/models/mjcf"
ACTION_ROT = {
    "pick": {
        "l": np.array([
            [0.,  0.7071,  0.7071],
            [1.,  0.,  0.],
            [0.,  0.7071, -0.7071],
        ]),
        "r": np.array([
            [0., -0.7071,  0.7071],
            [-1.,  0.,  0.],
            [0., -0.7071, -0.7071],
        ]),
    },
    "carry": {
        "l": np.array([
            [0., -0.7071,  0.7071],
            [1.,  0.,  0.],
            [0.,  0.7071,  0.7071],
        ]),
        "r": np.array([
            [0.,  0.7071,  0.7071],
            [-1.,  0.,  0.],
            [0., -0.7071,  0.7071],
        ]),
    },
    "look": {
        "l": np.array([
            [0.707, -0.707,  0.],
            [0.,  0., -1.],
            [0.707,  0.707,  0.],
        ]),
        "r": np.array([
            [0.707,  0.707, -0.],
            [0.,  0.,  1.],
            [0.707, -0.707,  0.],
        ])

    }
}


class GraspAppleTask:
    def __init__(self, obj_name: str, base_name: str, cam_name: str, arm: str = "l", action: str = "pick") -> None:
        assert SIM2REAL_ASSERT_DIR is os.path.abspath(
            SIM2REAL_ASSERT_DIR), "SIM2REAL_ASSERT_DIR不是绝对路径"
        self.obj_name = obj_name
        self.cam_name = cam_name
        self.base_name = base_name
        self.action = action
        self.arm = arm
        self.action_rot = ACTION_ROT[self.action][self.arm]
        assert self.base_name[0] == self.arm[0], "机械臂基座名称与机械臂名称不匹配"

        self.xml_path = os.path.join(SIM2REAL_ASSERT_DIR, "grasp_apple.xml")
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.table_pos = np.array([0.0, -0.75, 0.0])  # 桌子中心位置
        self.table_size = np.array([1.6, 0.8, 0.75])  # 桌子尺寸
        self.obj_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "apple_free_joint")
        self.obj_qpos = self.model.jnt_qposadr[self.obj_id]
        self.MMK2FIK = MMK2FIK()

    def reset(self) -> None:
        """重置环境并随机生成苹果位置"""
        mujoco.mj_resetData(self.model, self.data)

        # 在桌面范围内随机选择位置（保留20%边距）
        x_range = self.table_size[0] * 0.1  # 使用桌子长度的80%
        y_range = self.table_size[1] * 0.3  # 使用桌子宽度的80%

        apple_x = self.table_pos[0] + np.random.uniform(0, x_range)
        apple_y = self.table_pos[1] + np.random.uniform(y_range * 0.5, y_range)
        apple_z = self.table_pos[2] + self.table_size[2] + 0.3  # 桌面高度加上偏移
        self.data.qpos[self.obj_qpos:self.obj_qpos +
                       3] = np.array([apple_x, apple_y, apple_z])

        random_rot = R.random()
        random_quat = random_rot.as_quat()
        random_quat = np.array(
            [random_quat[3], random_quat[0], random_quat[1], random_quat[2]])
        self.data.qpos[self.obj_qpos+3:self.obj_qpos+7] = random_quat

        mujoco.mj_forward(self.model, self.data)

    def render(self) -> None:
        """渲染当前场景"""
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def save_render_image(self, save_path: str) -> None:
        """
        将当前场景渲染为图像并保存

        Args:
            save_path: 图像保存路径
        """
        renderer = mujoco.Renderer(self.model, height=480, width=640)
        renderer.update_scene(self.data)
        image = renderer.render()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, image)

    def Base_Cam(self) -> np.ndarray:
        """
        计算相机坐标系在机械臂基座坐标系下的位姿

        Returns:
            np.ndarray: 从基座到相机的4x4变换矩阵 T_BC
        """
        T_HC = np.eye(4)
        T_BH = get_relative_pose(
            self.model, self.data, self.base_name, self.cam_name)
        T_BC = T_BH @ T_HC
        return T_BC

    def Cam_Obj(self) -> np.ndarray:
        """
        计算物体在相机坐标系下的位姿（理论上通过深度信息得到，目前通过仿真环境直接获取）

        Returns:
            np.ndarray: 从相机到物体的4x4变换矩阵 T_CO
        """
        T_OC = get_relative_pose(
            self.model, self.data, self.cam_name, self.obj_name)
        return T_OC

    def Base_Obj(self) -> np.ndarray:
        """
        计算物体在机械臂基座坐标系下的位姿

        Returns:
            np.ndarray: 从基座到物体的4x4变换矩阵 T_BO
        """
        T_BC = self.Base_Cam()
        T_CO = self.Cam_Obj()
        T_BO = T_BC @ T_CO
        return T_BO

    def inverse_kinematics(self, T_BO: np.ndarray) -> List[np.ndarray]:
        """
        计算机械臂的逆运动学，得到各关节位置

        Args:
            T_BO: 抓取物体在机械臂坐标系下的位姿(4x4变换矩阵)

        Returns:
            List[np.ndarray]: 机械臂各关节位置的可能解列表
        """
        trans = T_BO[:3, 3]
        rot = T_BO[:3, :3]
        joint_states_list = self.ik_solver.properIK(trans, rot)
        return joint_states_list

    def get_joint_states(self) -> List[np.ndarray]:
        """
        根据机械臂基座坐标系下的物体位姿计算机械臂各关节位置

        Returns:
            List[np.ndarray]: 机械臂各关节位置的可能解列表
        """
        T_BO = self.Base_Obj()
        joint_states_lists = self.inverse_kinematics(T_BO)
        return joint_states_lists

    def get_joint_state_from_cam(self, action_rot: Optional[np.ndarray] = np.eye(3)) -> List[np.ndarray]:
        """
        根据物体位置计算机械臂关节角度

        Args:
            action_rot: 额外的旋转矩阵，默认为单位矩阵

        Returns:
            List[np.ndarray]: 机械臂关节角度的可能解列表
        """
        obj_PT = self.Base_Obj()
        point3d = obj_PT[:3, 3]
        if action_rot is None:
            action_rot = Rotation.from_euler('zyx', [0, -1.1, 0]).as_matrix()
            joint_states = self.MMK2FIK.get_armjoint_pose_wrt_armbase(
                point3d, self.action, self.arm, None, action_rot)
        else:
            joint_states = self.MMK2FIK.get_armjoint_pose_wrt_armbase(
                point3d, self.action, self.arm)
        return joint_states

    def get_joint_qposnum(self, model: mujoco.MjModel, joint_id: int) -> int:
        """
        获取指定关节的自由度数量

        Args:
            model: MuJoCo模型
            joint_id: 关节ID

        Returns:
            int: 关节的自由度数量
        """
        jnt_type = model.jnt_type[joint_id]
        if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
            return 7  # 位置+四元数，共7个
        elif jnt_type == mujoco.mjtJoint.mjJNT_BALL:
            return 4  # 四元数，共4个
        elif jnt_type in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
            return 1  # 单个DOF
        else:
            raise ValueError(f"Unknown joint type: {jnt_type}")


def main() -> None:
    pass


if __name__ == '__main__':
    main()
