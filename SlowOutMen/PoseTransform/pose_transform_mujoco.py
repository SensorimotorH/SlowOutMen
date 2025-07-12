import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from discoverse.mmk2.mmk2_fik import MMK2FIK
from typing import Union, List, Optional
from SlowOutMen.Envs.grasp_apple import GraspAppleTask
import cv2


YAW_T_CB = {
    "l": {
        "0.2": [[0.70711412,  0.35746235, -0.6100904,   0.23056331],
                [0.,         -0.86280707, -0.50553334, -0.24780282],
                [-0.70709945,  0.35746976, -0.61010306,  0.13793469],
                [0.,          0.,          0.,          1.]],
        "0.4": [[0.70711412,  0.47154315, - 0.52691241,  0.2524332],
                [0., - 0.7451744 - 0.66686964, - 0.22245573],
                [-0.70709945,  0.47155293, - 0.52692334,  0.15980504],
                [0.,          0.,          0.,          1.]],
    },
    "r": {
        "0.2": [[0.70711412,  0.35746235, -0.6100904,   0.23056331],
                [0.,         -0.86280707, -0.50553334, -0.24780282],
                [-0.70709945,  0.35746976, -0.61010306,  0.13793469],
                [0.,          0.,          0.,          1.]],
        "0.4": [[0.70711412,  0.47154315, -0.52691241,  0.2524332],
                [0.,         -0.7451744,  -0.66686964, -0.22245573],
                [-0.70709945,  0.47155293, -0.52692334,  0.15980504],
                [0.,          0.,          0.,          1.]],
    }

}


class PoseTransform:
    """
    用于MuJoCo环境中的坐标变换和逆运动学计算
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """
        初始化坐标变换器

        :param model: MuJoCo模型 (mjModel)
        :param data: MuJoCo数据 (mjData)
        """
        self.model = model
        self.data = data
        self.mmk2_fik = MMK2FIK()
        self.action_rot = MMK2FIK.action_rot.copy()

    def get_relative_pose(self, model: mujoco.MjModel, data: mujoco.MjData,
                          body1_name: str, body2_name: str) -> np.ndarray:
        """
        计算从body1到body2的相对位姿变换

        :param model: MuJoCo模型 (mjModel)
        :param data: MuJoCo数据 (mjData)
        :param body1_name: 第一个物体名称
        :param body2_name: 第二个物体名称
        :return: 4x4的相对变换矩阵
        """
        # 获取物体ID
        body1_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, body1_name)
        body2_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, body2_name)

        if body1_id == -1 or body2_id == -1:
            raise ValueError(
                f"无法找到物体: {body1_name if body1_id == -1 else body2_name}")

        # 获取物体1的全局位姿
        body1_xpos = data.xpos[body1_id].copy()
        body1_xquat = data.xquat[body1_id].copy()

        # 获取物体2的全局位姿
        body2_xpos = data.xpos[body2_id].copy()
        body2_xquat = data.xquat[body2_id].copy()

        # 计算旋转矩阵
        body1_rot = Rotation.from_quat(
            [*body1_xquat[1:], body1_xquat[0]])  # 注意MuJoCo四元数是[w,x,y,z]格式
        body2_rot = Rotation.from_quat([*body2_xquat[1:], body2_xquat[0]])

        # 计算相对旋转：物体2相对于物体1的旋转
        relative_rot = body1_rot.inv() * body2_rot
        relative_rotmat = relative_rot.as_matrix()

        # 计算相对位置：物体2在物体1坐标系中的位置
        relative_pos = body1_rot.inv().apply(body2_xpos - body1_xpos)

        # 构建完整的4x4变换矩阵
        transform = np.eye(4)
        transform[:3, :3] = relative_rotmat
        transform[:3, 3] = relative_pos

        return transform

    def get_T_OC(self, object_name: str, camera_name: str) -> np.ndarray:
        """
        计算物体Object相对于相机Cam的位姿变换 T_OC

        :param object_name: 物体名称
        :param camera_name: 相机名称
        :return: 4x4的相对变换矩阵
        """
        return self.get_relative_pose(self.model, self.data, camera_name, object_name)

    def get_T_CB(self, camera_name: str, base_name: str) -> np.ndarray:
        """
        计算相机Cam相对于机械臂基座Base的位姿变换 T_CB

        :param camera_name: 相机名称
        :param base_name: 机械臂基座名称
        :return: 4x4的相对变换矩阵
        """
        return self.get_relative_pose(self.model, self.data, base_name, camera_name)

    def get_T_OB(self, object_name: str, base_name: str, camera_name: Optional[str] = None) -> np.ndarray:
        """
        计算物体Object相对于机械臂基座Base的位姿变换 T_OB

        :param object_name: 物体名称
        :param base_name: 机械臂基座名称
        :param camera_name: 相机名称 (可选，如果不提供则直接计算物体相对于基座的位姿)
        :return: 4x4的相对变换矩阵
        """
        if camera_name:
            # 通过相机中转计算: T_OB = T_CB * T_OC
            T_OC = self.get_T_OC(object_name, camera_name)
            T_CB = self.get_T_CB(camera_name, base_name)
            return T_CB @ T_OC
        else:
            # 直接计算物体相对于基座的位姿
            return self.get_relative_pose(self.model, self.data, base_name, object_name)

    def get_arm_joint_pose(self, pose3d: np.ndarray,
                           action: Union[str, np.ndarray],
                           arm: str = "r",
                           q_ref: Optional[np.ndarray] = None,
                           action_rot: Optional[np.ndarray] = np.eye(3)) -> List[np.ndarray]:
        """
        根据物体3D坐标计算机械臂关节角度

        :param pose3d: 物体在机械臂基座坐标系下的3D坐标[x, y, z]
        :param action: 动作名称 ("pick", "carry", "look") 或相对于机械臂基座坐标系旋转矩阵
        :param arm: 机械臂选择 ("l" 左臂, "r" 右臂)
        :param q_ref: 参考关节角度, 默认为None
        :param action_rot: 额外的旋转矩阵, 默认为None
        :return: 机械臂关节角度列表
        """
        if action_rot is None:
            action_rot = Rotation.from_euler('zyx', [0, -1.1, 0]).as_matrix()

        if isinstance(action, np.ndarray):
            assert action.shape == (3, 3), "action必须是3x3旋转矩阵"

        return self.mmk2_fik.get_armjoint_pose_wrt_armbase(pose3d, action, arm, q_ref, action_rot)

    def get_arm_joint_pose_full(self, pose6d: np.ndarray,
                                arm: str = "r",
                                q_ref: Optional[np.ndarray] = None,
                                action_rot: Optional[np.ndarray] = np.eye(3)) -> List[np.ndarray]:
        """
        慎用!!!根据完整的6DOF位姿计算机械臂关节角度, 可能会有奇异解!!!

        :param pose6d: 物体在机械臂基座坐标系下的位姿, 4x4变换矩阵
        :param arm: 机械臂选择 ("l" 左臂, "r" 右臂)
        :param q_ref: 参考关节角度, 默认为None
        :return: 机械臂关节角度列表
        """
        if not isinstance(pose6d, np.ndarray) or pose6d.shape != (4, 4):
            raise ValueError("pose6d必须是4x4变换矩阵")

        if action_rot is None:
            action_rot = Rotation.from_euler('zyx', [0, -1.1, 0]).as_matrix()

        pose3d = pose6d[:3, 3]
        action = pose6d[:3, :3]

        return self.mmk2_fik.get_armjoint_pose_wrt_armbase(pose3d, action, arm, q_ref, action_rot)

    def get_arm_joint_pose_from_object(self, object_name: str,
                                       base_name: str,
                                       camera_name: Optional[str] = None,
                                       action: Union[str, np.ndarray] = "pick",
                                       arm: str = "r",
                                       q_ref: Optional[np.ndarray] = None,
                                       action_rot: np.ndarray = np.eye(3)) -> List[np.ndarray]:
        """
        根据物体名称直接计算机械臂关节角度

        :param object_name: 物体名称
        :param base_name: 机械臂基座名称
        :param camera_name: 相机名称 (可选)
        :param action: 动作名称 ("pick", "carry", "look") 或旋转矩阵(可选)
        :param arm: 机械臂选择 ("l" 左臂, "r" 右臂)(可选)
        :param q_ref: 参考关节角度, 默认为None
        :param action_rot: 额外的旋转矩阵
        :return: 机械臂关节角度列表
        """
        # 获取物体相对于机械臂基座的位姿
        T_OB = self.get_T_OB(object_name, base_name, camera_name)

        # 提取物体位置
        pose3d = T_OB[:3, 3]

        # 计算机械臂关节角度
        return self.get_arm_joint_pose(pose3d, action, arm, q_ref, action_rot)


def render_head_rgb(renderer: mujoco.Renderer, data: mujoco.MjData) -> None:
    cv2.namedWindow('Head_a View', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Head_a View', 640, 480)
    while True:
        renderer.update_scene(data)
        img = renderer.render()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('Head_a View', img)
        cv2.waitKey(1)


def main() -> None:
    try:
        # 测试相对位姿计算
        object_name = "apple"  # 假设场景中有一个名为apple的物体
        camera_name = "head_cam"  # 假设相机的名称
        base_name = "lft_arm_base"  # 假设左臂机械臂基座的名称
        action = "pick"  # 假设拾取动作
        arm = "r"  # 假设使用左臂

        grasp_env = GraspAppleTask(object_name, base_name, action)
        grasp_env.reset()
        model = grasp_env.model
        data = grasp_env.data

        data.qpos[11:12] = 0.2
        mujoco.mj_forward(model, data)

        pose_transform = PoseTransform(model, data)
        T_CB = pose_transform.get_T_CB(camera_name, base_name)
        print(f"相机相对于机械臂基座的位姿:\n{T_CB}")

        renderer = mujoco.Renderer(model, 640, 480)
        render_head_rgb(renderer, data)

        # 创建PoseTransform对象
        pose_transform = PoseTransform(model, data)

        print("测试相对位姿计算：")
        try:
            T_OC = pose_transform.get_T_OC(object_name, camera_name)
            print(f"物体相对于相机的位姿:\n{T_OC}")

            T_CB = pose_transform.get_T_CB(camera_name, base_name)
            print(f"相机相对于机械臂基座的位姿:\n{T_CB}")

            T_OB = pose_transform.get_T_OB(object_name, base_name, camera_name)
            print(f"物体相对于机械臂基座的位姿, 通过cam计算:\n{T_OB}")

            T_OB_direct = pose_transform.get_T_OB(object_name, base_name)
            print(f"物体相对于机械臂基座的位姿, 直接计算:\n{T_OB_direct}")

            # 测试逆运动学计算
            position = T_OB[:3, 3]
            print(f"物体在机械臂基座坐标系下的位置: {position}")

            # 使用预设动作计算关节角度
            joint_angles = pose_transform.get_arm_joint_pose(
                position, action, arm, action_rot=None)
            data.qpos[12:18] = joint_angles[0]
            mujoco.mj_forward(model, data)
            grasp_env.save_render_image(
                "SIM2REAL-2025/s2r2025/results/test_1.png")
            print(f"左臂拾取姿态的关节角度: {joint_angles[0]}")

            # 使用完整位姿计算关节角度
            joint_angles_full = pose_transform.get_arm_joint_pose_full(
                T_OB, arm)
            data.qpos[12:18] = joint_angles_full[0]
            mujoco.mj_forward(model, data)
            grasp_env.save_render_image(
                "SIM2REAL-2025/s2r2025/results/test_2.png")
            print(f"左臂使用完整位姿计算的关节角度: {joint_angles_full[0]}")

            # 直接从物体名称计算关节角度
            joint_angles_direct = pose_transform.get_arm_joint_pose_from_object(
                object_name, base_name, camera_name, action, arm)
            data.qpos[12:18] = joint_angles_direct[0]
            mujoco.mj_forward(model, data)
            grasp_env.save_render_image(
                "SIM2REAL-2025/s2r2025/results/test_3.png")
            print(f"左臂直接从物体名称计算的关节角度: {joint_angles_direct[0]}")

        except Exception as e:
            print(f"测试遇到错误: {e}")

    except Exception as e:
        print(f"加载模型失败: {e}")
        print("运行测试需要正确设置模型路径")


if __name__ == "__main__":
    main()
