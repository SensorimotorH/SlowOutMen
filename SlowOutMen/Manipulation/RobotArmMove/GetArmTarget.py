from SlowOutMen.Communication.ros2_mmk2_api import MMK2_Receiver
import numpy as np
from typing import List, Union



YAW_T_CB = {
    "l": {
        "0.0": np.array([
            [7.07114116e-01, 2.29130646e-01, -6.68946017e-01, 2.05568628e-01],
            [2.77555756e-17, -9.46042344e-01, -3.24043028e-01, -2.66500000e-01],
            [-7.07099446e-01, 2.29135400e-01, -6.68959896e-01, 1.12939487e-01],
            [0., 0., 0., 1.]
        ]),
        "0.2": np.array([
            [0.70711412,  0.35746235, -0.6100904,   0.23056331],
            [0.,         -0.86280707, -0.50553334, -0.24780282],
            [-0.70709945,  0.35746976, -0.61010306,  0.13793469],
            [0.,          0.,          0.,          1.]
        ]),
        "0.4": np.array([
            [0.70711412,  0.47154315, - 0.52691241,  0.2524332],
            [0., - 0.7451744, - 0.66686964, - 0.22245573],
            [-0.70709945,  0.47155293, - 0.52692334,  0.15980504],
            [0.,          0.,          0.,          1.]
        ])
    },
    "r": {
        "0.0": np.array([
            [-0.70710892, 0.22913233, -0.66895093, 0.256108037],
            [0., 0.94604234, 0.32404303, 0.2665],
            [0.70710464, 0.22913372, -0.66895498, 0.06202843],
            [0., 0., 0., 1.]
        ]),
        "0.2": np.array([
            [0.70711412,  0.35746235, -0.6100904,   0.23056331],
            [0.,         -0.86280707, -0.50553334, -0.24780282],
            [-0.70709945,  0.35746976, -0.61010306,  0.13793469],
            [0.,          0.,          0.,          1.]
        ]),
        "0.4": np.array([
            [0.70711412,  0.47154315, -0.52691241,  0.2524332],
            [0.,         -0.7451744,  -0.66686964, -0.22245573],
            [-0.70709945,  0.47155293, -0.52692334,  0.15980504],
            [0.,          0.,          0.,          1.]
        ])
    }

}


class ArmPlanner():
    def __init__(self, receiver: MMK2_Receiver):
        
        self.receiver = receiver
        self.pos = None
        self.action = None

    def base_cam(self, arm: str):
        """
        计算相机坐标系在机械臂基座坐标系下的位姿
        """
        T_CB = {
            "l": np.array([
                [7.07114116e-01, 2.29130646e-01, -6.68946017e-01, 2.05568628e-01],
                [2.77555756e-17, -9.46042344e-01, -
                    3.24043028e-01, -2.66500000e-01],
                [-7.07099446e-01, 2.29135400e-01, -6.68959896e-01, 1.12939487e-01],
                [0., 0., 0., 1.]
            ]),
            "r": np.array([
                [-0.70710892, 0.22913233, -0.66895093, 0.256108037],
                [0., 0.94604234, 0.32404303, 0.2665],
                [0.70710464, 0.22913372, -0.66895498, 0.06202843],
                [0., 0., 0., 1.]
            ])
        }
        return T_CB[arm]

    def base_cam_yaw(self, arm: str, yaw: float):
        """
        计算相机坐标系在机械臂基座坐标系下的位姿
        """
        assert arm in ["l", "r"], "arm error( Neither 'l' nor 'r' )"
        assert yaw in [
            0.0, 0.2, 0.4], "yaw error( Neither 0.0 nor 0.2 nor 0.4 )"
        T_CB = YAW_T_CB[arm][str(yaw)]
        return T_CB

    def getarmtarget(self, T_arget, arm, action: Union[str, np.ndarray] = None, action_rot: np.ndarray = None) -> List[List]:
        
        T_arget = np.array(T_arget)
        if T_arget.shape == (4, 4):
            self.pos = T_arget[:3, 3]
            if action is None:
                self.action = T_arget[:3, :3]
                raise Warning("使用完整转换矩阵进行运动学反解")
            else:
                self.action = action
        elif T_arget.shape == (3,) and action is not None:
            self.pos = T_arget
            self.action = action
        else:
            raise TypeError("未检测到action,请检查")

        if arm == "l":
            # T_CB = self.posetran.get_T_CB(base_name='lftarm_base')
            q_ref = np.array(self.receiver.get_joint_states()
                             ["positions"][3:9])
        elif arm == "r":
            # T_CB = self.posetran.get_T_CB(base_name='rgtarm_base')
            q_ref = np.array(self.receiver.get_joint_states()
                             ["positions"][10:16])
        else:
            raise TypeError("arm error( Neither 'l' nor 'r' )")

        T_OC = np.eye(4)
        T_OC[:3, 3] = self.pos
        T_OC[:3, :3] = np.eye(3)

        # T_CB = self.base_cam_yaw(arm, yaw)
        T_CB = self.posetran.get_T_CB(arm)
        T_OB = T_CB @ T_OC
        self.pos = T_OB[:3, 3]

        target_arm_states = self.posetran.get_arm_joint_pose(
            pose3d=self.pos, q_ref = q_ref, action=self.action, arm=arm)

        return target_arm_states
    def __del__(self):
        """确保清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown()
            if hasattr(self, 'executor_thread'):
                self.executor_thread.join()

    def stop(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown()
            if hasattr(self, 'executor_thread'):
                self.executor_thread.join()
