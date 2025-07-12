import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import time

def get_relative_pose(model, data, body1_name, body2_name):
    """
    计算从body1到body2的相对位姿变换
    
    :param model: MuJoCo模型 (mjModel)
    :param data: MuJoCo数据 (mjData)
    :param body1_name: 第一个物体名称
    :param body2_name: 第二个物体名称
    :return: 4x4的相对变换矩阵
    """
    # 获取物体ID
    body1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body1_name)
    body2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body2_name)
    
    if body1_id == -1 or body2_id == -1:
        raise ValueError(f"无法找到物体: {body1_name if body1_id == -1 else body2_name}")
    
    # 获取物体1的全局位姿
    body1_xpos = data.xpos[body1_id].copy()
    body1_xquat = data.xquat[body1_id].copy()
    
    # 获取物体2的全局位姿
    body2_xpos = data.xpos[body2_id].copy()
    body2_xquat = data.xquat[body2_id].copy()
    
    # 计算旋转矩阵
    body1_rot = Rotation.from_quat([*body1_xquat[1:], body1_xquat[0]])  # 注意MuJoCo四元数是[w,x,y,z]格式
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


def main():
    # 加载MuJoCo模型
    model = mujoco.MjModel.from_xml_path('/workspace/DISCOVERSE/models/mjcf/mmk2_floor.xml')
    data = mujoco.MjData(model)
    
    # 相机和末端执行器的名称
    camera_name = "head_cam"
    end_effector_name = "lft_finger_right_link"
    
    # 初始化物理仿真
    mujoco.mj_resetData(model, data)
    
    # 仿真循环
    for i in range(100):
        mujoco.mj_step(model, data)
        if i % 10 == 0:
            try:
                # 计算相机到末端执行器的相对位姿
                camera_to_ee = get_relative_pose(model, data, camera_name, end_effector_name)
                
                # 输出变换矩阵
                print(f"相机到末端执行器的变换矩阵:\n{camera_to_ee}")
                
                # 分解变换矩阵为平移和旋转
                translation = camera_to_ee[:3, 3]
                rotation = Rotation.from_matrix(camera_to_ee[:3, :3])
                euler = rotation.as_euler('xyz', degrees=True)
                
                print(f"相对平移 (x,y,z): {translation}")
                print(f"相对旋转 (欧拉角 xyz): {euler}")
                
            except ValueError as e:
                print(f"错误: {e}")
                
        time.sleep(0.01)


if __name__ == "__main__":
    main()
