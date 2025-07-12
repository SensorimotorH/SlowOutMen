import numpy as np
from SlowOutMen.Communication.mujoco_mmk2_receiver_api import MMK2_Receiver
from SlowOutMen.Communication.mujoco_mmk2_controller_api import MMK2_Controller
from SlowOutMen.PoseTransform.pose_transform_mujoco import PoseTransform
import mujoco
from scipy.spatial.transform import Rotation


POSITIVE_DERECTION = [0.707, 0,  0, -0.707]
INPUT_KEY = ["x", "y", "theta"]


class MoveToPoint:
    def __init__(self, source: dict, target: dict, receiver: MMK2_Receiver, controller: MMK2_Controller):
        assert all([key in source for key in INPUT_KEY]), f"source should contain all keys in {INPUT_KEY}"
        self.soucre = source
        self.target = target
        self.receiver = receiver
        self.controller = controller
        
        self.x_1 = source["x"]
        self.x_2 = target["x"]
        self.y_1 = source["y"]
        self.y_2 = target["y"]
        self.theta_1 = source["theta"]
        self.theta_2 = target["theta"]
        
        self.turn_angle_1 = None
        
    def _turn_1(self):
        turn_angle_1 = 90 - self.theta_1 + np.rad2deg(np.arctan((self.y_1 - self.y_2) / (self.x_1 - self.x_2)))
        self.turn_angle_1 = turn_angle_1
        self.controller.set_turn_left_angle(turn_angle_1)
    def _move(self):
        move_distance = np.sqrt((self.y_1 - self.y_2)**2 + (self.x_1 - self.x_2)**2)
        self.controller.set_move_forward_pos(move_distance)
        
    def _turn_2(self):
        assert self.turn_angle_1 is not None, "_turn_1 should be called first"
        turn_angle_2 = self.theta_2 - self.theta_1 -self.turn_angle_1
        self.controller.set_turn_left_angle(turn_angle_2)
        
    def move_to_point(self):
        self._turn_1()
        self._move()
        self._turn_2()
        self.controller.stop_all()
        

class Trans:
    def __init__(self, controller: MMK2_Controller, pose_transform: PoseTransform, data: mujoco.MjData, body_name: str = "box_carton"):
        self.controller = controller
        self.pose_transform = pose_transform
        self.data = data
        self.body_name = body_name
        
        self.left_arm_name = "lft_arm_base"
        self.right_arm_name = "rgt_arm_base"
        
        self.action = "pick"
        self.left_action_rot = Rotation.from_euler('zyx', [ np.pi / 2, -0.0551 + np.pi, np.pi / 8]).as_matrix()
        self.right_action_rot = Rotation.from_euler('zyx', [ -np.pi / 2, -0.0551 + np.pi, -np.pi / 8]).as_matrix()
    
    def get_body_tmat(self):
        tmat = np.eye(4)
        tmat[:3,:3] = Rotation.from_quat(self.data.body(self.body_name).xquat[[1,2,3,0]]).as_matrix()
        tmat[:3,3] = self.data.body(self.body_name).xpos
        return tmat
    
    def pick_box(self):
        tmat_box = self.get_body_tmat()
        self.controller.set_head_slide_pos(1.22 - 1.08 * tmat_box[2, 3])

        tmat_box = self.get_body_tmat()
        left_T_OB = self.pose_transform.get_T_OB(self.body_name, self.left_arm_name)
        right_T_OB = self.pose_transform.get_T_OB(self.body_name, self.right_arm_name) 
        left_target_pos = left_T_OB[:3, 3] + np.array([0.0, 0.15, 0.13])
        right_target_pos = right_T_OB[:3, 3] + np.array([0.0,-0.15, 0.13])
        
        left_arm_action = self.pose_transform.get_arm_joint_pose(left_target_pos, self.action, "l", action_rot=self.left_action_rot)
        right_arm_action = self.pose_transform.get_arm_joint_pose(right_target_pos, self.action, "r", action_rot=self.right_action_rot)
        
        self.controller.set_left_arm_pos(left_arm_action[0])
        self.controller.set_right_arm_pos(right_arm_action[0])
    
    
def main():
    model = mujoco.MjModel.from_xml_path("/workspace/DISCOVERSE/models/mjcf/s2r2025_env.xml")
    data = mujoco.MjData(model)
    
    receiver = MMK2_Receiver(model, data)
    controller = MMK2_Controller(model, data, receiver)
    pose_transform = PoseTransform(model, data)
    trans = Trans(controller, pose_transform, data)
    
    source = {
        "x" : 0,
        "y" : 0, 
        "theta" : 90
    }
    
    target = {
        "x" : 0.40,
        "y" : 0.40, 
        "theta" : 90
    }
    
    controller.set_move_forward_pos(-0.4)
    controller.set_head_slide_pos(1.22 - 1.08 * 0.742)
    
    # move_to_point = MoveToPoint(source, target, receiver, controller)
    # move_to_point.move_to_point()
    
    # trans.pick_box()
    
if __name__ == "__main__":
    main()
    