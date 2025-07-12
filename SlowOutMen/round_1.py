from SlowOutMen.TaskUnderstanding.task_parser import TaskParser
from SlowOutMen.Navigation.utils.move_to_point_ros2 import MoveToPoint
from SlowOutMen.Communication.ros2_mmk2_api import MMK2_Receiver, MMK2_Controller
from VisionUnderstanding.VisiontoPoindcloud.detect_and_segment import yolo_process
from VisionUnderstanding.VisiontoPoindcloud.vision_module import RosVisionModule
from Manipulation.utils.pose_transform_base import PoseTransform
from Manipulation.utils.mmk2.mmk2_fik import MMK2FIK

import rclpy
import cv2
import time
import threading
import numpy as np
from typing import Dict, List, Optional
from scipy.spatial.transform import Rotation as R
import subprocess

"""
比赛round_1的导航任务,目标是移动到目标box的前面固定位置,然后等待抓取任务
1. 解析任务 -> Target: cabinet_index + floor_index + prop_name
2. 走到中间点,调整高度到对应层数 -> [0.4, 0.4, CABINET_POSITION[cabinet_index][-1]] + SLIDE_POSITION[floor_index]
3. YOLO识别box + prop_name -> 识别到目标box/没识别到则选择离画面中心点最近的box
4. YOLO -> 识别到目标box的中心点pos(左右)
5. left_cabinet调整CABINET_POSITION[cabinet_index][0](x+ -> 右), right_cabinet调整CABINET_POSITION[cabinet_index][1](y+ -> 左)
6. 设定目标点 -> 根据5调整CABINET_POSITION对应的位置
7. move_to_point -> 移动到目标点
"""

SLIDE_POSITION = {
    "second": 0.87,
    "third": 0.6,
    "fourth": 0.3,  # 0.3->0.2
}
CABINET_POSITION = {
    "left": [0.5, 0.55, 180],
    "right": [0.55, 0.3, 90],
}
MIDDLE_POINT_POSITION = [0.4, 0.4]


class NavigationtoCabRound1:
    def __init__(self, receiver: MMK2_Receiver, controller: MMK2_Controller, task_parser: TaskParser, vision_module: RosVisionModule):
        self.task_parser = task_parser
        # 存储接收器和控制器的引用
        self.receiver = receiver
        self.controller = controller
        self.move_to_point = MoveToPoint(receiver, controller)
        self.vision_module = vision_module
        self.posetran = PoseTransform(self.receiver)
        # 存储解析后的任务信息
        self.cabinet_index: str = ""
        self.floor_index: str = ""
        self.prop_name: str = ""
        self.tn = "mmk2_pick_prop"
        self.ts = "20250406-155952"

        # 存储目标位置信息
        self.target_position: List[float] = []
        self.current_position: Dict[float] = {
            'x': 0.0,
            'y': 0.0
        }
        self.box_center_x: Optional[float] = None
        self.LEFT_X = 0
        self.RIGHT_Y = 0

    def run(self):
        self._step_1()
        self._observation()
        self._step_2()
        self._step_3()
        self._step_4()
        self._step_5()
        self._step_6()
        self._step_7()
        self._step_8()
        self._step_9_desk()
        self._step_10_yolo()

    def _step_1(self) -> None:
        """
        步骤1: 解析任务,从任务描述中提取cabinet_index, floor_index和prop_name
        """
        print("步骤1: 解析任务")

        # 获取任务指令
        instruction = self.receiver.get_taskinfo()

        # 解析任务
        task_info = self.task_parser.parse_task(instruction)

        self.cabinet_index = task_info['cabinet_side']
        self.floor_index = task_info['floor']
        self.prop_name = task_info['prop']
        self.table_index = task_info['target_table']
        if self.prop_name == "carton":
            self.tn = "mmk2_pick_carton"
            self.ts = "20250411-120702"
        elif self.prop_name == "disk":
            self.tn = "mmk2_pick_disk"
            self.ts = "20250411-110940"
        elif self.prop_name == "sheet":
            self.tn = "mmk2_pick_sheet"
            self.ts = "20250411-142605"

        print(
            f"任务解析n结果: 柜子位置={self.cabinet_index}, 层数={self.floor_index}, 物品={self.prop_name}, 目标桌子={self.table_index}")

    def _step_2(self) -> None:
        """
        步骤2: 走到中间点,调整高度到对应层数
        """
        print("步骤2: 走到中间点,调整高度")
        # self.controller.reset_arm()
        # 设置移动目标点为中间点
        target_middle_point = MIDDLE_POINT_POSITION + \
            [CABINET_POSITION[self.cabinet_index][-1]]
        print(f"移动到中间点: {target_middle_point}")

        # 移动到中间点
        self.move_to_point.move_to_point(target_middle_point)

        # 根据楼层调整滑轨高度
        slide_position = SLIDE_POSITION[self.floor_index]
        print(f"调整滑轨高度到: {slide_position}")
        self.controller.set_head_slide_position(slide_position)

        # 等待调整完成
        print("等待高度调整完成...")
        time.sleep(1.0)

    def _step_3(self) -> None:
        """
        步骤3: YOLO识别box + prop_name,识别到目标box或选择最近的box
        """
        print("步骤3: 识别目标盒子和物品")

        # 获取头部RGB图像
        rgb_image = self.receiver.get_head_rgb()

        # 使用YOLO进行物体检测
        detection_results = yolo_process(rgb_image)

        # 使用VisionModule进行物体检测
        self.box_pose = self.vision_module.get_Toc_box_from_yolo_result(
            detection_results, self.prop_name)

    def _step_4(self) -> None:
        """
        步骤4: 根据相机坐标系下的box位置计算偏移量
        box_pose[0]为x轴位置(左负右正)
        """
        print("步骤4: 计算box位置偏移量")

        if self.box_pose is None:
            print("未能获取box位置,使用默认位置")
            self.box_offset = 0.0
            return

        # box_pose[0]就是相机坐标系下的左右偏移
        self.box_offset = self.box_pose[0]
        print(
            f"box在相机坐标系下的位置: x={self.box_pose[0]:.3f}, y={self.box_pose[1]:.3f}, z={self.box_pose[2]:.3f}")
        print(f"左右偏移量: {self.box_offset:.3f}m")

    def _step_5(self) -> None:
        """
        步骤5: 根据柜子类型调整目标位置
        left_cabinet调整CABINET_POSITION[cabinet_index][0](x+ -> 右)
        right_cabinet调整CABINET_POSITION[cabinet_index][1](y+ -> 左)
        根据相机坐标系下的偏移直接调整位置
        """
        print("步骤5: 根据柜子类型调整目标位置")

        # 获取基础位置
        base_position = MIDDLE_POINT_POSITION.copy()

        # 根据box在相机坐标系下的偏移调整位置
        # 使用一个缩放因子来控制调整幅度
        adjustment_scale = 1.0  # 可以根据实际情况调整这个缩放因子

        if self.cabinet_index == "left":
            # 左柜子调整x坐标
            # 相机坐标系中x为左负右正,机器人坐标系中x正方向为右
            # 所以对于左柜子,可以直接加上偏移量
            base_position[0] += self.box_offset * adjustment_scale
            base_position[1] = CABINET_POSITION[self.cabinet_index][1]
            base_position.append(CABINET_POSITION[self.cabinet_index][-1])
            print(
                f"左柜子调整: x坐标从 {MIDDLE_POINT_POSITION[0]} 调整到 {base_position[0]:.3f}")
        else:
            # 右柜子调整y坐标
            # 相机坐标系的左右偏移需要转换到机器人坐标系的y轴
            # 右柜子面向y轴正方向,相机坐标系右偏移对应y轴负方向
            base_position[1] -= self.box_offset * adjustment_scale
            base_position[0] = CABINET_POSITION[self.cabinet_index][0]
            base_position.append(CABINET_POSITION[self.cabinet_index][-1])
            print(
                f"右柜子调整: y坐标从 {MIDDLE_POINT_POSITION[1]} 调整到 {base_position[1]:.3f}")

        # 存储调整后的位置
        self.adjusted_position = base_position

    def _step_6(self) -> None:
        """
        步骤6: 设定最终目标点
        """
        print("步骤6: 设定最终目标点")

        # 使用步骤5中调整后的位置作为最终目标点
        self.target_position = self.adjusted_position

        print(f"最终目标点: {np.round(np.array(self.target_position).copy(), 2)}")

    def _step_7(self) -> None:
        """
        步骤7: 移动到目标点
        """
        print("步骤7: 移动到目标点")

        # 使用move_to_point移动到目标点
        self.move_to_point.move_to_point(self.target_position)

    def _step_8(self) -> None:
        """
        步骤8: 抓取box
        """
        print("步骤8: 抓取box")
        self.yaw_angle = 0.45
        current_slide_pos = self.receiver.get_joint_states()["positions"][0]

        self.controller.reset_head()
        print(f"current_slide_pos: {current_slide_pos}")
        self.controller.set_head_slide_position(current_slide_pos - 0.2)
        time.sleep(1)
        self.controller.set_head_pitch_angle(self.yaw_angle)
        time.sleep(1)

        # 得到对应盒子的相机坐标下位置以及基座坐标下位置
        box_pose_cam = self.vision_module.get_Toc_box_from_yolo(self.prop_name)
        box_pose_base = self.posetran.pos_cam_to_base(box_pose_cam)
        print(f"box_pose_cam:{box_pose_cam}")
        print(f"box_pose_base:{box_pose_base}")
        box_pose_cam[1] = box_pose_cam[1] + 0.1
        box_pose_base = self.posetran.pos_cam_to_base(box_pose_cam)
        tmp_lft_arm_target_pose = box_pose_base + \
            np.array([0.0, 0.11, 0.05])  # y,x,z
        tmp_rgt_arm_target_pose = box_pose_base + \
            np.array([0.0, -0.11, 0.05])  # y,x,z
        tmp_lft_arm_state = self.getArmEndTarget(tmp_lft_arm_target_pose, "pick", "l", np.array(
            self.receiver.get_joint_states()['positions'][3:9]), R.from_euler("zyx", [np.pi/2, -0.0551 + np.pi, np.pi / 8]).as_matrix())
        tmp_rgt_arm_state = self.getArmEndTarget(tmp_rgt_arm_target_pose, "pick", "r", np.array(
            self.receiver.get_joint_states()['positions'][10:16]), R.from_euler("zyx", [-np.pi/2, -0.0551 + np.pi, -np.pi / 8]).as_matrix())

        self.zhn_rgt = tmp_rgt_arm_state
        self.zhn_lft = tmp_lft_arm_state

        self.controller.set_right_arm_gripper(1.0)
        self.controller.set_left_arm_gripper(1.0)
        time.sleep(1)

        self.controller.set_move_backward_position(0.05)

        self.controller.set_left_arm_position(tmp_lft_arm_state)
        self.controller.set_right_arm_position(tmp_rgt_arm_state)

        time.sleep(1)

        print("第一次调整结束")

        self.controller.set_move_forward_position(0.2)
        print("移动完成,准备抓取box")

        tmp_lft_arm_target_pose = box_pose_base + \
            np.array([0.0, 0.08, 0.05])  # y,x,z
        tmp_rgt_arm_target_pose = box_pose_base + \
            np.array([0.0, -0.08, 0.05])  # y,x,z

        tmp_lft_arm_state = self.getArmEndTarget(tmp_lft_arm_target_pose, "pick", "l", np.array(
            self.receiver.get_joint_states()['positions'][3:9]), R.from_euler("zyx", [np.pi/2, -0.0551 + np.pi, np.pi / 8]).as_matrix())
        tmp_rgt_arm_state = self.getArmEndTarget(tmp_rgt_arm_target_pose, "pick", "r", np.array(
            self.receiver.get_joint_states()['positions'][10:16]), R.from_euler("zyx", [-np.pi/2, -0.0551 + np.pi, -np.pi / 8]).as_matrix())
        time.sleep(1)
        # self.controller.set_left_arm_position(tmp_lft_arm_state)
        # self.controller.set_right_arm_position(tmp_rgt_arm_state)
        
        self.controller.set_arm_position(tmp_lft_arm_state, tmp_rgt_arm_state)
        print("第二次调整结束")
        time.sleep(1)
        
        
        
        self.controller.set_move_backward_position(0.4)
        time.sleep(1)

        time.sleep(1)
        self.controller.set_head_pitch_angle(0)
        time.sleep(1)
        self.controller.set_head_slide_position(0)
        time.sleep(1)

    def _step_9_desk(self) -> None:
        """
        步骤9:读取桌子位置
        """
        print(self.table_index)
        
        if self.table_index == "left":
            
            if self.LEFT_X <= 0.4:
                self.move_to_point.move_to_point([0.4, 0.4, 90])
                self.controller.set_move_backward_position(0.4 - self.LEFT_X)
                self.move_to_point.move_to_point([self.LEFT_X - 0.04, 0.05, 0])
                
            else:
                self.move_to_point.move_to_point([0.4, 0.4, -90])
                self.controller.set_move_backward_position(self.LEFT_X - 0.4)
                self.move_to_point.move_to_point([self.LEFT_X - 0.04, 0.05, 0])

            
        elif self.table_index == "right":
            self.move_to_point.move_to_point([0.4, 0.4, -180])
            self.controller.set_move_backward_position(0.4 - self.RIGHT_Y)
            self.controller.set_turn_left_angle(90)
            self.move_to_point.move_to_point([-0.05, self.RIGHT_Y + 0.04, -90])
            
        for i in range(100):
            self.controller.set_head_slide_position(0.004 * i)
            time.sleep(0.1)

        self.controller.set_arm_position(self.zhn_lft, self.zhn_rgt)
        time.sleep(1)
        
        self.controller.set_move_backward_position(0.3)
        self.controller.reset_arms()
        self.controller.set_head_slide_position(0)

        self.current_position['x'] = self.receiver.get_odom()['position']['x']
        self.current_position['y'] = self.receiver.get_odom()['position']['y']
        
        self.controller.set_head_pitch_angle(0.5)
        
        rgb_image = self.receiver.get_head_rgb()
        detection_results = yolo_process(rgb_image)
        self.box_pose = self.vision_module.get_Toc_box_from_yolo_result(
            detection_results, self.prop_name)
        
        box_pose_base = self.posetran.pos_cam_to_base(self.box_pose)
        box_pose_world = self.posetran.pos_base_to_world(box_pose_base)
        

        if self.table_index == "left":
            self.move_to_point.move_to_point(
                [box_pose_world[0], -0.1, 0])
        elif self.table_index == "right":
            self.move_to_point.move_to_point(
                [-0.2, box_pose_world[1], -90.0])

        self.controller.set_head_pitch_angle(1)

        
        
    def _step_10_yolo(self) -> None:
        
        if self.prop_name == "carton":
            self.grasp_pos = 0.25
        else:
            self.grasp_pos = 0.0
            
        #self.controller.reset_arms()

        self.controller.set_move_backward_position(0.3)
        self.controller.set_head_slide_position(0)
        self.controller.set_head_pitch_angle(0.5)
        
        prop_pose_cam = self.vision_module.get_Toc_from_yolo(self.prop_name)
        prop_pose_base = self.posetran.pos_cam_to_base(prop_pose_cam)
        prop_pose_world = self.posetran.pos_base_to_world(prop_pose_base)
        
        self.controller.set_move_forward_position(0.3)
        
        self.controller.set_head_pitch_angle(1)
                
        prop_pose_base = self.posetran.pos_world_to_base(prop_pose_world)
        # if self.prop_name == "carton":
        #     prop_pose_cam= self.vision_module.get_Toc_box_from_yolo("carton")
        #     prop_pose_base=self.posetran.pos_cam_to_base(prop_pose_cam)
        
        # print(f"prop_pose_base:{prop_pose_base}")
        
        # 伸到目标位置附近，准备抓取
        tmp_lft_arm_target_pose = prop_pose_base + np.array([-0.03, -0.01, 0.15])
        self.controller.set_left_arm_gripper(1.0)
        time.sleep(1)
            
        tmp_lft_arm_state = self.getArmEndTarget(tmp_lft_arm_target_pose, "pick", "l", np.array(
            self.receiver.get_joint_states()['positions'][3:9]), R.from_euler("zyx", [0, -0.0551 , 0.001]).as_matrix())
        
        self.controller.set_left_arm_position(tmp_lft_arm_state)
        time.sleep(1)

        
        #机械爪下移，并合拢
        tmp_lft_arm_target_pose = prop_pose_base + np.array([-0.03, -0.01, 0])
        time.sleep(1)
            
        tmp_lft_arm_state = self.getArmEndTarget(tmp_lft_arm_target_pose, "pick", "l", np.array(
            self.receiver.get_joint_states()['positions'][3:9]), R.from_euler("zyx", [0, -0.0551 , 0.001]).as_matrix())
        print(tmp_lft_arm_state)
        self.controller.set_left_arm_position(tmp_lft_arm_state)
        time.sleep(1)
        self.controller.set_left_arm_gripper(self.grasp_pos)
        time.sleep(1)
        
        #移动手臂方案
        # self.controller.set_move_forward_position(0.1)
        
        # tmp_left_pos=[0.08557662271429829, -0.6966053127563154, 0.6838260011196069, -0.03769022019760482, 1.4458810502319481, 2.348444457790811]
        # self.controller.set_left_arm_position(tmp_left_pos)
        
        # time.sleep(1)
        # self.controller.set_move_forward_position(0.2)
        # time.sleep(1)
        # self.controller.set_left_arm_gripper(1.0)
        
        
        #抬高身体方式
        self.controller.set_head_slide_position(-0.04)
        time.sleep(1)
        self.controller.set_move_backward_position(0.4)
        
        self.controller.set_turn_left_angle(15)
        
        self.controller.set_move_forward_position(0.4)
        
        self.controller.set_left_arm_gripper(1.0)        
        
    
    def _step_10_act(self) -> None:
        self.controller.set_head_slide_position(0)
        self.controller.set_head_pitch_angle(1)
        try:
            print(f"ACT pick {self.prop_name}")
            command = f"cd /workspace/DISCOVERSE/policies/act && python3 policy_evaluate_ros.py -tn {self.tn} -ts {self.ts}"
            subprocess.run(command, shell=True, check=True)
        except Exception as e:
            print(f"ACT execution failed: {str(e)}")

    def getArmEndTarget(self, target_pose, arm_action, arm, q_ref, a_rot):

        self.slide_pos = self.receiver.get_joint_states()["positions"][0]
        try:
            rq = MMK2FIK().get_armjoint_pose_wrt_footprint(
                target_pose, arm_action, arm, self.slide_pos, q_ref, a_rot)
            return rq

        except ValueError as e:
            print(
                f"Failed to solve IK {e} params: arm={arm}, target={target_pose}, slide={self.slide_pos:.2f}")
            return False

    def navigation_to_obj(self, obj_name) -> str:

        self.move_to_point.move_to_point([0.5, 0.5, -65])
        right_result = self.vision_module.is_object_presence(obj_name)
        print(f"right_result: {right_result}")

        self.controller.set_turn_left_angle(55)
        left_result = self.vision_module.is_object_presence(obj_name)

        if left_result == None:
            self.controller.set_turn_right_angle(55)

        obj_pose = self.vision_module.get_Toc_from_yolo(obj_name)
        obj_pose_base = self.posetran.pos_cam_to_base(obj_pose)
        obj_world_pose = self.posetran.pos_base_to_world(obj_pose_base)

        print(right_result,left_result)
        if right_result == True and left_result == None:
            angle = -90
            time.sleep(1)
            self.move_to_point.move_to_point(
                [obj_world_pose[0]+0.7, obj_world_pose[1], angle])
            return "right"
        if left_result == True and right_result == None:
            angle = 0
            self.move_to_point.move_to_point(
                [obj_world_pose[0]-0.08, obj_world_pose[1]+0.7, angle])
            return "left"

    def _observation(self, debug=True) -> None:
        if self.table_index == "left":
            self.move_to_point.move_to_point([0.3, 0.5, 0])
            self.controller.set_head_pitch_angle(0.25)
            time.sleep(1)
        elif self.table_index == "right":
            self.move_to_point.move_to_point([0.3, 0, -90])
            self.controller.set_head_pitch_angle(0.3)
            time.sleep(1)
        vector = self.vision_module.find_space_for_box(
            self.table_index, debug=debug)
        obj_pose_base = self.posetran.pos_cam_to_base(vector)
        obj_pose_world = self.posetran.pos_base_to_world(obj_pose_base)
        if self.table_index == "left":
            self.LEFT_X = obj_pose_world[0]
        elif self.table_index == "right":
            self.RIGHT_Y = obj_pose_world[1]
        self.controller.reset_head()

    def drawer_open(self) -> None:
        obj_name="drawer"
        direction=self.navigation_to_obj(obj_name)
        self.controller.set_head_slide_position(0.2)
        self.controller.set_head_pitch_angle(0.6)
        time.sleep(1)
        self.current_position['x'] = self.receiver.get_odom()['position']['x']
        self.current_position['y'] = self.receiver.get_odom()['position']['y']
        obj_pose = self.vision_module.get_Toc_from_yolo("Linear motion handles",drawer=True)
        obj_pose_base = self.posetran.pos_cam_to_base(obj_pose)
        print(f"obj_pose_base:{obj_pose_base}")
        
        # 伸到目标位置附近，准备抓取
        tmp_lft_arm_target_pose = obj_pose_base + np.array([-0.15,0,0.07])
            
        tmp_lft_arm_state = self.getArmEndTarget(tmp_lft_arm_target_pose, "pick", "l", np.array(
            self.receiver.get_joint_states()['positions'][3:9]), R.from_euler('zyx', [1.5807, 0, 0.6]).as_matrix())
        
        self.controller.set_left_arm_position(tmp_lft_arm_state)
        time.sleep(1)
        self.controller.set_left_arm_gripper(1.0)
        print("伸到目标位置附近，准备抓取")
        
        if direction == "left":
            self.move_to_point.move_to_point(
                [self.current_position['x'], self.current_position['y'], 0])
        if direction == "right":
            self.move_to_point.move_to_point(
                [self.current_position['x'], self.current_position['y'], -90])  
        
        self.controller.set_move_forward_position(0.12)
        print("移动完成,准备抓取物品")
        time.sleep(1)
        self.controller.set_left_arm_gripper(0)
        time.sleep(1)
        self.controller.set_move_backward_position(0.2)
        print("拉开抽屉")
    
    def cabinet_door_open(self) -> None:
        obj_name="drawer"
        direction=self.navigation_to_obj(obj_name)
        self.controller.set_head_slide_position(0.05)
        self.controller.set_head_pitch_angle(0.6)
        time.sleep(1)
        
        self.controller.set_move_backward_position(0.1)
        self.current_position['x'] = self.receiver.get_odom()['position']['x']
        self.current_position['y'] = self.receiver.get_odom()['position']['y']
        
        obj_pose = self.vision_module.get_Toc_from_yolo("rotary handles",drawer=True)
        obj_pose_base = self.posetran.pos_cam_to_base(obj_pose)
        
        tmp_lft_arm_target_pose = obj_pose_base + np.array([-0.15,0,0])
            
        tmp_lft_arm_state = self.getArmEndTarget(tmp_lft_arm_target_pose, "pick", "l", np.array(
            self.receiver.get_joint_states()['positions'][3:9]), R.from_euler('zyx', [0, -1.5807, -1.5807]).as_matrix())
        
        self.controller.set_left_arm_position(tmp_lft_arm_state)
        time.sleep(1)
        self.controller.set_left_arm_gripper(1.0)
        self.controller.set_move_forward_position(0.13)
        self.controller.set_left_arm_gripper(0.0)
        time.sleep(1)
        self.controller.set_move_backward_position(0.2)
        print("拉开柜门")
        
        if direction == "left":
            self.move_to_point.move_to_point([self.current_position['x']-0.2, self.current_position['y']-0.1, 90])
        elif direction == "right":
            self.move_to_point.move_to_point([self.current_position['x']-0.1, self.current_position['y']+0.2, 0])
        
    def test(self)->None:
        self.controller.set_head_slide_position(0)
        pos_cam=[0.032, 0.014, 0.807]
        prop_pose_base = self.posetran.pos_cam_to_base(pos_cam)
        
        # #机械爪下移，并合拢
        # tmp_lft_arm_target_pose = prop_pose_base + np.array([-0.03, -0.01, 0])
        # time.sleep(1)
            
        # tmp_lft_arm_state = self.getArmEndTarget(tmp_lft_arm_target_pose, "pick", "l", np.array(
        #     self.receiver.get_joint_states()['positions'][3:9]), np.eye(3))
        
        # self.controller.set_left_arm_position(tmp_lft_arm_state)
        # time.sleep(1)
        # self.controller.set_left_arm_gripper(0.0)
        # time.sleep(1)
        
        
        tmp_lft_arm_target_pose = prop_pose_base + np.array([-0.03, 0.1, 0.15])
        time.sleep(1)
            
        tmp_lft_arm_state = self.getArmEndTarget(tmp_lft_arm_target_pose, "pick", "l", np.array(
            self.receiver.get_joint_states()['positions'][3:9]), R.from_euler("zyx", [0, -0.0551 , 0.001]).as_matrix())
        
        self.controller.set_left_arm_gripper(0.0)
        print("关闭夹爪")
        time.sleep(10)
        self.controller.set_left_arm_position(tmp_lft_arm_state)
        time.sleep(1)
        self.controller.set_left_arm_gripper(1.0)
        
def render_head_rgb(receiver: MMK2_Receiver, stop_event: threading.Event):
    cv2.namedWindow('Head_a View', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Head_a View', 640, 480)
    while not stop_event.is_set():
        img = receiver.get_head_rgb()
        cv2.imshow('Head_a View', img)
        cv2.waitKey(1)
    cv2.destroyAllWindows()


def main():
    rclpy.init()
    receiver = MMK2_Receiver()
    controller = MMK2_Controller(receiver)
    task_parser = TaskParser()
    vision_module = RosVisionModule(receiver, controller)
    navigation = NavigationtoCabRound1(
        receiver, controller, task_parser, vision_module)
    
    
    stop_event = threading.Event()
    render_thread = threading.Thread(
        target=render_head_rgb,
        args=(receiver, stop_event)
    )
    render_thread.daemon = True
    render_thread.start()


    try:
        
        navigation.run()
        # navigation._step_10_act()
    finally:
        # 按照正确的顺序关闭
        # stop_event.set()
        controller.stop_all()
        # 确保在最后关闭 ROS2
        rclpy.shutdown()


if __name__ == "__main__":
    main()
