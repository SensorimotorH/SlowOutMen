from SlowOutMen.TaskUnderstanding.task_parser import TaskParser
from SlowOutMen.Navigation.utils.move_to_point_ros2 import MoveToPoint
from SlowOutMen.Communication.ros2_mmk2_api import MMK2_Receiver, MMK2_Controller
from SlowOutMen.VisionUnderstanding.VisiontoPoindcloud.detect_and_segment import yolo_process
from SlowOutMen.VisionUnderstanding.VisiontoPoindcloud.vision_module import RosVisionModule
from SlowOutMen.PoseTransform.pose_transform_ros2 import PoseTransform

import rclpy
import cv2
import time
import threading
import numpy as np
from typing import List
from scipy.spatial.transform import Rotation as R
import subprocess


"""
比赛 Round 3 任务：机器人需打开指定层级的储物单元识别内部物品(prop)，找到并抓取装有该prop的盒子运至目标桌子旁，取出prop放置在参照物的指定方位
1. 解析任务 -> target_object, layer_index, target_direction
    1.1 解析任务指令，确定目标参照物(target_object)、储物单元层级(layer_index: bottom/top)、放置方位(target_direction)
2. 寻找目标参照物 -> table_index, target_object_position_wrt_world
    2.1 依次观察左右两张桌子
    2.2 当yolo检测到target_object时，记录其世界坐标(target_object_position_wrt_world)和所在桌子(table_index)
3. 打开储物单元并识别Prop -> self.prop_name
    3.1 导航至指定储物单元前
    3.2 根据layer_index执行操作：
        3.2.1 若为bottom：调用_drawer_open()定位把手拉开抽屉
        3.2.2 若为top：调用_cabinet_door_open()定位把手打开柜门
    3.3 视觉识别储物单元内部物品，确定prop名称(self.prop_name)
4. 定位Prop Box -> cabinet_index, floor_index
    4.1 移动至场地中心点，面向中央柜区
    4.2 依次观察左右柜子的各层(second/third/fourth)
    4.3 当yolo检测到包含self.prop_name的box时，记录柜号(cabinet_index)和层数(floor_index)
5. 抓取Prop Box并移至目标桌子
    5.1 导航至cabinet_index和floor_index指定的柜层前
    5.2 调用_hug_box()：
        5.2.1 视觉微调位置对准目标box
        5.2.2 双臂协同抓取box并取出
    5.3 调用_move_to_target_table()：
        5.3.1 根据table_index移动到目标桌子前
    5.4 放置box并调整姿态：
        5.4.1 控制手臂将box放置桌面
        5.4.2 后退并调整头部姿态，视觉精确定位box位置
6. 抓取Prop并放置
    6.1 调用_grasp_prop_via_yolo()：
        6.1.1 视觉定位box内的self.prop_name
        6.1.2 左臂抓取prop并取出
    6.2 调用_get_target_position_wrt_base()：
        6.2.1 将target_object_position_wrt_world转换为基坐标系
        6.2.2 结合target_direction计算最终放置点
    6.3 控制左臂将prop移动到目标放置点后松开
"""


SLIDE_POSITION = {
    "second": 0.87,
    "third": 0.6,
    "fourth": 0.3
}

CABINET_POSITION = {
    "left": [0.5, 0.55, 180],
    "right": [0.55, 0.3, 90]
}

TABLE_POSITION = {
    "left": [0.0, 0.1, 0],
    "right": [0.0, 0.0, -90]
}

MIDDLE_POINT_POSITION = [0.4, 0.4]

TARGET_DIRECTION = {
    "on ": 0.10,
    "left ": 0.15,
    "right ": 0.15,
    "front ": 0.15,
    "behind ": 0.15,
    "back": 0.15,
    "in ": 0.10
}

OBSERVE_RIGHT_TABLE_POSITION = [0.5, 0.5, -65]


class Round3():
    def __init__(self, receiver: MMK2_Receiver, controller: MMK2_Controller, task_parser: TaskParser, vision_module: RosVisionModule):
        self.receiver = receiver
        self.controller = controller
        self.task_parser = task_parser
        self.vision_module = vision_module
        self.move_to_point = MoveToPoint(self.receiver, self.controller)
        self.pose_transform = PoseTransform(self.receiver)

        # 任务相关变量
        self.prop_name = None
        self.texture_name = None
        self.target_object = None
        self.target_direction = None

        # 位置相关变量
        self.table_index = None
        self.cabinet_index = None
        self.floor_index = None
        self.target_position_wrt_base = None
        self.target_position_wrt_world = None
        self.left_table_x = 0.0
        self.right_table_y = 0.0
        self.target_object_position_wrt_world = None
        self.current_position = {
            'x': 0.0,
            'y': 0.0
        }

    def run(self):
        self._step_1()
        self._step_2()
        if self.layer_index == "bottom":
            self._drawer_open()
        elif self.layer_index == "top":
            self._cabinet_door_open()
        self._step_3()
        self._step_4()
        self._step_5()
        
    def _step_1(self):
        """解析任务,从任务描述中提取prop,texture,target_object和target_directions"""
        print("步骤1: 解析任务")

        instruction = self.receiver.get_taskinfo()
        task_info = self.task_parser.parse_task(instruction)

        self.target_object = task_info['target_object']
        self.layer_index  = task_info['layer_index']
        if self.target_object == "cup":
            self.target_object = "teacup"
        elif self.target_object == "wood":
            self.target_object = "plate"
        self.target_direction = task_info['target_direction']
        
        if self.prop_name == "carton":
            self.tn = "mmk2_pick_carton"
            self.ts = "20250411-120702"
        elif self.prop_name == "disk":
            self.tn = "mmk2_pick_disk"
            self.ts = "20250411-110940"
        elif self.prop_name == "sheet":
            self.ts = "mmk2_pick_sheet"
            self.tn = "20250411-142605"

        assert self.target_direction in TARGET_DIRECTION.keys(
        ), f"目标方位 {self.target_direction} 不在预设范围内, 请检查任务描述"

        print(
            f"任务解析结果: 抽屉层数 = {self.layer_index}, 目标物体={self.target_object}, 放置方位={self.target_direction}")

    def _step_2(self):
        """找到指定参照物,获取位置信息"""
        # TODO 根据is_target_object_finded进行多次寻找
        print("步骤2: 寻找目标物体")

        find_target_object_pitch_angle = 0.25
        is_target_object_finded = False
        self.move_to_point.move_to_point(OBSERVE_RIGHT_TABLE_POSITION)
        time.sleep(1)
        self.controller.set_head_pitch_angle(find_target_object_pitch_angle)
        time.sleep(1)

        right_result = self.vision_module.is_object_presence(
            self.target_object)
        if right_result:
            is_target_object_finded = True
            self.table_index = "right"
            object_position_wrt_camera = self.vision_module.get_Toc_from_yolo(
                self.target_object)
            object_position_wrt_base = self.pose_transform.transform_position_wrt_camera_to_base(
                object_position_wrt_camera)
            object_position_wrt_world = self.pose_transform.transform_position_wrt_base_to_world(
                object_position_wrt_base)

            # 目标位置的y坐标和参照物的y坐标一致,用于调整右桌子的左右偏移位置
            self.right_table_y = object_position_wrt_world[1]
            self.target_object_position_wrt_world = object_position_wrt_world

        else:
            self.controller.set_turn_left_angle(55)
            left_result = self.vision_module.is_object_presence(
                self.target_object)
            if left_result:
                is_target_object_finded = True
                self.table_index = "left"
                object_position_wrt_camera = self.vision_module.get_Toc_from_yolo(
                    self.target_object)
                object_position_wrt_base = self.pose_transform.transform_position_wrt_camera_to_base(
                    object_position_wrt_camera)
                object_position_wrt_world = self.pose_transform.transform_position_wrt_base_to_world(
                    object_position_wrt_base)

                # 目标位置的x坐标和参照物的x坐标一致,用于调整左桌子的左右偏移位置
                self.left_table_x = object_position_wrt_world[0]
                self.target_object_position_wrt_world = object_position_wrt_world
            else:
                raise (f"未找到目标物体 {self.target_object}")

        self.controller.reset_head()

    def _step_3(self):
        """走到中心点寻找目标prop"""
        print("步骤3: 寻找目标物品")

        # 观察左柜子
        self.move_to_point.move_to_point(
            MIDDLE_POINT_POSITION + [CABINET_POSITION["left"][-1]])
        for floor in ["second", "third", "fourth"]:
            self.controller.set_head_slide_position(SLIDE_POSITION[floor])
            time.sleep(1)
            if self.vision_module.is_object_presence(self.prop_name):
                self.cabinet_index = "left"
                self.floor_index = floor
                break

        if not self.cabinet_index:
            # 观察右柜子
            self.controller.set_turn_right_angle(90)
            for floor in ["second", "third", "fourth"]:
                self.controller.set_head_slide_position(SLIDE_POSITION[floor])
                time.sleep(1)
                if self.vision_module.is_object_presence(self.prop_name):
                    self.cabinet_index = "right"
                    self.floor_index = floor
                    break

    def _step_4(self):
        """抓取prop box并移动到目标table前"""
        if not self.cabinet_index or not self.floor_index:
            print("未找到目标物品")
            return

        # 移动到目标柜子前
        target_point = MIDDLE_POINT_POSITION + \
            [CABINET_POSITION[self.cabinet_index][-1]]
        self.move_to_point.move_to_point(target_point)
        self.controller.set_head_slide_position(
            SLIDE_POSITION[self.floor_index])
        time.sleep(1)

        # 执行抓取过程
        self._hug_box()

        # 移动到目标桌子,面对参照物
        self._move_to_target_table()

        self.controller.set_head_slide_position_smooth(0.4)
        time.sleep(1)

        # 打开手臂释放box
        self.controller.set_arm_position(
            self.left_arm_ready_joint_pose, self.right_arm_ready_joint_pose)
        time.sleep(1)
        
        # 后退并上升头部俯视观察
        self.controller.set_move_backward_position(0.3)
        self.controller.reset_arms()
        self.controller.set_head_slide_position_smooth(0)
        self.controller.set_head_pitch_angle(0.5)
        time.sleep(1)
        
        # 计算box位置并微调以补偿位置误差
        head_camera_rgb_image = self.receiver.get_head_rgb()
        yolo_detection_results = yolo_process(head_camera_rgb_image)
        target_box_position_wrt_camera = self.vision_module.get_Toc_box_from_yolo_result(
            yolo_detection_results, self.prop_name)
        
        target_box_position_wrt_base = self.pose_transform.transform_position_wrt_camera_to_base(target_box_position_wrt_camera)
        target_box_position_wrt_world = self.pose_transform.transform_position_wrt_base_to_world(target_box_position_wrt_base)
        
        if self.table_index == "left":
            self.move_to_point.move_to_point(
                [target_box_position_wrt_world[0], -0.1, 0])
        elif self.table_index == "right":
            self.move_to_point.move_to_point(
                [-0.2, target_box_position_wrt_world[1], -90.0])

        # 低头准备抓取
        self.controller.set_head_pitch_angle(1)

    def _step_5(self):
        """放下box,抓取prop并放置"""
        print("步骤5: 放置物品")

        # 抓取prop
        try:
            self._grasp_prop_via_yolo()
        except Exception as e:
            print(f"执行抓取失败: {e}")
            
        # 计算目标放置位置
        assert self.target_object_position_wrt_world is not None, "target_object_position_wrt_world is None, this method should be used AFTER _step_2"
        target_object_position_wrt_base = self.pose_transform.transform_position_wrt_base_to_world(self.target_object_position_wrt_world)
        target_position_wrt_base = self._get_target_position_wrt_base(target_object_position_wrt_base)
        target_joint_pose_wrt_base = self.pose_transform.get_arm_joint_pose(target_position_wrt_base, "pick", "l", action_rot=R.from_euler("zyx", [np.pi/2, -0.0551 + np.pi, np.pi / 8]).as_matrix())
        self.controller.set_left_arm_position(target_joint_pose_wrt_base)
        time.sleep(1)
        
        # 松开夹爪
        self.controller.set_left_arm_gripper(1.0)
            
    def _get_target_position_wrt_base(self, target_object_position_wrt_base: List[float]):
        """
        计算目标放置位置
        1. 根据target_direction和target_object_position_wrt_base计算目标放置位置target_position_wrt_base
        2. 将目标放置位置转换为世界坐标系下的位置
        """
        if self.target_direction == "on ":
            target_position_wrt_base = target_object_position_wrt_base + np.array(
                [0.0, 0.0, TARGET_DIRECTION["on "]])
        elif self.target_direction == "left ":
            target_position_wrt_base = target_object_position_wrt_base + np.array(
                [0.0, TARGET_DIRECTION["left "], 0.0])
        elif self.target_direction == "right ":
            target_position_wrt_base = target_object_position_wrt_base + np.array(
                [0.0, -TARGET_DIRECTION["right "], 0.0])
        elif self.target_direction == "front ":
            target_position_wrt_base = target_object_position_wrt_base + np.array(
                [-TARGET_DIRECTION["front "], 0.0, 0.0])
        elif self.target_direction == "behind ":
            target_position_wrt_base = target_object_position_wrt_base + np.array(
                [TARGET_DIRECTION["behind "], 0.0, 0.0])
        elif self.target_direction == "back":
            target_position_wrt_base = target_object_position_wrt_base + np.array(
                [TARGET_DIRECTION["back"], 0.0, 0.0])
        elif self.target_direction == "in ":
            target_position_wrt_base = target_object_position_wrt_base + np.array(
                [0.0, 0.0, TARGET_DIRECTION["in "]])
        
        # target_position_wrt_base每一位都变成float
        target_position_wrt_base = [float(i) for i in target_position_wrt_base]
        return target_position_wrt_base
    
    def _move_to_target_table(self):
        """
        移动到目标桌子前
        1. 根据table_index和target_position_wrt_base计算目标放置位置
        2. 将目标放置位置转换为世界坐标系下的位置
        3. 移动到目标桌子前
        """
        assert self.table_index is not None, "table_index is None, this method should be used AFTER _step_2"
        if self.table_index == "left":
            assert self.left_table_x is not None, "left_table_x is None, this method should be used AFTER _step_2"
            if self.left_table_x <= 0.4:
                self.move_to_point.move_to_point([0.4, 0.4, 90])
                self.controller.set_move_backward_position(0.4 - self.left_table_x)
                self.move_to_point.move_to_point([self.left_table_x, 0.05, 0])
            else:
                self.move_to_point.move_to_point([0.4, 0.4, -90])
                self.controller.set_move_backward_position(self.left_table_x - 0.4)
                self.move_to_point.move_to_point([self.left_table_x, 0.05, 0])

        elif self.table_index == "right":
            assert self.right_table_y is not None, "right_table_y is None, this method should be used AFTER _step_2"
            self.move_to_point.move_to_point([0.4, 0.4, -180])
            self.controller.set_move_backward_position(0.4 - self.right_table_y)
            self.controller.set_turn_left_angle(90)
            self.move_to_point.move_to_point([-0.05, self.right_table_y, -90])
            
    def _grasp_prop_via_yolo(self) -> None:
        if self.prop_name == "carton":
            gripper_position = 0.25
        else:
            gripper_position = 0.0
            
        self.controller.set_move_backward_position(0.3)
        self.controller.set_head_slide_position(0)
        self.controller.set_head_pitch_angle(1.0)
        time.sleep(1)
        
        target_prop_position_wrt_camera = self.vision_module.get_Toc_from_yolo(self.prop_name)
        target_prop_position_wrt_base = self.pose_transform.transform_position_wrt_camera_to_base(target_prop_position_wrt_camera)
        target_prop_position_wrt_world = self.pose_transform.transform_position_wrt_base_to_world(target_prop_position_wrt_base)
        self.controller.set_move_forward_position(0.3)
        self.controller.set_head_pitch_angle(1)
        target_prop_position_wrt_base = self.pose_transform.transform_position_wrt_world_to_base(target_prop_position_wrt_world)
        print(f"target_prop_position_wrt_base: {target_prop_position_wrt_base}")
        
        # 伸到目标位置附近，准备抓取
        left_arm_grasp_target_pose_wrt_base = target_prop_position_wrt_base + np.array([-0.03, -0.01, 0.15])
        self.controller.set_left_arm_gripper(1.0)
        time.sleep(1)
            
        left_arm_grasp_target_joint_pose = self.pose_transform.get_arm_joint_pose(left_arm_grasp_target_pose_wrt_base, "pick", "l", action_rot=R.from_euler("zyx", [0, -0.0551 , 0.001]).as_matrix())
        self.controller.set_left_arm_position(left_arm_grasp_target_joint_pose)
        time.sleep(1)
        
        # 夹爪下移，并合拢
        left_arm_grasp_target_pose_wrt_base = target_prop_position_wrt_base + np.array([-0.03, -0.01, 0.0])
        time.sleep(1)
        left_arm_grasp_target_joint_pose = self.pose_transform.get_arm_joint_pose(left_arm_grasp_target_pose_wrt_base, "pick", "l", action_rot=R.from_euler("zyx", [0, -0.0551 , 0.001]).as_matrix())
        
        self.controller.set_left_arm_position(left_arm_grasp_target_joint_pose)
        time.sleep(1)
        self.controller.set_left_arm_gripper(gripper_position)
        time.sleep(1)
        
        # 通过抬高身体让夹爪上移
        self.controller.set_head_slide_position_smooth(-0.04)
        self.controller.set_move_backward_position(0.4)
        self.controller.set_turn_left_angle(15)
        self.controller.set_move_forward_position(0.4)
        self.controller.set_left_arm_gripper(1.0)  
    
    def _grasp_prop_via_act(self) -> None:
        try:
            print("ACT pick prop")
            command = f"cd /workspace/DISCOVERSE/policies/act && python3 policy_evaluate_ros.py -tn {self.tn} -ts {self.ts}"
            subprocess.run(command, shell=True, check=True)
        except Exception as e:
            pass

    def _hug_box(self):
        """
        抓取box的辅助函数,只需要走到对应的柜子前,会自动进行调整,对准box,然后抓取
        step 1: 识别目标盒子和物品
        step 2: 计算box位置偏移量
        step 3: 根据柜子类型调整目标位置
        step 4: 设定最终目标点并移动
        step 5: 抓取box
        step 6: 后退0.6离开柜子
        """
        print("抓取box")

        # _step_3
        print("步骤1: 识别目标盒子和物品")
        rgb_image = self.receiver.get_head_rgb()
        detection_results = yolo_process(rgb_image)
        box_position_wrt_camera = self.vision_module.get_Toc_box_from_yolo_result(
            detection_results, self.prop_name)

        # _step_4
        print("步骤2: 计算box位置偏移量")
        if box_position_wrt_camera is None:
            print("未检测到box")
            box_offset = 0.0
            return False

        box_offset = box_position_wrt_camera[0]
        print(
            f"box在相机坐标系下的位置: x={box_position_wrt_camera[0]:.3f}, y={box_position_wrt_camera[1]:.3f}, z={box_position_wrt_camera[2]:.3f}")
        print(f"左右偏移量: {box_offset:.3f}m")

        # _step_5
        print("步骤3: 根据柜子类型调整目标位置")
        base_position = MIDDLE_POINT_POSITION.copy()
        adjustment_scale = 1.0

        if self.cabinet_index == "left":
            base_position[0] += box_offset * adjustment_scale
            base_position[1] = CABINET_POSITION[self.cabinet_index][1]
            base_position.append(CABINET_POSITION[self.cabinet_index][-1])
            print(
                f"左柜子调整: x坐标从 {MIDDLE_POINT_POSITION[0]} 调整到 {base_position[0]:.3f}")
        else:
            base_position[1] -= box_offset * adjustment_scale
            base_position[0] = CABINET_POSITION[self.cabinet_index][0]
            base_position.append(CABINET_POSITION[self.cabinet_index][-1])
            print(
                f"右柜子调整: y坐标从 {MIDDLE_POINT_POSITION[1]} 调整到 {base_position[1]:.3f}")

        adjusted_position = base_position

        # _step_6 _step_7
        print("步骤4: 设定最终目标点并移动")
        hug_box_final_target_position = adjusted_position
        print(
            f"最终目标点: {np.round(np.array(hug_box_final_target_position).copy(), 2)}")
        self.move_to_point.move_to_point(hug_box_final_target_position)

        # _step_8
        print("步骤5: 抓取box")
        hug_target_box_pitch_angle = 0.45
        current_head_slide_position = self.receiver.get_joint_states()[
            "positions"][0]
        self.controller.reset_head()
        self.controller.set_head_slide_position(
            current_head_slide_position - 0.2)
        time.sleep(1)
        self.controller.set_head_pitch_angle(hug_target_box_pitch_angle)
        time.sleep(1)

        box_position_wrt_camera = self.vision_module.get_Toc_box_from_yolo(
            self.prop_name)
        box_position_wrt_base = self.pose_transform.transform_position_wrt_camera_to_base(
            box_position_wrt_camera)
        print(f"box_position_wrt_camera: {box_position_wrt_camera}")
        print(f"box_position_wrt_base: {(box_position_wrt_base)}")

        # 用于调整夹爪的位置,整体向右移动0.1(camera坐标系下)
        box_position_wrt_camera[1] += 0.1
        box_position_wrt_base = self.pose_transform.transform_position_wrt_camera_to_base(
            box_position_wrt_camera)

        # 第一维是前后,第二维是左右,第三维是上下,绝对位置
        left_arm_target_position = box_position_wrt_base + \
            np.array([0.0, 0.11, 0.05])
        right_arm_target_position = box_position_wrt_base + \
            np.array([0.0, -0.11, 0.05])
        left_arm_joint_pose = self.pose_transform.get_arm_joint_pose(
            target_position=left_arm_target_position, arm_action="pick", arm="l", action_rot=R.from_euler("zyx", [np.pi/2, -0.0551 + np.pi, np.pi / 8]).as_matrix())
        right_arm_joint_pose = self.pose_transform.get_arm_joint_pose(
            target_position=right_arm_target_position, arm_action="pick", arm="r", action_rot=R.from_euler("zyx", [-np.pi/2, -0.0551 + np.pi, -np.pi / 8]).as_matrix())

        # 释放arm的位姿,用于调整arm姿态
        self.left_arm_ready_joint_pose = left_arm_joint_pose
        self.right_arm_ready_joint_pose = right_arm_joint_pose

        # 打开夹爪增大接触面积提高稳定性
        self.controller.set_right_arm_gripper(1.0)
        self.controller.set_left_arm_gripper(1.0)
        time.sleep(1)

        # 后退0.05防止打到cab或者box
        self.controller.set_move_backward_position(0.05)
        self.controller.set_left_arm_position(left_arm_joint_pose)
        self.controller.set_right_arm_position(right_arm_joint_pose)
        time.sleep(1)
        print("第一次调整结束")

        # 前进0.2到目标抓取位置
        self.controller.set_move_forward_position(0.2)
        time.sleep(1)
        print("移动完成,准备抓取box")

        # 夹爪向中间夹紧,左边先移动,右边后移动
        left_arm_target_position = box_position_wrt_base + \
            np.array([0.0, 0.08, 0.05])
        right_arm_target_position = box_position_wrt_base + \
            np.array([0.0, -0.08, 0.05])
        left_arm_joint_pose = self.pose_transform.get_arm_joint_pose(
            target_position=left_arm_target_position, arm_action="pick", arm="l", action_rot=R.from_euler("zyx", [np.pi/2, -0.0551 + np.pi, np.pi / 8]).as_matrix())
        right_arm_joint_pose = self.pose_transform.get_arm_joint_pose(
            target_position=right_arm_target_position, arm_action="pick", arm="r", action_rot=R.from_euler("zyx", [-np.pi/2, -0.0551 + np.pi, -np.pi / 8]).as_matrix())
        time.sleep(1)

        self.controller.set_left_arm_position(left_arm_joint_pose)
        self.controller.set_right_arm_position(right_arm_joint_pose)
        time.sleep(1)
        print("第二次调整结束")

        # 后退0.4离开柜子
        self.controller.set_move_backward_position(0.4)
        time.sleep(1)
        self.controller.set_head_pitch_angle(0)
        time.sleep(1)
        self.controller.set_head_slide_position_smooth(0)
        time.sleep(1)
        
    def _navigation_to_drawer(self) -> str:
        self.move_to_point.move_to_point([0.5, 0.5, -65])
        right_result = self.vision_module.is_object_presence('drawer')
        print(f"right_result: {right_result}")

        self.controller.set_turn_left_angle(55)
        left_result = self.vision_module.is_object_presence('drawer')

        if left_result == None:
            self.controller.set_turn_right_angle(55)

        obj_pose = self.vision_module.get_Toc_from_yolo('drawer')
        obj_pose_base = self.pose_transform.transform_position_wrt_camera_to_base(obj_pose)
        obj_world_pose = self.pose_transform.transform_position_wrt_base_to_world(obj_pose_base)

        # print(right_result, left_result)
        if right_result == True and left_result == None:
            angle = -90
            time.sleep(1)
            self.move_to_point.move_to_point(
                [obj_world_pose[0] + 0.7, obj_world_pose[1], angle])
            return "right"
        if left_result == True and right_result == None:
            angle = 0
            self.move_to_point.move_to_point(
                [obj_world_pose[0] - 0.08, obj_world_pose[1] + 0.7, angle])
            return "left"

    def _drawer_open(self) -> None:
        direction = self._navigation_to_drawer()
        self.controller.set_head_slide_position(0.2)
        self.controller.set_head_pitch_angle(0.6)
        time.sleep(1)
        self.current_position['x'] = self.receiver.get_odom()['position']['x']
        self.current_position['y'] = self.receiver.get_odom()['position']['y']
        obj_pose = self.vision_module.get_Toc_from_yolo("Linear motion handles", drawer=True)
        obj_pose_base = self.pose_transform.transform_position_wrt_camera_to_base(obj_pose)
        print(f"obj_pose_base:{obj_pose_base}")

        # 伸到目标位置附近，准备抓取
        tmp_lft_arm_target_pose = obj_pose_base + np.array([-0.15, 0, 0.07])

        tmp_lft_arm_state = self.pose_transform.get_arm_joint_pose(tmp_lft_arm_target_pose, "pick", "l",
            action_rot = R.from_euler('zyx', [1.5807, 0, 0.6]).as_matrix())

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
        
        if self.vision_module.is_object_presence("sheet"):
            self.prop_name = "sheet"
        elif self.vision_module.is_object_presence("carton"):
            self.prop_name = "carton"
        elif self.vision_module.is_object_presence("disk"):
            self.prop_name = "disk"
        
    def _cabinet_door_open(self) -> None:
        direction = self._navigation_to_drawer()
        self.controller.set_head_slide_position(0.05)
        self.controller.set_head_pitch_angle(0.6)
        time.sleep(1)

        self.controller.set_move_backward_position(0.1)
        self.current_position['x'] = self.receiver.get_odom()['position']['x']
        self.current_position['y'] = self.receiver.get_odom()['position']['y']

        obj_pose = self.vision_module.get_Toc_from_yolo("rotary handles", drawer=True)
        obj_pose_base = self.pose_transform.transform_position_wrt_camera_to_base(obj_pose)

        tmp_lft_arm_target_pose = obj_pose_base + np.array([-0.15, 0, 0])

        tmp_lft_arm_state = self.pose_transform(tmp_lft_arm_target_pose, "pick", "l",
            action_rot = R.from_euler('zyx', [0, -1.5807, -1.5807]).as_matrix())

        self.controller.set_left_arm_position(tmp_lft_arm_state)
        time.sleep(1)
        self.controller.set_left_arm_gripper(1.0)
        self.controller.set_move_forward_position(0.13)
        self.controller.set_left_arm_gripper(0.0)
        time.sleep(1)
        self.controller.set_move_backward_position(0.2)
        print("拉开柜门")

        if direction == "left":
            self.move_to_point.move_to_point([self.current_position['x'] - 0.2, self.current_position['y'] - 0.1, 90])
        elif direction == "right":
            self.move_to_point.move_to_point([self.current_position['x'] - 0.1, self.current_position['y'] + 0.2, 0])
            
        if self.vision_module.is_object_presence("sheet"):
            self.prop_name = "sheet"
        elif self.vision_module.is_object_presence("carton"):
            self.prop_name = "carton"
        elif self.vision_module.is_object_presence("disk"):
            self.prop_name = "disk"

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
    round_3 = Round3(
        receiver, controller, task_parser, vision_module)
    stop_event = threading.Event()
    render_thread = threading.Thread(
        target=render_head_rgb,
        args=(receiver, stop_event)
    )
    render_thread.daemon = True
    render_thread.start()

    try:
        round_3.run()

    finally:
        stop_event.set()
        controller.stop_all()
        render_thread.join(timeout=1)
        if render_thread.is_alive():
            print("Warning: render thread did not exit cleanly")
        rclpy.shutdown()


if __name__ == "__main__":
    print("this is round 3")
    main()
