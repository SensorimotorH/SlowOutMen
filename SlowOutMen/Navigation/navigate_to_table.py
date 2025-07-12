from SlowOutMen.TaskUnderstanding.task_parser import TaskParser
from SlowOutMen.Navigation.utils.move_to_point_ros2 import MoveToPoint
from SlowOutMen.Communication.ros2_mmk2_api import MMK2_Receiver, MMK2_Controller
from VisionUnderstanding.VisiontoPoindcloud.detect_and_segment import yolo_process
from VisionUnderstanding.VisiontoPoindcloud.vision_module import RosVisionModule
from Manipulation.RobotArmMove.GetArmTarget import ArmPlanner
from Manipulation.utils.pose_transform_base import PoseTransform
from Manipulation.utils.mmk2.mmk2_fik import MMK2FIK
import rclpy
import cv2
import time
import threading
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from scipy.spatial.transform import Rotation as R

"""
比赛round_1的导航任务,目标是移动到目标box的前面固定位置,然后等待抓取任务
1. 解析任务 -> Target: tablet_index
2. move_to_point -> 移动到目标点
"""

MIDDLE_POINT_POSITION = [0.4, 0.4]


class NavigationtoTableRound1:
    def __init__(self, receiver: MMK2_Receiver, controller: MMK2_Controller, task_parser: TaskParser, vision_module: RosVisionModule):
        self.task_parser = task_parser
        self.move_to_point = MoveToPoint(receiver, controller)
        self.vision_module = vision_module
        # 存储解析后的任务信息
        self.cabinet_index: str = ""
        self.floor_index: str = ""
        self.prop_name: str = ""
        # 存储目标位置信息
        self.target_position: List[float] = []
        self.box_center_x: Optional[float] = None
        # 存储接收器和控制器的引用
        self.receiver = receiver
        self.controller = controller

    def run(self):
        self._step_1()
        self._step_2()

    def _step_1(self) -> None:
        """
        步骤1: 解析任务,从任务描述中提取cabinet_index, floor_index和prop_name
        """
        print("步骤1: 解析任务")

        # 获取任务指令
        instruction = self.receiver.get_taskinfo()

        # 解析任务
        task_info = self.task_parser.parse_task(instruction)
        self.table = task_info['target_table']

        print(
            f"任务解析结果: 桌子位置：{self.table}")

    def _step_2(self) -> None:

        time.sleep(1.0)


def main():
    rclpy.init()
    receiver = MMK2_Receiver()
    controller = MMK2_Controller(receiver)
    task_parser = TaskParser()
    vision_module = RosVisionModule(receiver)
    navigation = NavigationtoTableRound1(
        receiver, controller, task_parser, vision_module)

    try:
        navigation.run()

    finally:
        controller.stop_all()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
