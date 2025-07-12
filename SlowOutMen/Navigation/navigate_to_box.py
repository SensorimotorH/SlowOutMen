from SlowOutMen.TaskUnderstanding.task_parser import TaskParser
from SlowOutMen.Navigation.utils.move_to_point_ros2 import MoveToPoint
from SlowOutMen.Communication.ros2_mmk2_api import MMK2_Receiver, MMK2_Controller
from VisionUnderstanding.VisiontoPoindcloud.detect_and_segment import yolo_process
from VisionUnderstanding.VisiontoPoindcloud.vision_module import RosVisionModule

import rclpy
import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from scipy.spatial.transform import Rotation as R

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
    "fourth": 0.3,
}
CABINET_POSITION = {
    "left": [0.5, 0.55, 180],
    "right": [0.55, 0.3, 90],
}
MIDDLE_POINT_POSITION = [0.4, 0.4]


class NavigationtoCabRound1:
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
        self._step_3()
        self._step_4()
        self._step_5()
        self._step_6()
        self._step_7()
        self._step_8()

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

        print(
            f"任务解析结果: 柜子位置={self.cabinet_index}, 层数={self.floor_index}, 物品={self.prop_name}")

    def _step_2(self) -> None:
        """
        步骤2: 走到中间点,调整高度到对应层数
        """
        print("步骤2: 走到中间点,调整高度")

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
            # 相机坐标系中x为左负右正，机器人坐标系中x正方向为右
            # 所以对于左柜子，可以直接加上偏移量
            base_position[0] += self.box_offset * adjustment_scale
            base_position[1] = CABINET_POSITION[self.cabinet_index][1]
            base_position.append(CABINET_POSITION[self.cabinet_index][-1])
            print(
                f"左柜子调整: x坐标从 {MIDDLE_POINT_POSITION[0]} 调整到 {base_position[0]:.3f}")
        else:
            # 右柜子调整y坐标
            # 相机坐标系的左右偏移需要转换到机器人坐标系的y轴
            # 右柜子面向y轴正方向，相机坐标系右偏移对应y轴负方向
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

        print("到达目标位置,准备进行抓取任务")

        # 可以在这里添加等待抓取任务的代码
        print("导航任务完成")

    def _step_8(self) -> None:
        """
        步骤8：抬高+低头
        """
        print("步骤8：调整头部位置")

        self.yaw_angle = 0.4
        current_slide_pos = self.receiver.get_joint_states()["positions"][0]
        self.controller.set_head_slide_position(current_slide_pos - 0.2)
        self.controller.set_head_pitch_angle(self.yaw_angle)


def main():
    rclpy.init()
    receiver = MMK2_Receiver()
    controller = MMK2_Controller(receiver)
    task_parser = TaskParser()
    vision_module = RosVisionModule(receiver)
    navigation = NavigationtoCabRound1(
        receiver, controller, task_parser, vision_module)

    try:
        navigation.run()

    finally:
        controller.stop_all()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
