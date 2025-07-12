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
    def __init__(self, action, receiver: MMK2_Receiver, controller: MMK2_Controller, task_parser: TaskParser, vision_module: RosVisionModule):
        self.task_parser = task_parser
        # 存储接收器和控制器的引用
        self.action = action
        self.receiver = receiver
        self.controller = controller
        self.move_to_point = MoveToPoint(receiver, controller)
        self.vision_module = vision_module
        self.posetran = PoseTransform(self.receiver)
        # 存储解析后的任务信息
        self.cabinet_index: str = ""
        self.floor_index: str = ""
        self.prop_name: str = ""

        # 存储目标位置信息
        self.target_position: List[float] = []
        self.current_position: Dict[float] = {
            'x': 0.0,
            'y': 0.0
        }
        self.box_center_x: Optional[float] = None

    def run(self):
        self._step_1()
        self._step_2()
        self._step_3()
        self._step_4()
        self._step_5()
        self._step_6()
        self._step_7()
        self._step_8()
        self._step_9_desk()
        # self._step_10()

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

        print(
            f"任务解析结果: 柜子位置={self.cabinet_index}, 层数={self.floor_index}, 物品={self.prop_name}, 目标桌子={self.table_index}")

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
        if self.action == "grasp":
            if self.floor_index == "second":
                box_pose_base[2] = 0.52
            elif self.floor_index == "third":
                box_pose_base[2] = 0.83
            elif self.floor_index == "fourth":
                box_pose_base[2] = 1.15  # 1.15可能有点高
            self.controller.set_left_arm_gripper(1.0)
            self.controller.set_right_arm_gripper(1.0)
            # 张开夹爪
            time.sleep(1)
            tmp_lft_arm_target_pose = box_pose_base + \
                np.array([-0.05, 0.1, 0.045])
            tmp_lft_arm_state = self.getArmEndTarget(tmp_lft_arm_target_pose, "carry", "l", np.array(
                self.receiver.get_joint_states()['positions'][3:9]), R.from_euler("zyx", [0., 0.8,  np.pi]).as_matrix())

            tmp_rgt_arm_target_pose = box_pose_base + \
                np.array([-0.05, -0.1, 0.045])
            tmp_rgt_arm_state = self.getArmEndTarget(tmp_rgt_arm_target_pose, "carry", "r", np.array(
                self.receiver.get_joint_states()['positions'][10:16]), R.from_euler("zyx", [0., 0.8, -np.pi]).as_matrix())

            self.controller.set_move_backward_position(0.1)
            print(tmp_lft_arm_state, tmp_rgt_arm_state)
            self.controller.set_right_arm_position(tmp_rgt_arm_state)
            self.controller.set_left_arm_position(tmp_lft_arm_state)
            time.sleep(1)
            self.controller.set_move_forward_position(0.15)

            time.sleep(1)

            self.controller.set_left_arm_gripper(0.05)
            self.controller.set_right_arm_gripper(0.05)
            print("已执行夹紧操作")
            time.sleep(1)

        elif self.action == "hug":
            box_pose_cam[1] = box_pose_cam[1] + 0.1
            box_pose_base = self.posetran.pos_cam_to_base(box_pose_cam)
            tmp_lft_arm_target_pose = box_pose_base + \
                np.array([0.0, 0.11, 0.05])  # y,x,z
            tmp_rgt_arm_target_pose = box_pose_base + \
                np.array([0.0, -0.11, 0.05])  # y,x,z
            tmp_lft_arm_state = self.getArmEndTarget(tmp_lft_arm_target_pose, "pick", "l", np.array(self.receiver.get_joint_states()[
                                                     'positions'][3:9]), R.from_euler("zyx", [np.pi/2, -0.0551 + np.pi, np.pi / 8]).as_matrix())
            tmp_rgt_arm_state = self.getArmEndTarget(tmp_rgt_arm_target_pose, "pick", "r", np.array(
                self.receiver.get_joint_states()['positions'][10:16]), R.from_euler("zyx", [-np.pi/2, -0.0551 + np.pi, -np.pi / 8]).as_matrix())
            self.controller.set_right_arm_gripper(1.0)
            self.controller.set_left_arm_gripper(1.0)
            time.sleep(1)

            self.controller.set_move_backward_position(0.05)
            self.controller.set_left_arm_position(tmp_lft_arm_state)
            self.controller.set_right_arm_position(tmp_rgt_arm_state)

            time.sleep(1)

            print("第一次调整结束")

            self.controller.set_move_forward_position(0.2)
            print("移动完成，准备抓取box")

            tmp_lft_arm_target_pose = box_pose_base + \
                np.array([0.0, 0.07, 0.05])  # y,x,z
            tmp_rgt_arm_target_pose = box_pose_base + \
                np.array([0.0, -0.07, 0.05])  # y,x,z
            tmp_lft_arm_state = self.getArmEndTarget(tmp_lft_arm_target_pose, "pick", "l", np.array(self.receiver.get_joint_states()[
                                                     'positions'][3:9]), R.from_euler("zyx", [np.pi/2, -0.0551 + np.pi, np.pi / 8]).as_matrix())
            tmp_rgt_arm_state = self.getArmEndTarget(tmp_rgt_arm_target_pose, "pick", "r", np.array(
                self.receiver.get_joint_states()['positions'][10:16]), R.from_euler("zyx", [-np.pi/2, -0.0551 + np.pi, -np.pi / 8]).as_matrix())
            time.sleep(1)
            self.controller.set_left_arm_position(tmp_lft_arm_state)
            self.controller.set_right_arm_position(tmp_rgt_arm_state)
            print("第二次调整结束")
            time.sleep(1)

            self.controller.set_move_backward_position(0.35)

    def _step_9_desk(self) -> None:
        """
        步骤9：读取桌子位置
        """
        current_slide_pose = self.receiver.get_joint_states()["positions"][0]
        self.controller.set_head_pitch_angle(0)

        for i in range(10):
            self.controller.set_head_slide_position(
                (10-i) * 0.1 * current_slide_pose)
            time.sleep(1)

        if self.cabinet_index == "left":

            if self.table_index == "left":
                self.controller.set_turn_left_angle(180)
                self.move_to_point.move_to_point([0.5, 0.23, 0])
            elif self.table_index == "right":
                self.controller.set_turn_left_angle(90)
                self.move_to_point.move_to_point([0.3, 0.09, -90])

        if self.cabinet_index == "right":

            if self.table_index == "left":
                self.controller.set_turn_right_angle(180)
                self.controller.set_move_forward_position(0.3)
                self.move_to_point.move_to_point([0.5, 0.23, 0])

            elif self.table_index == "right":
                self.controller.set_turn_right_angle(180)
                self.move_to_point.move_to_point([0.3, 0.09, -90])
        time.sleep(1)

        self.current_position['x'] = self.receiver.get_odom()['position']['x']
        self.current_position['y'] = self.receiver.get_odom()['position']['y']

        print(
            f"self.current_position:{self.current_position['x']}, {self.current_position['y']}")

        for i in range(100):
            self.controller.set_head_slide_position(0.004 * i)
            time.sleep(0.1)
        print("下降完成，准备抓取物品")

        self.controller.set_move_forward_position(0.15)
        print("移动完成，准备抓取物品")

        right_arm_pos = [-0.4641777170640647, -1.1499549351183942, 1.0497040081105575,
                         0.8421601585342549, 0.670234867120661, -0.7160617625400274]
        lft_arm_pos = [0.5395976853056692, -1.012183225006465, 1.066561862005807, -
                       0.6202181291863684, -0.6228169552416241, 0.44089390986698723]

        self.controller.set_arm_position(right_arm_pos, lft_arm_pos)

        self.controller.set_head_pitch_angle(0.55)

        time.sleep(1)

        print("移动完成，准备抓取物品")
        time.sleep(1)

    def _step_9_floor(self) -> None:

        current_slide_pose = self.receiver.get_joint_states()["positions"][0]

        if self.cabinet_index == "left":

            if self.table_index == "left":
                self.controller.set_move_forward_position(0.1)
                self.controller.set_turn_left_angle(180)
            elif self.table_index == "right":
                self.controller.set_turn_left_angle(90)

        if self.cabinet_index == "right":

            if self.table_index == "left":
                self.controller.set_turn_left_angle(90)
                self.controller.set_move_forward_position(0.2)
                self.controller.set_turn_right_angle(180)
            elif self.table_index == "right":
                self.controller.set_turn_right_angle(180)
        time.sleep(1)

        for i in range(10):
            if current_slide_pose+0.095 * i > 0.87:
                self.controller.set_head_slide_position(0.87)  # 放地上
                time.sleep(0.5)
                break
            else:
                self.controller.set_head_slide_position(
                    current_slide_pose+0.095 * i)
                time.sleep(0.1)
        print("下降完成，准备抓取prop")

        # right_arm_pos=[-0.10498875045841922, 0, 1.2, 1.6, 0.8, 1]
        # lft_arm_pos= [0.103648310695625948, 0,1.2, -1.6, -0.8, -1]

        # self.controller.set_arm_pos(right_arm_pos, lft_arm_pos)
        # time.sleep(1)

        box_pose_cam = self.vision_module.get_Toc_box_from_yolo(self.prop_name)
        box_pose_cam[1] = box_pose_cam[1] + 0.1
        box_pose_base = self.posetran.pos_cam_to_base(box_pose_cam)

        tmp_lft_arm_target_pose = box_pose_base + \
            np.array([-0.05, 0.09, -0.2])  # y,x,z
        tmp_rgt_arm_target_pose = box_pose_base + \
            np.array([-0.05, -0.09, -0.2])  # y,x,z
        tmp_lft_arm_state = self.getArmEndTarget(tmp_lft_arm_target_pose, "pick", "l", np.array(
            self.receiver.get_joint_states()['positions'][3:9]), R.from_euler("zyx", [np.pi/2, -0.0551 + np.pi, 0]).as_matrix())
        tmp_rgt_arm_state = self.getArmEndTarget(tmp_rgt_arm_target_pose, "pick", "r", np.array(
            self.receiver.get_joint_states()['positions'][10:16]), R.from_euler("zyx", [-np.pi/2, -0.0551 + np.pi, 0]).as_matrix())
        self.controller.set_arm_position(tmp_rgt_arm_state, tmp_lft_arm_state)

        self.controller.set_head_pitch_angle(0.55)

        print("移动完成，准备抓取prop")
        time.sleep(1)

        # self.controller.set_move_forward_pos(0.1)

    def _step_10(self) -> None:
        prop_pose_cam = self.vision_module.get_Toc_prop_from_yolo(
            self.prop_name)  # 获取相坐标
        prop_pose_base = self.posetran.pos_cam_to_base(prop_pose_cam)
        print(f"prop_pose_base:{prop_pose_base}")

        if prop_pose_base[1] > 0:
            # 设置左臂位置到圆盘上方 | Set left arm position above the disk
            tmp_lft_arm_target_pose = prop_pose_base + \
                np.array([0., 0.05, 0.1])
        # 设置左臂末端姿态，使用单位矩阵表示垂直向下抓取 | Set left arm end-effector pose, using identity matrix for vertical downward grasping
            tmp_lft_arm_state = self.getArmEndTarget(tmp_lft_arm_target_pose, "pick", "l", np.array(
                self.receiver.get_joint_states()['positions'][3:9]), np.eye(3))
            self.controller.set_left_arm_gripper = 0.5  # 半开左爪 | Half-open left gripper
            self.controller.set_left_arm_position(tmp_lft_arm_state)
            time.sleep(1)
        else:
            # 设置右臂位置到圆盘上方 | Set right arm position above the disk
            tmp_rgt_arm_target_pose = prop_pose_base + np.array([0., 0., 0.1])
        # 设置右臂末端姿态，使用单位矩阵表示垂直向下抓取 | Set right arm end-effector pose, using identity matrix for vertical downward grasping
            tmp_rgt_arm_state = self.getArmEndTarget(tmp_rgt_arm_target_pose, "pick", "r", np.array(
                self.receiver.get_joint_states()['positions'][10:16]), np.eye(3))
            self.controller.set_right_arm_gripper = 0.5  # 半开右爪 | Half-open right gripper
            self.controller.set_right_arm_position(tmp_rgt_arm_state)
            time.sleep(1)

    def _navigation_to_obj(self, obj_name) -> None:

        self.move_to_point.move_to_point([0.5, 0.5, -65])
        right_result = self.vision_module.is_object_presence(obj_name)
        print(f"left_result: {right_result}")

        self.controller.set_turn_left_angle(55)
        left_result = self.vision_module.is_object_presence(obj_name)

        if left_result == False:
            self.controller.set_turn_right_angle(55)

        obj_pose = self.vision_module.follow_objname(obj_name)
        obj_pose_base = self.posetran.pos_cam_to_base(obj_pose)
        obj_world_pose = self.posetran.pos_base_to_world(obj_pose_base)

        if right_result == True and left_result == False:
            angle = -90
            time.sleep(1)
            self.move_to_point.move_to_point(
                [obj_world_pose[0]+0.5, obj_world_pose[1]-0.1, angle])
        if left_result == True and right_result == False:
            angle = 0
            self.move_to_point.move_to_point(
                [obj_world_pose[0]-0.1, obj_world_pose[1]+0.5, angle])

        try:
            print("ACT pick disk")
            command = f"cd /workspace/DISCOVERSE/policies/act && python3 policy_evaluate_ros2_som.py -tn mmk2_drawer_open -ts 20250407-140604"
            subprocess.run(command, shell=True, check=True)
        except Exception as e:
            pass

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

    def reset_arm(self):
        self.controller.set_arm_position(
            rgt_positions=[0., 0., 0., 0., 0., 0.],
            lft_positions=[0., 0., 0., 0., 0., 0.]
        )
        time.sleep(2)


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
    action = "hug"
    navigation = NavigationtoCabRound1(action,
                                       receiver, controller, task_parser, vision_module)
    stop_event = threading.Event()
    render_thread = threading.Thread(
        target=render_head_rgb,
        args=(receiver, stop_event)
    )
    render_thread.daemon = True
    render_thread.start()

    try:
        navigation._navigation_to_obj("drawer")

    finally:
        # 按照正确的顺序关闭
        stop_event.set()
        controller.stop_all()
        # render_thread.join(timeout=1)
        # 确保在最后关闭 ROS2
        rclpy.shutdown()


if __name__ == "__main__":
    main()
