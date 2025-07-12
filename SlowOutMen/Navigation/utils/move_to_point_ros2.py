import numpy as np
from SlowOutMen.Communication.ros2_mmk2_api import MMK2_Receiver, MMK2_Controller
import rclpy
from scipy.spatial.transform import Rotation
import math


POSITIVE_DERECTION = [0.707, 0,  0, -0.707]
INPUT_KEY = ["x", "y", "theta"]


class MoveToPoint:
    def __init__(self, receiver: MMK2_Receiver, controller: MMK2_Controller):
        self.receiver = receiver
        self.controller = controller
        self.x_1 = None
        self.y_1 = None
        self.rot = None
        self.theta_1 = None

    def _move(self):
        if math.sqrt((self.x_2 - self.x_1) ** 2 + (self.y_2 - self.y_1) ** 2) < 0.02:
            # 当位置已经足够接近时，只需要调整朝向
            rotate = self.theta_2 - self.theta_1
            rotate = (rotate + 180) % 360 - 180  # 归一化到 (-180, 180)
            print(f"旋转 {rotate:.2f} 度，使机器人面向目标角度")
            self.controller.set_turn_left_angle(rotate)
            return
        # 计算目标方向角度（相对于y轴负方向）
        target_angle = math.degrees(math.atan2(
            self.x_2 - self.x_1, -(self.y_2 - self.y_1))) % 360

        # 计算旋转角度，使机器人面向目标方向
        rotate1 = target_angle - self.theta_1
        rotate1 = (rotate1 + 180) % 360 - 180  # 归一化到 (-180, 180)
        print(f"旋转 {rotate1:.2f} 度，使机器人面向目标位置")
        self.controller.set_turn_left_angle(rotate1)

        # 计算直线前进距离
        distance = math.sqrt((self.x_2 - self.x_1) **
                             2 + (self.y_2 - self.y_1) ** 2)
        print(f"前进 {distance:.2f} 个单位，到达目标位置")
        self.controller.set_move_forward_position(distance)

        # 计算最终旋转角度
        rotate2 = self.theta_2 - target_angle
        rotate2 = (rotate2 + 180) % 360 - 180  # 归一化到 (-180, 180)
        print(f"旋转 {rotate2:.2f} 度，使机器人朝向目标角度 {self.theta_2:.2f} 度")
        self.controller.set_turn_left_angle(rotate2)

    def _get_position(self):
        self.x_1 = self.receiver.get_odom()['position']['x']
        self.y_1 = self.receiver.get_odom()['position']['y']
        self.rot = self.receiver.get_odom()["orientation"]
        # 官方的正方向是面朝table_1(POSITIVE_DERECTION)再逆时针旋转90度, 所以这里要加90度
        # MuJoCo的x正方向是朝cabinet_2, y正方向是朝cabinet_1
        self.theta_1 = 90.0 + np.rad2deg(Rotation.from_quat(
            [self.rot['x'], self.rot["y"], self.rot["z"], self.rot["w"]]).as_euler('zyx')[0])
        self.theta_1 = (self.theta_1 + 180) % 360 - 180
        print(
            f"\nnow: x:{self.x_1:.2f} y:{self.y_1:.2f} theta:{self.theta_1:.2f}\n")

    def move_to_point(self, target: dict | list):
        if isinstance(target, list):
            assert len(target) == 3, "target should be a list with 3 elements"
            target = {key: value for key, value in zip(INPUT_KEY, target)}
        elif isinstance(target, dict):
            assert all([key in target for key in INPUT_KEY]
                       ), f"target should contain all keys in {INPUT_KEY}"
        self.target = target
        self.x_2 = self.target["x"]
        self.y_2 = self.target["y"]
        self.theta_2 = self.target["theta"]
        self._get_position()
        self._move()
        self._get_position()


def main():
    rclpy.init()
    receiver = MMK2_Receiver()
    controller = MMK2_Controller(receiver)

    target = {
        "x": 0.40,
        "y": 0.40,
        "theta": 90
    }

    move_to_point = MoveToPoint(receiver, controller)
    move_to_point.move_to_point(target)

    controller.set_head_slide_position(1.22 - 1.08 * 0.742)
    controller.set_turn_left_angle(90)

    controller.stop_all()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
