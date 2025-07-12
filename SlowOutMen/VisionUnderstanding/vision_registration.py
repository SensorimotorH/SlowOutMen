from SlowOutMen.VisionUnderstanding.PoseEstimation.pose_estimate import PoseEstimator
from VisionUnderstanding.VisiontoPoindcloud.vision_module import get_pointcloud
from SlowOutMen.Communication.ros2_mmk2_api import MMK2_Controller, MMK2_Receiver

import rclpy
import time


COMPLETE_PCL_PATH = "/workspace/SlowOutMen/VisionUnderstanding/PoseEstimation/obj_ply"
TEMP_PATH = "/workspace/SlowOutMen/VisionUnderstanding/temp"


def vision_registration(receiver: MMK2_Receiver, obj_name: str, debug=False):
    partial_point_cloud = get_pointcloud(receiver, obj_name)
    pose_estimator = PoseEstimator(partial_point_cloud, obj_name, debug)
    pose_result = pose_estimator()

    return pose_result


if __name__ == "__main__":
    rclpy.init()
    receiver = MMK2_Receiver()
    controller = MMK2_Controller(receiver)
    obj_name = "book"
    controller.set_head_slide_position(0.4)
    time.sleep(3)

    pose_result = vision_registration(receiver, obj_name, debug=False)
