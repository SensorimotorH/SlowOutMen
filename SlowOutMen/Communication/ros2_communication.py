import argparse
import os
import cv2
import json
from datetime import datetime
import time
import numpy as np
import mediapy as media
from typing import List

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, JointState
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float64MultiArray
from geometry_msgs.msg import Twist

AVAILABLE_SUBSCIPTION_TOPICS = {
    'clock': {
        'topic': '/clock',
        'type': Clock,
        'callback': 'clock_callback'
    },
    'head_rgb': {
        'topic': '/mmk2/head_camera/color/image_raw',
        'type': Image,
        'callback': 'head_rgb_callback'
    },
    'head_depth': {
        'topic': '/mmk2/head_camera/aligned_depth_to_color/image_raw',
        'type': Image,
        'callback': 'head_depth_callback'
    },
    'head_camera_info': {
        'topic': '/mmk2/head_camera/color/camera_info',
        'type': CameraInfo,
        'callback': 'head_camera_info_callback'
    },
    'head_depth_camera_info': {
        'topic': '/mmk2/head_camera/aligned_depth_to_color/camera_info',
        'type': CameraInfo,
        'callback': 'head_depth_camera_info_callback'
    },
    'left_rgb': {
        'topic': '/mmk2/left_camera/color/image_raw',
        'type': Image,
        'callback': 'left_rgb_callback'
    },
    'left_camera_info': {
        'topic': '/mmk2/left_camera/color/camera_info',
        'type': CameraInfo,
        'callback': 'left_camera_info_callback'
    },
    'right_rgb': {
        'topic': '/mmk2/right_camera/color/image_raw',
        'type': Image,
        'callback': 'right_rgb_callback'
    },
    'right_camera_info': {
        'topic': '/mmk2/right_camera/color/camera_info',
        'type': CameraInfo,
        'callback': 'right_camera_info_callback'
    },
    'odom': {
        'topic': '/mmk2/odom',
        'type': Odometry,
        'callback': 'odom_callback'
    },
    'joint_states': {
        'topic': '/mmk2/joint_states',
        'type': JointState,
        'callback': 'joint_states_callback'
    },
    'taskinfo': {
        'topic': '/s2r2025/taskinfo',
        'type': String,
        'callback': 'taskinfo_callback'
    },
    'gameinfo': {
        'topic': '/s2r2025/gameinfo',
        'type': String,
        'callback': 'gameinfo_callback'
    }
}

AVAILABLE_PUBLICATION_TOPICS = {
    'cmd_vel': {
        'topic': '/mmk2/cmd_vel',
        'type': Twist
    },
    'head': {
        'topic': '/mmk2/head_forward_position_controller/commands',
        'type': Float64MultiArray
    },
    'left_arm': {
        'topic': '/mmk2/left_arm_forward_position_controller/commands',
        'type': Float64MultiArray
    },
    'right_arm': {
        'topic': '/mmk2/right_arm_forward_position_controller/commands',
        'type': Float64MultiArray
    },
    'spine': {
        'topic': '/mmk2/spine_forward_position_controller/commands',
        'type': Float64MultiArray
    }
}


class TopicSubscriber(Node):
    def __init__(self, save_dir: str = None, topics: List = None, debug: bool = False):
        super().__init__('mmk2_subscriber')
        # 1. 初始化基础变量
        self.sub_imgs = {}
        self.sub_files = {}
        self.bridge = CvBridge()
        self.ros_info = {
            key: None for key in AVAILABLE_SUBSCIPTION_TOPICS.keys()}
        self.ros_info['joint_states'] = {
            'timestamp': 0,
            'positions': np.zeros(17),
            'velocities': np.zeros(17),
            'efforts': np.zeros(17)
        }
        self.debug = debug
        self.save_dir = save_dir

        # 2. 设置话题列表
        if topics is None:
            topics = list(AVAILABLE_SUBSCIPTION_TOPICS.keys())

        # 3. 创建订阅者
        for topic_name in topics:
            if topic_name in AVAILABLE_SUBSCIPTION_TOPICS:
                topic_info = AVAILABLE_SUBSCIPTION_TOPICS[topic_name]
                self.create_subscription(
                    topic_info['type'],
                    topic_info['topic'],
                    getattr(self, topic_info['callback']),
                    10
                )

        # 4. 设置保存目录
        if self.debug:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.head_rgb_list = []
            self.head_depth_list = []
            self.right_rgb_list = []
            self.left_rgb_list = []
            if save_dir is None:
                self.save_dir = f"/workspace/SlowOutMen/DebugOutput/recorded_data_{timestamp}"
                os.makedirs(self.save_dir, exist_ok=True)

                # 5. 初始化文件和目录
                for topic_name in topics:
                    if topic_name not in AVAILABLE_SUBSCIPTION_TOPICS:
                        self.get_logger().warn(f'Unknown topic: {topic_name}')
                        continue

                    if topic_name.endswith('_rgb') or topic_name.endswith('_depth'):
                        self.sub_imgs[topic_name] = os.path.join(
                            self.save_dir, topic_name)
                        os.makedirs(self.sub_imgs[topic_name], exist_ok=True)
                    else:
                        ext = '.json' if topic_name.endswith(
                            '_camera_info') else '.txt'
                        self.sub_files[topic_name] = open(os.path.join(
                            self.save_dir, f'{topic_name}{ext}'), 'w')

        # 6. 打印日志
        if self.save_dir is not None:
            self.get_logger().info(
                f'Topic subscriber initialized. Saving data to: {self.save_dir}')
        self.get_logger().info(f'Subscribed topics: {topics}')

        # 7. 调用spin_once多次以接收初始消息
        self.get_logger().info("等待接收初始消息...")
        start_time = time.time()
        timeout = 1.0

        # 创建一个函数来检查topic是否已接收数据
        def _has_received_data(topic):
            if topic == 'joint_states':
                return self.ros_info[topic]['timestamp'] != 0
            else:
                return self.ros_info[topic] is not None

        while time.time() - start_time < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)

            # 检查所有关注的主题是否都已收到数据
            received_topics = [
                topic for topic in topics if _has_received_data(topic)]
            self.get_logger().debug(f"已接收数据的主题: {received_topics}")

            # 当所有主题都收到数据或超时时退出
            if len(received_topics) == len(topics) or time.time() - start_time >= timeout:
                break

        # 报告接收情况
        received_topics = [
            topic for topic in topics if _has_received_data(topic)]
        missing_topics = [
            topic for topic in topics if topic not in received_topics]
        self.get_logger().info(
            f"初始化完成,已接收 {len(received_topics)}/{len(topics)} 个主题数据")
        if missing_topics:
            self.get_logger().warn(f"未接收到数据的主题: {missing_topics}")

    def __del__(self):
        for f in self.sub_files.values():
            f.close()

    # 新增回调函数注释：处理 Clock 消息,记录时间戳
    def clock_callback(self, msg: Clock) -> None:
        # 回调：处理 clock 消息,记录时间戳
        timestamp = f"{msg.clock.sec}.{msg.clock.nanosec:09d}"
        if self.debug:
            self.sub_files['clock'].write(f"{timestamp}\n")
            self.sub_files['clock'].flush()
        self.ros_info['clock'] = timestamp

    # 回调：处理 head_camera_info 消息,保存相机信息
    def head_camera_info_callback(self, msg: CameraInfo) -> None:
        if self.debug:
            json.dump({"timestamp": self.get_clock().now().to_msg(), "data": msg},
                      self.sub_files['head_camera_info'], default=str)
            self.sub_files['head_camera_info'].write('\n')
            self.sub_files['head_camera_info'].flush()
        self.ros_info['head_camera_info'] = msg

    # 回调：处理 head_depth_camera_info 消息,保存相机信息
    def head_depth_camera_info_callback(self, msg: CameraInfo) -> None:
        if self.debug:
            json.dump({"timestamp": self.get_clock().now().to_msg(), "data": msg},
                      self.sub_files['head_depth_camera_info'], default=str)
            self.sub_files['head_depth_camera_info'].write('\n')
            self.sub_files['head_depth_camera_info'].flush()
        self.ros_info['head_depth_camera_info'] = msg

    # 回调：处理 head_rgb 图像消息,保存图片文件
    def head_rgb_callback(self, msg: Image) -> None:
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        cv_image = cv2.cvtColor(self.bridge.imgmsg_to_cv2(
            msg, "rgb8"), cv2.COLOR_RGB2BGR)
        if self.debug:
            head_rgb_path = os.path.join(
                self.sub_imgs['head_rgb'], f"{timestamp:.6f}.png")
            cv2.imwrite(head_rgb_path, cv_image)
            self.head_rgb_list.append(cv_image)
        self.ros_info['head_rgb'] = cv_image

    # 回调：处理 head_depth 图像消息,保存图片文件
    def head_depth_callback(self, msg: Image) -> None:
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        if self.debug:
            head_depth_path = os.path.join(
                self.sub_imgs['head_depth'], f"{timestamp:.6f}.png")
            cv2.imwrite(head_depth_path, depth_image)
            self.head_depth_list.append(depth_image)
        self.ros_info['head_depth'] = depth_image

    # 回调：处理 left_camera_info 消息,保存相机信息
    def left_camera_info_callback(self, msg: CameraInfo) -> None:
        if self.debug:
            json.dump({"timestamp": self.get_clock().now().to_msg(), "data": msg},
                      self.sub_files['left_camera_info'], default=str)
            self.sub_files['left_camera_info'].write('\n')
            self.sub_files['left_camera_info'].flush()
        self.ros_info['left_camera_info'] = msg

    # 回调：处理 left_rgb 图像消息,保存图片文件
    def left_rgb_callback(self, msg: Image) -> None:
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        cv_image = cv2.cvtColor(self.bridge.imgmsg_to_cv2(
            msg, "rgb8"), cv2.COLOR_RGB2BGR)
        if self.debug:
            filename = f"{timestamp:.6f}.png"
            cv2.imwrite(os.path.join(
                self.sub_imgs['left_rgb'], filename), cv_image)
            self.left_rgb_list.append(cv_image)
        self.ros_info['left_rgb'] = cv_image

    # 回调：处理 right_camera_info 消息,保存相机信息
    def right_camera_info_callback(self, msg: CameraInfo) -> None:
        if self.debug:
            json.dump({"timestamp": self.get_clock().now().to_msg(), "data": msg},
                      self.sub_files['right_camera_info'], default=str)
            self.sub_files['right_camera_info'].write('\n')
            self.sub_files['right_camera_info'].flush()
        self.ros_info['right_camera_info'] = msg

    # 回调：处理 right_rgb 图像消息,保存图片文件
    def right_rgb_callback(self, msg: Image) -> None:
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        cv_image = cv2.cvtColor(self.bridge.imgmsg_to_cv2(
            msg, "rgb8"), cv2.COLOR_RGB2BGR)
        if self.debug:
            filename = f"{timestamp:.6f}.png"
            cv2.imwrite(os.path.join(
                self.sub_imgs['right_rgb'], filename), cv_image)
            self.right_rgb_list.append(cv_image)
        self.ros_info['right_rgb'] = cv_image

    # 回调：处理 odom 消息,记录位姿和速度数据
    def odom_callback(self, msg: Odometry) -> None:
        timestamp = self.get_clock().now().to_msg()
        positions = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        twist = msg.twist.twist

        if self.debug:
            self.sub_files['odom'].write(f"{timestamp},{positions.x},{positions.y},{positions.z}," +
                                         f"{orientation.w},{orientation.x},{orientation.y},{orientation.z}," +
                                         f"{twist.linear.x},{twist.linear.y},{twist.linear.z}," +
                                         f"{twist.angular.x},{twist.angular.y},{twist.angular.z}\n")
            self.sub_files['odom'].flush()

        self.ros_info['odom'] = {
            'timestamp': timestamp,
            'position': {'x': positions.x, 'y': positions.y, 'z': positions.z},
            'orientation': {'w': orientation.w, 'x': orientation.x, 'y': orientation.y, 'z': orientation.z},
            'twist': {'linear': {'x': twist.linear.x, 'y': twist.linear.y, 'z': twist.linear.z},
                      'angular': {'x': twist.angular.x, 'y': twist.angular.y, 'z': twist.angular.z}}
        }

    # 回调：处理 joint_states 消息,记录关节状态
    def joint_states_callback(self, msg: JointState) -> None:
        timestamp = self.get_clock().now().to_msg()
        positions = ','.join([f"{p}" for p in msg.position])
        velocities = ','.join([f"{v}" for v in msg.velocity])
        efforts = ','.join([f"{e}" for e in msg.effort])
        if self.debug:
            self.sub_files['joint_states'].write(
                f"{timestamp},{positions},{velocities},{efforts}\n")
            self.sub_files['joint_states'].flush()

        positions = [float(p) for p in msg.position]
        velocities = [float(v) for v in msg.velocity]
        efforts = [float(e) for e in msg.effort]
        self.ros_info['joint_states'] = {
            'timestamp': timestamp,
            'positions': positions,
            'velocities': velocities,
            'efforts': efforts
        }

    # 回调：处理 taskinfo 消息,记录任务信息
    def taskinfo_callback(self, msg: String) -> None:
        timestamp = self.get_clock().now().to_msg()
        if self.debug:
            if msg.data != self.ros_info['taskinfo']:
                self.sub_files['taskinfo'].write(f"{timestamp},{msg.data}\n")
                self.sub_files['taskinfo'].flush()

        self.ros_info['taskinfo'] = msg.data

    # 回调：处理 gameinfo 消息,记录游戏信息
    def gameinfo_callback(self, msg: String) -> None:
        timestamp = self.get_clock().now().to_msg()
        if self.debug:
            self.sub_files['gameinfo'].write(f"{timestamp},{msg.data}\n")
            self.sub_files['gameinfo'].flush()
        self.ros_info['gameinfo'] = msg.data

    # 保存数据到本地
    def record_data(self):
        if self.debug:
            os.makedirs(os.path.join(self.save_dir, "video"), exist_ok=True)
            media.write_video(
                os.path.join(self.save_dir, "video/head_rgb.mp4"),
                self.head_rgb_list, fps=24)
            media.write_video(
                os.path.join(self.save_dir, "video/head_depth.mp4"),
                self.head_depth_list, fps=24)
            media.write_video(
                os.path.join(self.save_dir, "video/left_rgb.mp4"),
                self.left_rgb_list, fps=24)
            media.write_video(
                os.path.join(self.save_dir, "video/right_rgb.mp4"),
                self.right_rgb_list, fps=24)


class TopicPublisher(Node):
    def __init__(self, save_dir=None, topics=None, debug=False):
        super().__init__('mmk2_publisher')

        # 1. 初始化基础变量
        self.pubs = {}
        self.pub_files = {}
        self.debug = debug

        # 2. 设置话题列表
        if topics is None:
            topics = list(AVAILABLE_PUBLICATION_TOPICS.keys())

        # 3. 创建发布者
        for topic_name, topic_info in AVAILABLE_PUBLICATION_TOPICS.items():
            self.pubs[topic_name] = self.create_publisher(
                topic_info['type'],
                topic_info['topic'],
                10
            )

        # 4. 设置保存目录
        if save_dir is None and self.debug:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            current_file_dir = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))
            folder_path = os.path.join(
                current_file_dir, f'ros2_data/control_data/recorded_data_{timestamp}')
            self.save_dir = folder_path
        else:
            self.save_dir = save_dir

        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

            # 5. 初始化文件
            for topic_name in AVAILABLE_PUBLICATION_TOPICS:
                self.pub_files[topic_name] = open(os.path.join(
                    self.save_dir, f'{topic_name}.txt'), 'w')

        # 6. 打印日志
        if self.save_dir is not None:
            self.get_logger().info(
                f'Topic publisher initialized. Saving data to: {self.save_dir}')
        self.get_logger().info(
            f'Available publication topics: {list(AVAILABLE_PUBLICATION_TOPICS.keys())}')

    # 发布底盘移动命令,并记录发布的速度
    def publish_cmd_vel(self, linear: float, angular: float) -> None:
        """
        发布底盘移动命令
        :param linear: 前后速度（前为正）
        :param angular: 旋转速度（逆时针为正）
        """
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self.pubs['cmd_vel'].publish(msg)

        if self.debug:
            self.pub_files['cmd_vel'].write(
                f"cmd_vel: linear_x={linear}, angular_z={angular}\n")
            self.pub_files['cmd_vel'].flush()

    # 发布头部位置命令,并记录发布的关节位置
    def publish_head_position(self, positions: List[float]) -> None:
        """
        发布头部位置命令
        :param positions: 头部关节位置列表
        """
        msg = Float64MultiArray()
        msg.data = positions
        self.pubs['head'].publish(msg)
        if self.debug:
            self.pub_files['head'].write(f"head: {positions}\n")
            self.pub_files['head'].flush()

    # 发布左臂位置命令,并记录发布内容
    def publish_left_arm_position(self, positions: List[float]) -> None:
        """
        发布左臂位置命令
        :param positions: 左臂关节位置列表
        """
        msg = Float64MultiArray()
        msg.data = positions
        self.pubs['left_arm'].publish(msg)
        if self.debug:
            self.pub_files['left_arm'].write(f"left_arm: {positions}\n")
            self.pub_files['left_arm'].flush()

    # 发布右臂位置命令,并记录发布内容
    def publish_right_arm_position(self, positions: List[float]) -> None:
        """
        发布右臂位置命令
        :param positions: 右臂关节位置列表
        """
        msg = Float64MultiArray()
        msg.data = positions
        self.pubs['right_arm'].publish(msg)
        if self.debug:
            self.pub_files['right_arm'].write(f"right_arm: {positions}\n")
            self.pub_files['right_arm'].flush()

    # 发布升降位置命令,并记录发布内容
    def publish_spine_position(self, positions: List[float]) -> None:
        """
        发布升降位置命令
        :param positions: 升降关节位置值
        """
        msg = Float64MultiArray()
        msg.data = positions
        self.pubs['spine'].publish(msg)
        if self.debug:
            self.pub_files['spine'].write(f"spine: {positions}\n")
            self.pub_files['spine'].flush()


def main_pub(args=None):
    parser = argparse.ArgumentParser(
        description='ROS2 topic subscriber with configurable save path and topics')
    parser.add_argument('--save-dir',
                        type=str,
                        help='Directory to save the recorded data',
                        default=None)
    parser.add_argument('--debug',
                        type=bool,
                        help='Enable debug mode (default: False)',
                        default=False)
    parser.add_argument('--topics',
                        type=str,
                        nargs='*',
                        help='List of topics to subscribe (default: all topics)',
                        default=None,
                        choices=AVAILABLE_SUBSCIPTION_TOPICS.keys())

    args = parser.parse_args()

    try:
        rclpy.init()
        publisher = TopicPublisher(
            save_dir=args.save_dir, topics=args.topics, debug=args.debug)

        try:
            publisher.publish_cmd_vel(0.2, 0.0)
            time.sleep(2)
            publisher.publish_cmd_vel(0.0, 0.0)
            rclpy.spin(publisher)

        except KeyboardInterrupt:
            print("\n程序被用户中断")
        finally:
            # 清理资源
            for file in publisher.pub_files.values():
                file.close()
            publisher.destroy_node()
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


def main_sub(args=None):
    parser = argparse.ArgumentParser(
        description='ROS2 topic subscriber with configurable save path and topics')
    parser.add_argument('--save-dir',
                        type=str,
                        help='Directory to save the recorded data',
                        default=None)
    parser.add_argument('--debug',
                        type=bool,
                        help='Enable debug mode (default: False)',
                        default=False)
    parser.add_argument('--topics',
                        type=str,
                        nargs='*',
                        help='List of topics to subscribe (default: all topics)',
                        default=None,
                        choices=AVAILABLE_SUBSCIPTION_TOPICS.keys())

    args = parser.parse_args()

    try:
        rclpy.init()
        subscriber = TopicSubscriber(
            save_dir=args.save_dir, topics=args.topics, debug=args.debug)

        try:
            while rclpy.ok():
                rclpy.spin_once(subscriber)

        except KeyboardInterrupt:
            media.write_video(
                os.path.join(subscriber.save_dir, "video/head_rgb.mp4"),
                subscriber.head_rgb_list, fps=24)
            media.write_video(
                os.path.join(subscriber.save_dir, "video/head_depth.mp4"),
                subscriber.head_depth_list, fps=24)
            media.write_video(
                os.path.join(subscriber.save_dir, "video/left_rgb.mp4"),
                subscriber.left_rgb_list, fps=24)
            media.write_video(
                os.path.join(subscriber.save_dir, "video/right_rgb.mp4"),
                subscriber.right_rgb_list, fps=24)
            print("\n程序被用户中断")
        finally:
            # 清理资源
            for file in subscriber.sub_files.values():
                file.close()
            subscriber.destroy_node()
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    pass
