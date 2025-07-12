from VisionUnderstanding.VisiontoPoindcloud.detect_and_segment import yolo_process, sam_process, is_object_in_yolo_result
from SlowOutMen.Communication.ros2_mmk2_api import MMK2_Controller, MMK2_Receiver
from SlowOutMen.Communication.ros2_mmk2_api import MMK2_Receiver, MMK2_Controller
from SlowOutMen.PoseTransform.utils.mmk2.mmk2_fk import MMK2FK
from SlowOutMen.TaskUnderstanding.task_parser import TaskParser

import rclpy
import open3d as o3d
import numpy as np
import cv2
import time
from typing import Tuple, List, Dict, Any
import os
from datetime import datetime


class RosVisionModule():
    def __init__(self, receiver: MMK2_Receiver, controller: MMK2_Controller):
        self.receiver = receiver
        self.controller = controller
        self.mmk2_fik = MMK2FK()

    def _get_rgb_and_depth(self) -> Tuple[np.ndarray, np.ndarray]:
        try:
            head_rgb = self.receiver.get_head_rgb()
            head_depth = self.receiver.get_head_depth()

            return head_rgb, head_depth
        except Exception as e:
            print(f"获取视觉数据时发生错误: {e}")
            return None, None

    def get_empty_area_position(self, table: str) -> List[float]:
        rgb, depth = self._get_rgb_and_depth()
        yolo_results = yolo_process(rgb)
        detected_items = set(item['class_name'] for item in yolo_results)
        if table == "left":
            points = np.array([[120, 290], [120, 430], [630, 450], [630, 300]])
        elif table == "right":
            points = np.array([[200, 300], [200, 500], [500, 500], [500, 300]])

        # 将“瞄定框”转换为矩形格式（找到最小外接矩形）
        x_min = np.min(points[:, 0])
        y_min = np.min(points[:, 1])
        x_max = np.max(points[:, 0])
        y_max = np.max(points[:, 1])

        num_columns = 5  # 动态划分，按照识别的类别数量
        column_width = (x_max - x_min) // num_columns

        # 创建小框
        small_boxes = []
        for i in range(num_columns):
            x_start = x_min + i * column_width
            x_end = x_start + column_width
            small_box = [x_start, y_min, x_end, y_max]
            small_boxes.append(small_box)

        # 提取 class_name 为 'drawer' 的检测结果
        drawer_detections = [
            detection for detection in yolo_results if detection['class_name'] == 'drawer']

        # 找出除去 YOLO 检测框之外，面积最大的小框
        max_empty_area = 0
        best_empty_box = None

        for small_box in small_boxes:
            small_x_start, small_y_start, small_x_end, small_y_end = small_box
            small_box_area = (small_x_end - small_x_start) * \
                (small_y_end - small_y_start)
            # 检查小框是否包含 'drawer' 检测框
            contains_drawer = False
            for drawer in drawer_detections:
                drawer_box = drawer['box']
                drawer_x_start, drawer_y_start, drawer_x_end, drawer_y_end = map(
                    int, drawer_box)

                # 计算小框与 'drawer' 检测框的交集区域
                x1 = max(small_x_start, drawer_x_start)
                y1 = max(small_y_start, drawer_y_start)
                x2 = min(small_x_end, drawer_x_end)
                y2 = min(small_y_end, drawer_y_end)

                if x1 < x2 and y1 < y2:
                    contains_drawer = True
                    break

            if contains_drawer:
                continue  # 跳过包含 'drawer' 的小框

            # 计算小框与 YOLO 检测框的交集面积（排除 'drawer'）
            overlap_area = 0
            for detection in yolo_results:
                if detection['class_name'] == 'drawer':
                    continue  # 跳过 'drawer' 检测框

                item_box = detection['box']
                item_x_start, item_y_start, item_x_end, item_y_end = map(
                    int, item_box)

                # 计算交集区域
                x1 = max(small_x_start, item_x_start)
                y1 = max(small_y_start, item_y_start)
                x2 = min(small_x_end, item_x_end)
                y2 = min(small_y_end, item_y_end)

                if x1 < x2 and y1 < y2:
                    overlap_area += (x2 - x1) * (y2 - y1)

            # 计算空闲面积
            empty_area = small_box_area - overlap_area

            # 更新最大空闲面积
            if empty_area > max_empty_area:
                max_empty_area = empty_area
                best_empty_box = small_box

        if best_empty_box:
            x_start, y_start, x_end, y_end = best_empty_box
            center_x = (x_start + x_end) // 2
            center_y = (y_start + y_end) // 2

        K = self.receiver.get_head_rgb_camera_info().k
        fx = K[0]
        fy = K[4]
        cx = K[2]
        cy = K[5]
        Z = round(depth[int(center_y), int(center_x)]/1000, 3)
        X = round((int(center_x)-cx)*Z/fx, 3)
        Y = round((int(center_y)-cy)*Z/fy, 3)
        vector = [X, Y, Z]

        return vector

    def is_object_presence(self, obj_name: str, debug=False) -> bool:
        rgb, _ = self._get_rgb_and_depth()
        yolo_result = yolo_process(rgb, debug)
        if debug:
            print(yolo_result)
        existance = is_object_in_yolo_result(yolo_result, obj_name)
        print(f"{obj_name}是否存在:{existance}")
        return existance

    def get_sam_result(self, obj_name: str, debug=False) -> Dict[str, Any]:
        rgb, _ = self._get_rgb_and_depth()
        yolo_result = yolo_process(rgb, debug)
        if debug:
            print(yolo_result)
        existance = is_object_in_yolo_result(yolo_result, obj_name)
        print(f"{obj_name}是否存在:{existance}")
        sam2_result = sam_process(rgb, yolo_result)
        if debug:
            print(sam2_result)
        return sam2_result

    def get_Toc_from_yolo(self, object_name: str,drawer=False) -> List[float]:
        K = self.receiver.get_head_rgb_camera_info().k
        fx = K[0]
        fy = K[4]
        cx = K[2]
        cy = K[5]
        rgb, depth = self._get_rgb_and_depth()
        yolo_results = yolo_process(rgb, drawer=drawer)
        target_box = [item["box"]
                      for item in yolo_results if item["class_name"] == object_name]
        x0, y0, x1, y1 =map(int,target_box[0])

        x = int((x0 + x1) / 2)
        y = int((y0 + y1) / 2)
        Z = round(depth[y, x] / 1000, 3)
        X = round((x - cx) * Z / fx, 3)
        Y = round((y - cy) * Z / fy, 3)
        vector = [X, Y, Z]
        print(
            f"binding box of {object_name} is: {np.round(np.array(target_box[0]), 3)}")
        print(f"{object_name} position in camera is: {vector}")
        return vector

    def get_Toc_from_yolo_result(self, yolo_result: List[Dict], object_name: str) -> List[float]:
        K = self.receiver.get_head_rgb_camera_info().k
        fx = K[0]
        fy = K[4]
        cx = K[2]
        cy = K[5]
        _, depth = self._get_rgb_and_depth()
        yolo_results = yolo_result
        target_box = [item["box"]
                      for item in yolo_results if item["class_name"] == object_name]

        x0, y0, x1, y1 = map(int, target_box[0])

        x = int((x0+x1)/2)
        y = int((y0+y1)/2)
        Z = round(depth[y, x]/1000, 3)
        X = round((x-cx)*Z/fx, 3)
        Y = round((y-cy)*Z/fy, 3)

        vector = [X, Y, Z]
        return vector

    def get_Toc_box_from_yolo(self, prop_name: str) -> List[float]:
        K = self.receiver.get_head_rgb_camera_info().k
        fx = K[0]
        fy = K[4]
        cx = K[2]
        cy = K[5]
        rgb, depth = self._get_rgb_and_depth()
        yolo_results = yolo_process(rgb)

        # 获取所有box和prop的检测结果
        target_boxes = [item["box"]
                        for item in yolo_results if item["class_name"] == "box"]

        # 如果检测到超过2个box,只保留距离中心点最近的两个
        if len(target_boxes) > 2:
            # 计算图像中心点
            rgb_center = [rgb.shape[1]/2, rgb.shape[0]/2]

            # 计算每个box的中心点到图像中心的距离
            box_distances = []
            for box in target_boxes:
                box_center = [(box[0] + box[2])/2, (box[1] + box[3])/2]
                dist = (box_center[0] - rgb_center[0])**2 + \
                    (box_center[1] - rgb_center[1])**2
                box_distances.append((dist, box))

            # 按距离排序并只保留最近的两个box
            box_distances.sort(key=lambda x: x[0])
            target_boxes = [item[1] for item in box_distances[:2]]

        target_prop = [item["box"]
                       for item in yolo_results if item["class_name"] == prop_name]
        other_props = [item["box"] for item in yolo_results
                       if item["class_name"] != "box" and item["class_name"] != prop_name]

        # 如果只有一个box,直接使用它
        if len(target_boxes) == 1:
            selected_box = target_boxes[0]

        # 如果有两个box
        elif len(target_boxes) == 2:
            # 如果检测到目标prop
            if len(target_prop) > 0:
                # 计算两个box到prop的距离,选择距离近的
                prop_center = [(target_prop[0][0] + target_prop[0][2])/2,
                               (target_prop[0][1] + target_prop[0][3])/2]

                dist1 = ((target_boxes[0][0] + target_boxes[0][2])/2 - prop_center[0])**2 + \
                        ((target_boxes[0][1] + target_boxes[0]
                         [3])/2 - prop_center[1])**2
                dist2 = ((target_boxes[1][0] + target_boxes[1][2])/2 - prop_center[0])**2 + \
                        ((target_boxes[1][1] + target_boxes[1]
                         [3])/2 - prop_center[1])**2

                selected_box = target_boxes[0] if dist1 < dist2 else target_boxes[1]

            # 如果没检测到目标prop但检测到其他prop
            elif len(other_props) == 1:
                # 计算两个box到其他prop的距离,选择距离远的
                other_center = [(other_props[0][0] + other_props[0][2])/2,
                                (other_props[0][1] + other_props[0][3])/2]

                dist1 = ((target_boxes[0][0] + target_boxes[0][2])/2 - other_center[0])**2 + \
                        ((target_boxes[0][1] + target_boxes[0]
                         [3])/2 - other_center[1])**2
                dist2 = ((target_boxes[1][0] + target_boxes[1][2])/2 - other_center[0])**2 + \
                        ((target_boxes[1][1] + target_boxes[1]
                         [3])/2 - other_center[1])**2

                selected_box = target_boxes[0] if dist1 > dist2 else target_boxes[1]

            # 如果检测到两个其他prop或没有检测到任何prop
            else:
                selected_box = target_boxes[0]
        else:
            raise ValueError("未检测到box或检测到超过2个box")

        # 计算选定box的3D坐标
        x0, y0, x1, y1 = map(int, selected_box)
        x = int((x0+x1)/2)
        y = int((y0+y1)/2)
        Z = round(depth[y, x]/1000, 3)
        X = round((x-cx)*Z/fx, 3)
        Y = round((y-cy)*Z/fy, 3)
        vector = [X, Y, Z]
        return vector

    def get_Toc_prop_from_yolo(self, prop_name: str) -> List[float]:
        K = self.receiver.get_head_rgb_camera_info().k
        fx = K[0]
        fy = K[4]
        cx = K[2]
        cy = K[5]
        rgb, depth = self._get_rgb_and_depth()
        yolo_results = yolo_process(rgb)

        # 获取所有box和prop的检测结果
        target_box = [item["box"]
                      for item in yolo_results if item["class_name"] == prop_name]

        print(f"target_box:{target_box}")
        # 计算选定box的3D坐标
        x0, y0, x1, y1 = target_box[0][0], target_box[0][1], target_box[0][2], target_box[0][3]
        x = int((x0+x1)/2)
        y = int((y0+y1)/2)
        Z = round(depth[y, x]/1000, 3)
        X = round((x-cx)*Z/fx, 3)
        Y = round((y-cy)*Z/fy, 3)
        vector = [X, Y, Z]
        return vector

    def get_Toc_box_from_yolo_result(self, yolo_result: List[Dict], prop_name: str) -> List[float]:
        K = self.receiver.get_head_rgb_camera_info().k
        fx = K[0]
        fy = K[4]
        cx = K[2]
        cy = K[5]
        rgb, depth = self._get_rgb_and_depth()
        yolo_results = yolo_result
        # 获取所有box和prop的检测结果
        target_boxes = [item["box"]
                        for item in yolo_results if item["class_name"] == "box"]

        # 如果检测到超过2个box,只保留距离中心点最近的两个
        if len(target_boxes) > 2:
            # 计算图像中心点
            rgb_center = [rgb.shape[1]/2, rgb.shape[0]/2]

            # 计算每个box的中心点到图像中心的距离
            box_distances = []
            for box in target_boxes:
                box_center = [(box[0] + box[2])/2, (box[1] + box[3])/2]
                dist = (box_center[0] - rgb_center[0])**2 + \
                    (box_center[1] - rgb_center[1])**2
                box_distances.append((dist, box))

            # 按距离排序并只保留最近的两个box
            box_distances.sort(key=lambda x: x[0])
            target_boxes = [item[1] for item in box_distances[:2]]

        target_prop = [item["box"]
                       for item in yolo_results if item["class_name"] == prop_name]
        other_props = [item["box"] for item in yolo_results
                       if item["class_name"] != "box" and item["class_name"] != prop_name]

        # 如果只有一个box,直接使用它
        if len(target_boxes) == 1:
            selected_box = target_boxes[0]

        # 如果有两个box
        elif len(target_boxes) == 2:
            # 如果检测到目标prop
            if len(target_prop) > 0:
                # 计算两个box到prop的距离,选择距离近的
                prop_center = [(target_prop[0][0] + target_prop[0][2])/2,
                               (target_prop[0][1] + target_prop[0][3])/2]

                dist1 = ((target_boxes[0][0] + target_boxes[0][2])/2 - prop_center[0])**2 + \
                        ((target_boxes[0][1] + target_boxes[0]
                         [3])/2 - prop_center[1])**2
                dist2 = ((target_boxes[1][0] + target_boxes[1][2])/2 - prop_center[0])**2 + \
                        ((target_boxes[1][1] + target_boxes[1]
                         [3])/2 - prop_center[1])**2

                selected_box = target_boxes[0] if dist1 < dist2 else target_boxes[1]

            # 如果没检测到目标prop但检测到其他prop
            elif len(other_props) == 1:
                # 计算两个box到其他prop的距离,选择距离远的
                other_center = [(other_props[0][0] + other_props[0][2])/2,
                                (other_props[0][1] + other_props[0][3])/2]

                dist1 = ((target_boxes[0][0] + target_boxes[0][2])/2 - other_center[0])**2 + \
                        ((target_boxes[0][1] + target_boxes[0]
                         [3])/2 - other_center[1])**2
                dist2 = ((target_boxes[1][0] + target_boxes[1][2])/2 - other_center[0])**2 + \
                        ((target_boxes[1][1] + target_boxes[1]
                         [3])/2 - other_center[1])**2

                selected_box = target_boxes[0] if dist1 > dist2 else target_boxes[1]

            # 如果检测到两个其他prop或没有检测到任何prop
            else:
                selected_box = target_boxes[0]
        else:
            raise ValueError("未检测到box或检测到超过2个box")

        # 计算选定box的3D坐标
        x0, y0, x1, y1 = map(int, selected_box)
        x = int((x0+x1)/2)
        y = int((y0+y1)/2)
        Z = round(depth[y, x]/1000, 3)
        X = round((x-cx)*Z/fx, 3)
        Y = round((y-cy)*Z/fy, 3)
        vector = [X, Y, Z]
        return vector

    def find_space_for_box(self, table: str, debug: bool = False) -> Tuple[float, float]:
        """
        查找桌子上靠近机器人一侧最适合放置盒子的中心位置 (x0, y0)

        :param table: "left" 或 "right"
        :param debug: 是否启用调试信息与可视化保存
        :return: (x0, y0)，若未找到则返回空元组
        """
        # Step 1: 获取图像与YOLO结果
        rgb, depth = self._get_rgb_and_depth()
        yolo_results = yolo_process(rgb)
        height, width = rgb.shape[:2]

        # Step 2: 桌面下缘点和放置线位置配置
        if table == "left":
            line_y = 320  # TODO: 根据需要微调
            table_edge_x1, table_edge_y1 = 80, 442
            table_edge_x2, table_edge_y2 = 640, 460
        elif table == "right":
            line_y = 330
            table_edge_x1, table_edge_y1 = 75, 450
            table_edge_x2, table_edge_y2 = 540, 460
        else:
            raise ValueError(
                f"Invalid table name '{table}', must be 'left' or 'right'.")

        # Step 3: 拟合桌面下缘直线 y = ax + b
        a = (table_edge_y2 - table_edge_y1) / (table_edge_x2 - table_edge_x1)
        b = table_edge_y1 - a * table_edge_x1

        def table_edge_y(x: float) -> float:
            return a * x + b

        # Step 4: 筛选与 line_y 相交以及在line_y之下的物体 bbox
        intersecting_boxes = []
        for item in yolo_results:
            x1, y1, x2, y2 = map(int, item["box"])
            if y1 >= line_y:
                intersecting_boxes.append((x1, x2))
            elif y1 <= line_y <= y2:
                intersecting_boxes.append((x1, x2))

        # Step 5: 找最大 gap
        intersecting_boxes.sort()
        gaps = []
        last_x = table_edge_x1
        for x1, x2 in intersecting_boxes:
            if x1 > last_x:
                gaps.append((last_x, x1))
            last_x = max(last_x, x2)
        if last_x < table_edge_x2:
            gaps.append((last_x, table_edge_x2))

        print(f"gaps:{gaps}")

        min_gap_width = 0  # 可调整
        widest_gap = max(
            (gap for gap in gaps if (gap[1] - gap[0]) >= min_gap_width),
            key=lambda g: g[1] - g[0],
            default=None
        )

        # Step 6: 计算输出坐标
        if widest_gap:
            x0 = (widest_gap[0] + widest_gap[1]) / 2
            y0 = (line_y + table_edge_y(x0)) / 2
        else:
            raise ValueError("未找到合适的放置位置！")

        # Step 7: Debug 可视化与保存
        vis_image = rgb.copy()

        # 画所有bbox
        for item in yolo_results:
            x1, y1, x2, y2 = map(int, item["box"])
            color = (0, 0, 255) if y1 <= line_y <= y2 else (255, 0, 0)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

        # 画放置线
        cv2.line(vis_image, (0, line_y), (width, line_y), (255, 255, 255), 2)

        # 画桌子边缘线
        cv2.line(vis_image, (table_edge_x1, table_edge_y1),
                 (table_edge_x2, table_edge_y2), (128, 128, 0), 2)

        # 画放置位置点和区域
        if widest_gap:
            cv2.rectangle(vis_image, (int(widest_gap[0]), line_y - 10),
                          (int(widest_gap[1]), line_y + 10), (0, 255, 0), -1)
            cv2.circle(vis_image, (int(x0), int(y0)), 6, (0, 255, 255), -1)
        else:
            print("未找到合适位置！")

        if debug:
            # 保存原图和可视化图
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = "/workspace/SlowOutMen/VisionUnderstanding/debug_pic"
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(
                save_dir, f"original_{table}_{timestamp}.png"), rgb)
            cv2.imwrite(os.path.join(
                save_dir, f"debug_{table}_{timestamp}.png"), vis_image)

            # # 显示图像
            # cv2.imshow(f"Placement Debug - {table}", vis_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        K = self.receiver.get_head_rgb_camera_info().k
        fx = K[0]
        fy = K[4]
        cx = K[2]
        cy = K[5]
        Z = round(depth[int(y0), int(x0)]/1000, 3)
        X = round((int(x0)-cx)*Z/fx, 3)
        Y = round((int(y0)-cy)*Z/fy, 3)
        vector = [X, Y, Z]

        return vector


def get_pointcloud_from_depth(depth: np.ndarray, K: np.ndarray, sam_result: Dict[str, np.ndarray], obj_name: str, debug=False):
    if depth is None or sam_result is None or obj_name not in sam_result:
        print(f"Error: 无法生成点云,原因:深度图或掩码为空,或未找到目标类别 {obj_name}")
        return None

    # 获取目标物体的掩码
    mask = sam_result[obj_name]
    print(mask)
    # 检查掩码和深度图的形状是否匹配
    # 确保掩码是二维数组
    if mask.ndim == 4:
        mask = mask.squeeze(1)
    print(mask.shape, depth.shape)
    mask = np.squeeze(mask, axis=0)  # 形状变为 (1080, 1920)
    if mask.shape != depth.shape:
        print("Error: 掩码和深度图的形状不匹配")
        return None

    # 应用掩码到深度图
    masked_depth = np.where(mask, depth, 0)
    depth_meters = masked_depth.astype(float) / 1000.0  # 转换为米

    # 相机内参
    FX = K[0]
    FY = K[4]
    CX = K[2]
    CY = K[5]

    # 生成像素网格
    height, width = depth_meters.shape
    u = np.arange(width)
    v = np.arange(height)
    u_grid, v_grid = np.meshgrid(u, v)

    # 计算三维坐标
    z = depth_meters
    x = (u_grid - CX) * z / FX
    y = (v_grid - CY) * z / FY

    # 合并坐标并过滤无效点（深度为0）
    points = np.stack([x, y, z], axis=-1)
    valid_points = points[z > 0]

    if valid_points.size == 0:
        print(f"类别 {obj_name} 无有效点云数据,无法生成点云")
        return None

    # 创建点云
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(valid_points)

    # 调试模式下保存点云
    if debug:
        output_path = rf"/workspace/SlowOutMen/VisionUnderstanding/vision2poindcloud/{obj_name}.ply"
        o3d.io.write_point_cloud(output_path, pointcloud)
        print(f"已保存 {obj_name} 的点云文件至 {output_path}")

    return pointcloud


def get_pointcloud(receiver: MMK2_Receiver, controller: MMK2_Controller, obj_name: str, debug=False):
    K = receiver.get_head_rgb_camera_info().k
    vision_module = RosVisionModule(receiver, controller)
    head_rgb, head_depth = vision_module._get_rgb_and_depth()
    print(f"rgb_shape:{head_rgb.shape},depth_shape:{head_depth.shape}")
    head_rgb = cv2.cvtColor(head_rgb, cv2.COLOR_BGR2RGB)
    sam_result = vision_module.get_sam_result(obj_name, debug)
    object_pointcloud = get_pointcloud_from_depth(
        head_depth, K, sam_result, obj_name)

    return object_pointcloud


def main_space():
    rclpy.init()
    receiver = MMK2_Receiver()
    controller = MMK2_Controller(receiver)
    task_info = receiver.get_taskinfo()
    taskParser = TaskParser()
    task_key = taskParser.parse_task(task_info)
    table_index = task_key['target_table']
    vision_module = RosVisionModule(receiver, controller)
    from SlowOutMen.Manipulation.utils.pose_transform_base import PoseTransform
    from SlowOutMen.Navigation.utils.move_to_point_ros2 import MoveToPoint
    posetran = PoseTransform(receiver)
    move_to_point = MoveToPoint(receiver, controller)
    if table_index == "left":
        move_to_point.move_to_point([0.3, 0.5, 0])
        controller.set_head_pitch_angle(0.25)
        time.sleep(1)
    elif table_index == "right":
        move_to_point.move_to_point([0.3, 0, -90])
        controller.set_head_pitch_angle(0.3)
        time.sleep(1)
    result = vision_module.find_space_for_box(table_index, debug=True)
    print(f"result:{result}")


def main_pointcloud():
    rclpy.init()
    receiver = MMK2_Receiver()
    controller = MMK2_Controller(receiver)
    controller.set_head_slide_position(0.4)
    import time
    time.sleep(3)
    obj_name = 'box'
    obj_ply = get_pointcloud(receiver, obj_name)
    obj_name = 'box'
    obj_ply = get_pointcloud(receiver, obj_name)
    o3d.visualization.draw_geometries(
        [obj_ply], window_name=obj_name, width=800, height=600)


if __name__ == "__main__":
    main_space()
