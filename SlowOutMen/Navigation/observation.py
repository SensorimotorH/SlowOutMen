import cv2
import numpy as np
import os
from VisionUnderstanding.VisiontoPoindcloud.detect_and_segment import yolo_process


# 读取图片
image_path = '/workspace/SlowOutMen/VisionUnderstanding/debug/left_table_result.jpg'  # 替换为你的输入图片路径

image = cv2.imread(image_path)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

if image is None:
    print(f"无法读取图片: {image_path}")
    exit()

yolo_results = yolo_process(image)
print(yolo_results)

detected_items = set(item['class_name'] for item in yolo_results)
count = len(detected_items)

# 定义“瞄定框”（格式：[x1, y1, x2, y2, x3, y3, x4, y4]）
points = np.array([[80, 250], [80, 430], [630, 450], [630, 250]])

# 将“瞄定框”转换为矩形格式（找到最小外接矩形）
x_min = np.min(points[:, 0])
y_min = np.min(points[:, 1])
x_max = np.max(points[:, 0])
y_max = np.max(points[:, 1])

# 绘制“瞄定框”
cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

# 将“瞄定框”划分为一行多列的小框
num_columns = count  # 划分为5列

# 计算每个小框的宽度
column_width = (x_max - x_min) // num_columns

# 创建小框
small_boxes = []
for i in range(num_columns):
    x_start = x_min + i * column_width
    x_end = x_start + column_width
    small_box = [x_start, y_min, x_end, y_max]
    small_boxes.append(small_box)

# 绘制小框
for box in small_boxes:
    x_start, y_start, x_end, y_end = box
    cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 0, 255), 1)

# 提取 class_name 为 'drawer' 的检测结果
drawer_detections = [
    detection for detection in yolo_results if detection['class_name'] == 'drawer']

# 绘制所有 YOLO 检测框
for detection in yolo_results:
    box = detection['box']
    x_start, y_start, x_end, y_end = map(int, box)
    cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)

# 找出除去包含 'drawer' 的小框之外，面积最大的小框
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
        item_x_start, item_y_start, item_x_end, item_y_end = map(int, item_box)

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

# 如果找到面积最大的空闲小框，绘制它并计算中心点坐标
if best_empty_box:
    x_start, y_start, x_end, y_end = best_empty_box
    cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 255, 255), 2)
    cv2.putText(image, "Largest Empty", (x_start, y_start - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # 计算中心点坐标
    center_x = (x_start + x_end) // 2
    center_y = (y_start + y_end) // 2
    cv2.circle(image, (center_x, center_y), 5, (0, 255, 255), -1)
    print(f"中心点坐标: ({center_x}, {center_y})")

# 指定保存路径

output_path = '/workspace/SlowOutMen/VisionUnderstanding/debug/obser.jpg'  # 替换为你的输出路径

# 确保路径存在，如果不存在则创建
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 保存结果
cv2.imwrite(output_path, image)

print(f"结果已保存到: {output_path}")
