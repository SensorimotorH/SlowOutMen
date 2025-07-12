# Vision Understanding System

### 介绍

本仓库包含一个综合的视觉理解系统，集成了目标检测、分割和点云处理功能。该系统旨在从 RGB-D 图像中检测目标、分割目标，并处理点云以进行进一步分析。

### 功能

- **目标检测**：使用 YOLO（You Only Look Once）进行高效且准确的目标检测。
- **目标分割**：使用 SAM2（Segment Anything Model2）进行高质量的目标分割。
- **点云处理**：将深度图像和掩码转换为点云，并与完整点云数据库进行匹配。
- **模块化设计**：易于与其他系统集成，并可根据具体应用进行定制。

### 文件结构

VisionUnderstanding/

├── object_detection.py    # 目标检测模块,yolo

├── object_segmentation.py # 目标分割模块,sam2

├── point_cloud_processing.py # 点云处理模块,生成匹配

├── main_task.py          # 主任务流程控制

├── api.py                # 接口文件，方便直接调用

└── init.py               # 初始化文件

##调用demo

```python
# 导入所需的模块
from VisionUnderstanding.api import VisionAPI
from s2r2025.simple_api.ros2_mmk2_api import MMK2_Controller, MMK2_Receiver
import rclpy

# 初始化 ROS2 节点
rclpy.init()

# 创建接收器
receiver = MMK2_Receiver()

# 创建控制器（需要传入接收器来获取关节状态）
controller = MMK2_Controller(receiver)

# 相机内参
fx = 735.959835
fy = 735.959835
cx = 960
cy = 540

# 初始化视觉 API
vision_api = VisionAPI(
    yolo_model_path="path_to_your_yolo_model",
    sam_checkpoint="path_to_your_sam_checkpoint",
    model_type="vit_h", # SAM-1
    fx=fx,
    fy=fy,
    cx=cx,
    cy=cy,
    debug=True
)

# 目标类别名称列表
target_class_names = ['box', 'sheet','disk','carton','apple','plate','drawer','teacup','clock','kettle','xbox','bowl','scissors','book']

# 处理 RGB-D 图像
vision_api.process_rgb_d_images(target_class_names)

# 确保 ROS2 节点正确关闭
rclpy.shutdown()

```

### object_detection.py

#### 简介

该文件实现了目标检测模块，使用 YOLO 模型进行目标检测，提供了从图像中检测目标、处理相机图像以及检查目标存在性的功能。
函数接口说明:

#### 1. ObjectDetector 类
初始化方法

``` Python
def __init__(self, model_path, debug=False, debug_save_dir="debug_images")
#描述: 初始化目标检测器，加载 YOLO 模型，设置调试模式和调试图片保存文件夹。

#参数:
model_path: YOLO 模型路径。
debug: 是否启用调试模式，默认为 False。
debug_save_dir: 调试图片保存文件夹，默认为 "debug_images"。

``` 

#### 方法

``` Python
a. def detect_objects_from_image(self, image)
#描述: 从单张图像中检测目标，返回检测框和对应类别名称。
#参数:
#image: 输入的 RGB 图像 (numpy 数组)。
#返回值: 检测结果，列表，每个元素为字典，包含 "class_name"（类别名称）和 "box"（检测框坐标）。

b. def process_head_camera(self, receiver)
#描述: 处理头部相机图像并进行目标检测，获取 RGB 和深度图像，进行目标检测。
#参数:
#receiver: 数据接收器，用于获取头部相机的 RGB 和深度图像。
#返回值: 检测结果和图像信息，字典，包含 "rgb_image"（RGB 图像）、"depth_image"（深度图像）和 "detections"（检测结果）。

c. def check_target_presence(self, detection_results, target_class_names)
#描述: 检查检测结果中是否存在指定的目标类别。
#参数:
#detection_results: 检测结果。
#target_class_names: 目标类别名称列表。
#返回值: 布尔值，表示是否存在目标。
```

#### 使用示例

```python
# 初始化目标检测器
detector = ObjectDetector(model_path="path_to_your_yolo_model", debug=True)

# 获取头部相机图像并进行目标检测
head_results = detector.process_head_camera(receiver)

# 检查是否存在目标
if head_results:
    detection_results = head_results["detections"]
    target_class_names = ['kettle', 'apple']
    if detector.check_target_presence(detection_results, target_class_names):
        print("检测到目标")
    else:
        print("未检测到目标")
#注意事项
#确保 YOLO 模型路径正确。
#根据实际需求调整调试模式和调试图片保存文件夹。
#确保输入图像格式正确，为 RGB 格式的 numpy 数组。
```




