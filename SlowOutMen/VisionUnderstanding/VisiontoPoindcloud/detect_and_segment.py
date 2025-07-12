import cv2
import numpy as np
from typing import Dict, List

from SlowOutMen.VisionUnderstanding.VisiontoPoindcloud.yolov12.ultralytics.models import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def yolo_process(rgb: np.ndarray, debug=False,drawer=False) -> Dict:
    """
    使用YOLOv12模型处理RGB图像,返回检测结果。
    :param rgb: 输入的RGB图像,numpy数组格式,形状为(H, W, 3)
    :return: 检测结果字典,包含classname和boundingbox
    """

    # 加载并运行YOLOv12模型
    if drawer:
        model = YOLO(r'/workspace/SlowOutMen/VisionUnderstanding/VisiontoPoindcloud/checkpoints/best.pt')
    else:
        model = YOLO(
            r'/workspace/SlowOutMen/VisionUnderstanding/VisiontoPoindcloud/checkpoints/best_128_200_984.pt')
    
    results = model.predict(source=rgb, save=debug)
    # 提取 boxes 信息
    boxes = results[0].boxes
    # 遍历每个检测到的框
    yolo_results_dict = []  # 用于存储所有检测结果的字典列表

    for box in boxes:
        # 获取边界框的坐标 (x1, y1, x2, y2)
        bbox = box.xyxy[0].tolist()  # 转换为列表格式
        # 获取类别 ID
        class_id = int(box.cls[0])
        # 获取类别名称
        class_name = results[0].names[class_id]
        # 获取置信度
        confidence = box.conf[0].item()

        
        
        # 打包成字典
        yolo_result_dict = {
            "class_name": class_name,
            "box": bbox,
            "confidence": confidence
        }
        yolo_results_dict.append(yolo_result_dict)

    return yolo_results_dict


def is_object_in_yolo_result(yolo_result: List[Dict], obj_name: str) -> bool:
    # 如果 obj_name 是字符串,转换为列表以便统一处理
    if isinstance(obj_name, str):
        obj_name = [obj_name]

    # 提取 class_name 到列表
    class_names = [item["class_name"] for item in yolo_result]

    for element in obj_name:
        # 检查元素是否在搜索列表中
        if element in class_names:
            return True


def sam_process(rgb: np.ndarray, yolo_result: List[Dict]) -> Dict:
    # 提取 class_name 到列表
    class_names = [item["class_name"] for item in yolo_result]
    # 提取 box 到列表
    boxes = [item["box"] for item in yolo_result]

    # sam2配置过程
    checkpoint = "/workspace/SlowOutMen/VisionUnderstanding/VisiontoPoindcloud/checkpoints/sam2.1_hiera_tiny.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    sam2_model = build_sam2(model_cfg, checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(rgb)

    # 循环识别每一类的mask
    classname_mask_dict = {}
    for classname, box in zip(class_names, boxes):
        # 转换 box 格式为 numpy 数组
        input_box = np.array(box)

        # 使用 box 提示进行预测
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],  # SAM2 需要 box 的形状为 [1, 4]
            multimask_output=False,  # 设置为 False 以获取单个 mask
        )
        classname_mask_dict[classname] = masks

    return classname_mask_dict


def yolo_to_sam(rgb: np.ndarray) -> Dict:
    result_dict = yolo_process(rgb)
    sam2_mask = sam_process(rgb, result_dict)

    return sam2_mask


def main():
    image_path = r'E:\yolov12\test316.png'
    rgb_image = cv2.imread(image_path)

    # 调用rgb_process.yolo_process函数
    result_dict = yolo_process(rgb_image)
    print(result_dict)

    # 调用rgb_process.is_object_in_yolo_result函数
    existance = is_object_in_yolo_result(result_dict, "clock")
    print(existance)

    # 调用rgb_process.sam_process函数
    sam2_mask = sam_process(rgb_image, result_dict)
    print(sam2_mask)


if __name__ == "__main__":
    image_path = r'/workspace/SlowOutMen/VisionUnderstanding/vision2poindcloud/debug_images/head_rgb_test.png'
    rgb_image = cv2.imread(image_path)
    obj_mask = yolo_to_sam(rgb_image)
    print(obj_mask)
