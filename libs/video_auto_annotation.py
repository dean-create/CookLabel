"""
视频自动标注模块（单目标追踪版本 - ByteTrack + 卡尔曼滤波）
结合YOLOv8检测模型、ByteTrack算法和卡尔曼滤波器，实现高精度单目标追踪

工作流：
1. YOLO检测出目标 → 得到框
2. 单目标跟踪借鉴ByteTrack算法用IoU + 贪心匹配 → 把这些框和上帧的目标ID对应起来
3. 卡尔曼思想 → 让目标短暂被遮挡时还能"猜到"它在哪

主要功能：
1. 基于手动标注初始化单目标跟踪器
2. YOLOv8目标检测
3. ByteTrack算法：高低置信度检测分离 + 多阶段匹配
4. 卡尔曼滤波：运动预测 + 状态估计
5. 单目标跟踪（只追踪用户框选的目标）
6. 追踪框跟随目标移动
7. 自动生成标注文件

特点：
- 专注于单目标追踪，避免多目标干扰
- 借鉴ByteTrack的高低置信度分离策略
- 卡尔曼滤波器预测目标运动轨迹
- 处理目标被遮挡的情况，提高追踪鲁棒性
- 追踪框能够实时跟随目标移动
- 支持目标暂时消失后重新出现的情况

算法优势：
- 高置信度检测优先匹配，确保准确性
- 低置信度检测补充匹配，提高召回率
- 卡尔曼滤波平滑轨迹，减少抖动
- 运动预测处理遮挡，增强鲁棒性

作者：CookLabel项目组
版本：3.0（ByteTrack + 卡尔曼滤波版本）
"""

import cv2
import numpy as np
import os
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import torch
from ultralytics import YOLO


@dataclass
class DetectionBox:
    """检测框数据结构"""
    x1: float  # 左上角x坐标
    y1: float  # 左上角y坐标
    x2: float  # 右下角x坐标
    y2: float  # 右下角y坐标
    confidence: float  # 置信度
    class_id: int  # 类别ID
    class_name: str  # 类别名称


@dataclass
class TrackingBox:
    """跟踪框数据结构"""
    track_id: int  # 跟踪ID
    x1: float  # 左上角x坐标
    y1: float  # 左上角y坐标
    x2: float  # 右下角x坐标
    y2: float  # 右下角y坐标
    confidence: float  # 置信度
    class_id: int  # 类别ID
    class_name: str  # 类别名称
    frame_id: int  # 帧ID


class KalmanFilter:
    """
    卡尔曼滤波器，用于预测目标位置
    状态向量: [center_x, center_y, width, height, dx, dy, dw, dh]
    """
    
    def __init__(self):
        """初始化卡尔曼滤波器"""
        self.dt = 1.0  # 时间间隔
        
        # 状态向量 [x, y, w, h, dx, dy, dw, dh]
        self.x = np.zeros((8, 1))  # 状态向量
        
        # 状态转移矩阵
        self.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  # x = x + dx
            [0, 1, 0, 0, 0, 1, 0, 0],  # y = y + dy
            [0, 0, 1, 0, 0, 0, 1, 0],  # w = w + dw
            [0, 0, 0, 1, 0, 0, 0, 1],  # h = h + dh
            [0, 0, 0, 0, 1, 0, 0, 0],  # dx = dx
            [0, 0, 0, 0, 0, 1, 0, 0],  # dy = dy
            [0, 0, 0, 0, 0, 0, 1, 0],  # dw = dw
            [0, 0, 0, 0, 0, 0, 0, 1],  # dh = dh
        ], dtype=np.float32)
        
        # 观测矩阵 (只观测位置和尺寸)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ], dtype=np.float32)
        
        # 过程噪声协方差矩阵
        self.Q = np.eye(8, dtype=np.float32) * 0.1
        self.Q[4:, 4:] *= 0.01  # 速度变化的噪声更小
        
        # 观测噪声协方差矩阵
        self.R = np.eye(4, dtype=np.float32) * 10.0
        
        # 误差协方差矩阵
        self.P = np.eye(8, dtype=np.float32) * 1000.0
        
        self.is_initialized = False
    
    def initialize(self, bbox):
        """
        初始化卡尔曼滤波器
        
        参数:
        - bbox: (x1, y1, x2, y2) 格式的边界框
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        
        # 初始化状态向量 [x, y, w, h, dx, dy, dw, dh]
        self.x = np.array([
            [center_x],
            [center_y],
            [width],
            [height],
            [0],  # 初始速度为0
            [0],
            [0],
            [0]
        ], dtype=np.float32)
        
        self.is_initialized = True
    
    def predict(self):
        """
        预测下一帧的状态
        
        返回:
        - 预测的边界框 (x1, y1, x2, y2)
        """
        if not self.is_initialized:
            return None
        
        # 预测步骤
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
        # 转换为边界框格式
        center_x, center_y, width, height = self.x[0, 0], self.x[1, 0], self.x[2, 0], self.x[3, 0]
        x1 = center_x - width / 2.0
        y1 = center_y - height / 2.0
        x2 = center_x + width / 2.0
        y2 = center_y + height / 2.0
        
        return (x1, y1, x2, y2)
    
    def update(self, bbox):
        """
        更新卡尔曼滤波器状态
        
        参数:
        - bbox: (x1, y1, x2, y2) 格式的观测边界框
        """
        if not self.is_initialized:
            self.initialize(bbox)
            return
        
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        
        # 观测向量
        z = np.array([
            [center_x],
            [center_y],
            [width],
            [height]
        ], dtype=np.float32)
        
        # 更新步骤
        y = z - np.dot(self.H, self.x)  # 残差
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R  # 残差协方差
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # 卡尔曼增益
        
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)
    
    def get_current_bbox(self):
        """
        获取当前状态的边界框
        
        返回:
        - (x1, y1, x2, y2) 格式的边界框
        """
        if not self.is_initialized:
            return None
        
        center_x, center_y, width, height = self.x[0, 0], self.x[1, 0], self.x[2, 0], self.x[3, 0]
        x1 = center_x - width / 2.0
        y1 = center_y - height / 2.0
        x2 = center_x + width / 2.0
        y2 = center_y + height / 2.0
        
        return (x1, y1, x2, y2)


class SingleObjectTracker:
    """
    单目标跟踪器（借鉴ByteTrack + 卡尔曼滤波）
    工作流：
    1. YOLO检测出目标 → 得到框
    2. 单目标跟踪借鉴ByteTrack算法用IoU + 贪心匹配 → 把这些框和上帧的目标ID对应起来
    3. 卡尔曼思想 → 让目标短暂被遮挡时还能"猜到"它在哪
    """
    
    def __init__(self, max_disappeared: int = 30, 
                 high_iou_threshold: float = 0.3,
                 low_iou_threshold: float = 0.15):
        """
        初始化单目标跟踪器
        
        参数:
        - max_disappeared: 目标消失的最大帧数，超过则停止追踪
        - high_iou_threshold: 高置信度检测的IoU匹配阈值（降低以提高鲁棒性）
        - low_iou_threshold: 低置信度检测的IoU匹配阈值（ByteTrack思想）
        """
        self.target_track = None  # 当前追踪的目标信息
        self.max_disappeared = max_disappeared
        self.high_iou_threshold = high_iou_threshold
        self.low_iou_threshold = low_iou_threshold
        self.is_tracking = False  # 是否正在追踪
        self.track_id = 0  # 固定的追踪ID
        
        # 卡尔曼滤波器
        self.kalman_filter = KalmanFilter()
        
        # ByteTrack相关参数（降低阈值以提高检测率）
        self.high_conf_threshold = 0.5  # 高置信度阈值（从0.6降低到0.5）
        self.low_conf_threshold = 0.25  # 低置信度阈值（从0.3降低到0.25）
        
    def calculate_iou(self, box1: Tuple[float, float, float, float], 
                     box2: Tuple[float, float, float, float]) -> float:
        """
        计算两个边界框的IoU（交并比）
        
        参数:
        - box1, box2: (x1, y1, x2, y2) 格式的边界框
        
        返回:
        - IoU值 (0-1之间)
        """
        # 计算交集区域
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # 如果没有交集，返回0
        if x1 >= x2 or y1 >= y2:
            return 0.0
            
        # 计算交集面积
        intersection = (x2 - x1) * (y2 - y1)
        
        # 计算并集面积
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        # 避免除零错误
        if union == 0:
            return 0.0
            
        return intersection / union
    
    def initialize_target(self, manual_box: Dict, frame_id: int):
        """
        初始化要追踪的目标
        
        参数:
        - manual_box: 手动标注框 {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'class_id': id, 'class_name': name}
        - frame_id: 当前帧ID
        """
        bbox = (manual_box['x1'], manual_box['y1'], manual_box['x2'], manual_box['y2'])
        
        self.target_track = {
            'box': bbox,
            'class_id': manual_box['class_id'],
            'class_name': manual_box['class_name'],
            'disappeared': 0,
            'confidence': 1.0,  # 手动标注给予最高置信度
            'last_frame': frame_id,
            'state': 'tracked'  # 追踪状态: tracked, lost
        }
        
        # 初始化卡尔曼滤波器
        self.kalman_filter.initialize(bbox)
        
        self.is_tracking = True
        print(f"单目标追踪器已初始化，追踪目标: {manual_box['class_name']}")
        print(f"卡尔曼滤波器已初始化，初始位置: {bbox}")
    
    def update(self, detections: List[DetectionBox], frame_id: int) -> List[TrackingBox]:
        """
        更新单目标追踪器状态（ByteTrack + 卡尔曼滤波）
        
        工作流：
        1. YOLO检测出目标 → 得到框
        2. 单目标跟踪借鉴ByteTrack算法用IoU + 贪心匹配 → 把这些框和上帧的目标ID对应起来
        3. 卡尔曼思想 → 让目标短暂被遮挡时还能"猜到"它在哪
        
        参数:
        - detections: 当前帧的检测结果列表
        - frame_id: 当前帧ID
        
        返回:
        - 跟踪结果列表（最多包含一个目标）
        """
        tracking_results = []
        
        # 如果没有初始化目标，返回空结果
        if not self.is_tracking or self.target_track is None:
            return tracking_results
        
        # 步骤1: 卡尔曼滤波器预测当前帧位置
        predicted_bbox = self.kalman_filter.predict()
        if predicted_bbox is None:
            predicted_bbox = self.target_track['box']
        
        print(f"第 {frame_id} 帧: 卡尔曼预测位置 {predicted_bbox}")
        
        # 步骤2: ByteTrack算法 - 分离高低置信度检测
        high_conf_detections = [d for d in detections if d.confidence >= self.high_conf_threshold]
        low_conf_detections = [d for d in detections if self.low_conf_threshold <= d.confidence < self.high_conf_threshold]
        
        print(f"第 {frame_id} 帧: 高置信度检测 {len(high_conf_detections)} 个，低置信度检测 {len(low_conf_detections)} 个")
        
        # 步骤3: 第一轮匹配 - 使用高置信度检测和高IoU阈值
        best_match = None
        best_iou = 0
        match_source = "none"
        
        # 优先使用预测位置进行匹配
        reference_box = predicted_bbox
        
        # 与高置信度检测进行匹配
        for detection in high_conf_detections:
            det_box = (detection.x1, detection.y1, detection.x2, detection.y2)
            iou = self.calculate_iou(reference_box, det_box)
            
            # 优先匹配相同类别的检测结果
            if detection.class_id == self.target_track['class_id']:
                if iou > best_iou and iou > self.high_iou_threshold:
                    best_iou = iou
                    best_match = detection
                    match_source = "high_conf_same_class"
            # 如果没有相同类别的，考虑其他类别但要求更高的IoU
            elif best_match is None and iou > self.high_iou_threshold * 1.2:
                best_iou = iou
                best_match = detection
                match_source = "high_conf_diff_class"
        
        # 步骤4: 第二轮匹配 - 如果第一轮没有匹配成功，使用低置信度检测和低IoU阈值
        if best_match is None:
            for detection in low_conf_detections:
                det_box = (detection.x1, detection.y1, detection.x2, detection.y2)
                iou = self.calculate_iou(reference_box, det_box)
                
                # 对于低置信度检测，要求更严格的类别匹配
                if detection.class_id == self.target_track['class_id']:
                    if iou > best_iou and iou > self.low_iou_threshold:
                        best_iou = iou
                        best_match = detection
                        match_source = "low_conf_same_class"
        
        # 步骤5: 第三轮匹配 - 如果目标已经丢失，使用更宽松的条件尝试恢复追踪
        if best_match is None and self.target_track['state'] == 'lost' and self.target_track['disappeared'] < 10:
            # 当目标丢失时，降低IoU要求，但仍要求相同类别
            relaxed_iou_threshold = max(0.1, self.low_iou_threshold * 0.7)
            
            # 重新检查所有检测（包括高置信度和低置信度）
            all_detections = high_conf_detections + low_conf_detections
            for detection in all_detections:
                det_box = (detection.x1, detection.y1, detection.x2, detection.y2)
                iou = self.calculate_iou(reference_box, det_box)
                
                # 只匹配相同类别，但IoU要求更宽松
                if detection.class_id == self.target_track['class_id']:
                    if iou > best_iou and iou > relaxed_iou_threshold:
                        best_iou = iou
                        best_match = detection
                        match_source = "recovery_match"
                        print(f"第 {frame_id} 帧: 尝试恢复追踪，IoU={best_iou:.3f}，阈值={relaxed_iou_threshold:.3f}")
        
        # 步骤6: 根据匹配结果更新追踪状态
        if best_match is not None:
            # 找到匹配，更新追踪目标
            matched_bbox = (best_match.x1, best_match.y1, best_match.x2, best_match.y2)
            
            # 更新卡尔曼滤波器
            self.kalman_filter.update(matched_bbox)
            
            # 更新追踪信息
            self.target_track['box'] = matched_bbox
            self.target_track['confidence'] = best_match.confidence
            self.target_track['disappeared'] = 0
            self.target_track['last_frame'] = frame_id
            self.target_track['state'] = 'tracked'
            
            # 获取卡尔曼滤波器的当前估计位置（更平滑）
            kalman_bbox = self.kalman_filter.get_current_bbox()
            if kalman_bbox is not None:
                final_bbox = kalman_bbox
            else:
                final_bbox = matched_bbox
            
            tracking_results.append(TrackingBox(
                track_id=self.track_id,
                x1=final_bbox[0],
                y1=final_bbox[1],
                x2=final_bbox[2],
                y2=final_bbox[3],
                confidence=best_match.confidence,
                class_id=self.target_track['class_id'],
                class_name=self.target_track['class_name'],
                frame_id=frame_id
            ))
            
            print(f"第 {frame_id} 帧: 成功追踪到目标，IoU={best_iou:.3f}，匹配来源={match_source}")
            
        else:
            # 没有找到匹配，目标可能被遮挡
            self.target_track['disappeared'] += 1
            self.target_track['state'] = 'lost'
            
            if self.target_track['disappeared'] > self.max_disappeared:
                print(f"目标消失超过 {self.max_disappeared} 帧，停止追踪")
                self.is_tracking = False
                self.target_track = None
                return tracking_results
            else:
                # 使用卡尔曼滤波器预测的位置来维持追踪
                if predicted_bbox is not None:
                    # 降低置信度，但保持追踪
                    confidence = max(0.1, self.target_track['confidence'] - 0.05 * self.target_track['disappeared'])
                    
                    tracking_results.append(TrackingBox(
                        track_id=self.track_id,
                        x1=predicted_bbox[0],
                        y1=predicted_bbox[1],
                        x2=predicted_bbox[2],
                        y2=predicted_bbox[3],
                        confidence=confidence,
                        class_id=self.target_track['class_id'],
                        class_name=self.target_track['class_name'],
                        frame_id=frame_id
                    ))
                    
                    print(f"第 {frame_id} 帧: 目标被遮挡，使用卡尔曼预测位置 (消失计数: {self.target_track['disappeared']})")
                else:
                    # 如果卡尔曼滤波器也无法预测，使用上一帧位置
                    tracking_results.append(TrackingBox(
                        track_id=self.track_id,
                        x1=self.target_track['box'][0],
                        y1=self.target_track['box'][1],
                        x2=self.target_track['box'][2],
                        y2=self.target_track['box'][3],
                        confidence=max(0.1, self.target_track['confidence'] - 0.1 * self.target_track['disappeared']),
                        class_id=self.target_track['class_id'],
                        class_name=self.target_track['class_name'],
                        frame_id=frame_id
                    ))
                    
                    print(f"第 {frame_id} 帧: 目标被遮挡，保持上一帧位置 (消失计数: {self.target_track['disappeared']})")
        
        return tracking_results
    
    def reset(self):
        """重置追踪器"""
        self.target_track = None
        self.is_tracking = False
        
        # 重置卡尔曼滤波器
        self.kalman_filter = KalmanFilter()
        
        print("单目标追踪器已重置（包括卡尔曼滤波器）")


class YOLODetector:
    """
    YOLOv8检测器封装类
    负责加载模型和执行目标检测
    """
    
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.5):
        """
        初始化YOLO检测器
        
        参数:
        - model_path: YOLO模型文件路径，如果为None则使用官方预训练模型
        - confidence_threshold: 置信度阈值
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names = []
        
        # 加载模型
        self.load_model()
    
    def load_model(self):
        """加载YOLO模型"""
        try:
            # 如果没有指定模型路径，使用官方预训练模型
            if self.model_path is None or not os.path.exists(self.model_path):
                if self.model_path is not None:
                    print(f"⚠ 指定的模型文件不存在: {self.model_path}")
                print("正在使用YOLOv8官方预训练模型: yolov8n.pt")
                self.model_path = "yolov8n.pt"  # 使用nano版本，速度快
            
            print(f"正在加载YOLO模型: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # 获取类别名称
            if hasattr(self.model, 'names'):
                self.class_names = list(self.model.names.values())
            else:
                # 如果模型没有类别名称，使用COCO数据集的默认类别
                self.class_names = [
                    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
                ]
            
            print(f"✓ 模型加载成功，支持 {len(self.class_names)} 个类别")
            print(f"✓ 主要检测类别: {', '.join(self.class_names[:10])}...")
            
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            print("请确保已安装ultralytics库: pip install ultralytics")
            raise
    
    def detect(self, image: np.ndarray) -> List[DetectionBox]:
        """
        执行目标检测
        
        参数:
        - image: 输入图像 (BGR格式)
        
        返回:
        - 检测结果列表
        """
        if self.model is None:
            print("模型未加载，无法执行检测")
            return []
        
        try:
            # 执行推理
            results = self.model(image, conf=self.confidence_threshold, verbose=False)
            
            detections = []
            
            # 解析检测结果
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # 边界框坐标
                    confidences = result.boxes.conf.cpu().numpy()  # 置信度
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)  # 类别ID
                    
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes[i]
                        confidence = float(confidences[i])
                        class_id = int(class_ids[i])
                        
                        # 确保类别ID在有效范围内
                        if 0 <= class_id < len(self.class_names):
                            class_name = self.class_names[class_id]
                        else:
                            class_name = f"unknown_{class_id}"
                        
                        detections.append(DetectionBox(
                            x1=float(x1),
                            y1=float(y1),
                            x2=float(x2),
                            y2=float(y2),
                            confidence=confidence,
                            class_id=class_id,
                            class_name=class_name
                        ))
            
            return detections
            
        except Exception as e:
            print(f"检测过程中出错: {e}")
            return []


class AnnotationGenerator:
    """
    标注文件生成器
    负责将跟踪结果转换为YOLO格式的标注文件
    """
    
    def __init__(self, output_dir: str):
        """
        初始化标注生成器
        
        参数:
        - output_dir: 输出目录路径
        """
        self.output_dir = output_dir
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
    
    def convert_to_yolo_format(self, tracking_box: TrackingBox, 
                              image_width: int, image_height: int) -> str:
        """
        将跟踪框转换为YOLO格式
        
        参数:
        - tracking_box: 跟踪框对象
        - image_width: 图像宽度
        - image_height: 图像高度
        
        返回:
        - YOLO格式的标注字符串
        """
        # 计算中心点坐标和宽高（归一化）
        center_x = (tracking_box.x1 + tracking_box.x2) / 2.0 / image_width
        center_y = (tracking_box.y1 + tracking_box.y2) / 2.0 / image_height
        width = (tracking_box.x2 - tracking_box.x1) / image_width
        height = (tracking_box.y2 - tracking_box.y1) / image_height
        
        # 确保坐标在[0,1]范围内
        center_x = max(0, min(1, center_x))
        center_y = max(0, min(1, center_y))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        return f"{tracking_box.class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
    
    def save_frame_annotation(self, frame_name: str, tracking_results: List[TrackingBox],
                            image_width: int, image_height: int):
        """
        保存单帧的标注文件
        
        参数:
        - frame_name: 帧文件名（不含扩展名）
        - tracking_results: 跟踪结果列表
        - image_width: 图像宽度
        - image_height: 图像高度
        
        返回:
        - bool: 是否成功保存了标注文件
        """
        # 检查是否有追踪结果，没有结果则不保存文件
        if not tracking_results:
            print(f"⚠ 第 {frame_name} 帧没有追踪结果，跳过保存标注文件")
            return False
        
        # 确保labels子文件夹存在
        labels_dir = os.path.join(self.output_dir, 'labels')
        os.makedirs(labels_dir, exist_ok=True)
        
        # 将标注文件保存到labels子文件夹中
        annotation_file = os.path.join(labels_dir, f"{frame_name}.txt")
        
        with open(annotation_file, 'w', encoding='utf-8') as f:
            for tracking_box in tracking_results:
                yolo_line = self.convert_to_yolo_format(tracking_box, image_width, image_height)
                f.write(yolo_line + '\n')
        
        print(f"✓ 已保存标注文件: {annotation_file}")
        return True


class VideoAutoAnnotator:
    """
    视频自动标注主控制器（单目标追踪版本）
    整合检测器、单目标跟踪器和标注生成器，实现单目标自动标注流程
    """
    
    def __init__(self, output_dir: str, model_path: str = None,
                 confidence_threshold: float = 0.5,
                 high_iou_threshold: float = 0.5,
                 low_iou_threshold: float = 0.2,
                 max_disappeared: int = 30):
        """
        初始化视频自动标注器（单目标追踪版本 - ByteTrack + 卡尔曼滤波）
        
        参数:
        - output_dir: 输出目录
        - model_path: YOLO模型路径，如果为None则使用官方预训练模型
        - confidence_threshold: 检测置信度阈值
        - high_iou_threshold: 高置信度检测的IoU匹配阈值
        - low_iou_threshold: 低置信度检测的IoU匹配阈值（ByteTrack思想）
        - max_disappeared: 目标消失的最大帧数
        """
        self.detector = YOLODetector(model_path, confidence_threshold)
        self.tracker = SingleObjectTracker(
            max_disappeared=max_disappeared,
            high_iou_threshold=high_iou_threshold,
            low_iou_threshold=low_iou_threshold
        )
        self.annotation_generator = AnnotationGenerator(output_dir)
        
        self.frame_count = 0
        self.tracking_history = []  # 存储所有跟踪历史
        self.is_initialized = False  # 是否已初始化追踪目标
        
        # 初始化统计信息
        self.statistics = {
            'total_frames': 0,
            'total_tracks': 0,
            'active_tracks': 0,
            'total_detections': 0,
            'high_conf_detections': 0,
            'low_conf_detections': 0,
            'successful_tracks': 0,
            'lost_tracks': 0,
            'saved_images': 0,
            'saved_annotations': 0,
            'saved_labels': 0,  # 兼容性别名
            'tracking_start_frame': 0,
            'tracking_end_frame': 0,
            'target_class': '',
            'confidence_sum': 0.0,
            'tracking_success_rate': 0.0
        }
    
    def initialize_from_manual_annotation(self, image: np.ndarray, 
                                        manual_boxes: List[Dict]) -> List[TrackingBox]:
        """
        从手动标注初始化单目标跟踪器
        
        参数:
        - image: 第一帧图像
        - manual_boxes: 手动标注框列表，只使用第一个框进行单目标追踪
        
        返回:
        - 初始化的跟踪结果
        """
        if not manual_boxes:
            print("⚠ 没有手动标注框，无法初始化追踪器")
            return []
        
        # 只使用第一个手动标注框进行单目标追踪
        target_box = manual_boxes[0]
        if len(manual_boxes) > 1:
            print(f"⚠ 检测到 {len(manual_boxes)} 个标注框，单目标追踪模式只使用第一个: {target_box['class_name']}")
        
        print(f"正在初始化单目标追踪器，目标: {target_box['class_name']}")
        
        # 初始化追踪器
        self.tracker.initialize_target(target_box, self.frame_count)
        self.is_initialized = True
        
        # 更新统计信息
        self.statistics['total_tracks'] = 1  # 单目标追踪，轨迹数为1
        self.statistics['active_tracks'] = 1  # 初始化时轨迹为活跃状态
        self.statistics['tracking_start_frame'] = self.frame_count
        self.statistics['target_class'] = target_box['class_name']
        
        # 创建初始追踪结果
        tracking_results = [TrackingBox(
            track_id=self.tracker.track_id,
            x1=target_box['x1'],
            y1=target_box['y1'],
            x2=target_box['x2'],
            y2=target_box['y2'],
            confidence=1.0,
            class_id=target_box['class_id'],
            class_name=target_box['class_name'],
            frame_id=self.frame_count
        )]
        
        self.frame_count += 1
        
        # 记录跟踪历史
        self.tracking_history.extend(tracking_results)
        
        # 更新统计信息
        self.statistics['confidence_sum'] += 1.0
        self.statistics['high_conf_detections'] += 1  # 手动标注视为高置信度
        
        print(f"✓ 单目标追踪器初始化完成，开始追踪: {target_box['class_name']}")
        return tracking_results
    
    def process_frame(self, image: np.ndarray, frame_name: str = None, 
                     save_annotation: bool = False, save_image: bool = False) -> List[TrackingBox]:
        """
        处理单帧图像（单目标追踪）
        
        参数:
        - image: 输入图像
        - frame_name: 帧名称（用于保存标注文件）
        - save_annotation: 是否自动保存标注文件（默认False，由外部控制）
        - save_image: 是否保存图像文件（默认False，由外部控制）
        
        返回:
        - 跟踪结果列表（最多包含一个目标）
        """
        # 如果追踪器未初始化，返回空结果
        if not self.is_initialized:
            print("⚠ 追踪器未初始化，请先调用initialize_from_manual_annotation()")
            return []
        
        # 执行检测
        detections = self.detector.detect(image)
        print(f"第 {self.frame_count} 帧检测到 {len(detections)} 个目标")
        
        # 统计检测信息
        self.statistics['total_detections'] += len(detections)
        
        # 按置信度分类检测结果（用于统计）
        high_conf_threshold = 0.6
        low_conf_threshold = 0.3
        
        for detection in detections:
            if detection.confidence >= high_conf_threshold:
                self.statistics['high_conf_detections'] += 1
            elif detection.confidence >= low_conf_threshold:
                self.statistics['low_conf_detections'] += 1
        
        # 更新单目标跟踪器
        tracking_results = self.tracker.update(detections, self.frame_count)
        
        # 更新统计信息
        if tracking_results:
            print(f"第 {self.frame_count} 帧成功追踪到目标")
            self.statistics['successful_tracks'] += 1
            self.statistics['active_tracks'] = 1  # 单目标追踪，成功时活跃轨迹为1
            
            # 累计置信度
            for result in tracking_results:
                self.statistics['confidence_sum'] += result.confidence
        else:
            print(f"第 {self.frame_count} 帧未追踪到目标")
            self.statistics['lost_tracks'] += 1
            self.statistics['active_tracks'] = 0  # 单目标追踪，失败时活跃轨迹为0
        
        # 记录跟踪历史
        self.tracking_history.extend(tracking_results)
        
        # 保存图像文件（如果需要）
        if save_image and frame_name:
            image_dir = os.path.join(self.annotation_generator.output_dir, 'images')
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f"{frame_name}.jpg")
            cv2.imwrite(image_path, image)
            self.statistics['saved_images'] += 1
        
        # 保存标注文件（只有在明确要求时才保存）
        if save_annotation and frame_name:
            height, width = image.shape[:2]
            annotation_saved = self.annotation_generator.save_frame_annotation(
                frame_name, tracking_results, width, height
            )
            
            # 只有成功保存标注文件时才更新计数
            if annotation_saved:
                self.statistics['saved_annotations'] += 1
                self.statistics['saved_labels'] += 1  # 保持兼容性
        
        # 更新总帧数和结束帧
        self.statistics['total_frames'] = self.frame_count + 1
        self.statistics['tracking_end_frame'] = self.frame_count
        
        self.frame_count += 1
        return tracking_results
    
    def reset_tracker(self):
        """重置追踪器，可以重新开始追踪新目标"""
        self.tracker.reset()
        self.is_initialized = False
        self.frame_count = 0
        self.tracking_history = []
        print("✓ 追踪器已重置，可以重新初始化新目标")
    
    def get_tracking_statistics(self) -> Dict:
        """
        获取详细的单目标跟踪统计信息
        
        返回:
        - 完整的统计信息字典
        """
        # 更新最终统计信息
        final_stats = self.statistics.copy()
        
        # 计算平均置信度
        if final_stats['successful_tracks'] > 0:
            final_stats['average_confidence'] = final_stats['confidence_sum'] / final_stats['successful_tracks']
        else:
            final_stats['average_confidence'] = 0.0
        
        # 计算追踪成功率
        if final_stats['total_frames'] > 0:
            final_stats['tracking_success_rate'] = final_stats['successful_tracks'] / final_stats['total_frames']
        else:
            final_stats['tracking_success_rate'] = 0.0
        
        # 添加当前追踪状态
        final_stats['is_tracking'] = self.tracker.is_tracking if self.tracker else False
        
        return final_stats
    
    def print_tracking_statistics(self) -> str:
        """
        打印并返回格式化的追踪统计信息
        
        返回:
        - 格式化的统计信息字符串
        """
        stats = self.get_tracking_statistics()
        
        # 构建统计信息字符串
        stats_text = "处理完成！统计信息：\n"
        stats_text += "{\n"
        stats_text += f"  'total_frames': {stats['total_frames']},\n"
        stats_text += f"  'total_tracks': {stats['total_tracks']},\n"
        stats_text += f"  'active_tracks': {stats['active_tracks']},\n"
        stats_text += f"  'total_detections': {stats['total_detections']}\n"
        stats_text += "}\n"
        
        # 添加保存路径信息
        if stats['saved_images'] > 0:
            images_path = os.path.join(self.annotation_generator.output_dir, 'images')
            stats_text += f"共保存 {stats['saved_images']} 组完整原图到：{os.path.abspath(images_path)}\n"
        
        if stats['saved_annotations'] > 0:
            labels_path = os.path.join(self.annotation_generator.output_dir, 'labels')
            stats_text += f"共保存 {stats['saved_annotations']} 组YOLO标签到：{os.path.abspath(labels_path)}\n"
        
        # 打印统计信息
        print(stats_text)
        
        return stats_text
    
    def export_tracking_results(self, output_file: str):
        """
        导出完整的跟踪结果到JSON文件
        
        参数:
        - output_file: 输出文件路径
        """
        results = []
        for track in self.tracking_history:
            # 确保所有数值都转换为Python原生类型，避免numpy类型序列化问题
            results.append({
                "frame_id": int(track.frame_id),
                "track_id": int(track.track_id),
                "x1": float(track.x1),
                "y1": float(track.y1),
                "x2": float(track.x2),
                "y2": float(track.y2),
                "confidence": float(track.confidence),
                "class_id": int(track.class_id),
                "class_name": str(track.class_name)
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"跟踪结果已导出到: {output_file}")


# 测试和示例代码
if __name__ == "__main__":
    print("视频自动标注模块测试")
    print("=" * 50)
    
    try:
        # 示例：创建自动标注器实例（使用官方预训练模型）
        output_dir = "./test_annotations"
        
        print("正在创建视频自动标注器...")
        annotator = VideoAutoAnnotator(
            output_dir=output_dir,
            model_path=None,  # 使用官方预训练模型
            confidence_threshold=0.5,
            iou_threshold=0.3
        )
        print("✓ 自动标注器创建成功")
        print(f"✓ 检测器已加载，支持 {len(annotator.detector.class_names)} 个类别")
        
        # 显示一些主要的检测类别
        main_classes = annotator.detector.class_names[:15]
        print(f"✓ 主要检测类别: {', '.join(main_classes)}")
        
        # 测试基本功能
        print("\n测试基本数据结构...")
        
        # 测试DetectionBox
        test_detection = DetectionBox(100, 100, 200, 200, 0.9, 0, "person")
        print(f"✓ DetectionBox测试: {test_detection.class_name} ({test_detection.confidence:.2f})")
        
        # 测试TrackingBox
        test_tracking = TrackingBox(1, 100, 100, 200, 200, 0.9, 0, "person", 0)
        print(f"✓ TrackingBox测试: Track ID {test_tracking.track_id}")
        
        # 测试IoU计算
        box1 = (100, 100, 200, 200)
        box2 = (150, 150, 250, 250)
        iou = annotator.tracker.calculate_iou(box1, box2)
        print(f"✓ IoU计算测试: {iou:.3f}")
        
        # 测试单目标追踪器初始化
        test_manual_box = {
            'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200,
            'class_id': 0, 'class_name': 'person'
        }
        print(f"✓ 单目标追踪器测试: 目标类别 {test_manual_box['class_name']}")
        
        print("\n✓ 所有基本功能测试通过")
        
    except ImportError as e:
        print(f"✗ 导入错误: {e}")
        print("请确保已安装必要的依赖库:")
        print("  pip install ultralytics opencv-python numpy")
        
    except Exception as e:
        print(f"✗ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 50)
    print("模块加载完成，可以在CookLabel中调用相关功能")
    print("使用方法:")
    print("1. 创建VideoAutoAnnotator实例")
    print("2. 使用initialize_from_manual_annotation()初始化第一帧")
    print("3. 使用process_frame()处理后续帧")
    print("4. 使用get_tracking_statistics()获取统计信息")