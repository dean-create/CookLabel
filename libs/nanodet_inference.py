#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NanoDet推理模块 - 基于成功的独立版本实现
使用训练好的NanoDet模型进行目标检测推理
功能: 提供NanoDet模型的推理接口，完全基于auto_annotation_standalone.py的成功实现
"""

import os
import cv2
import numpy as np
from pathlib import Path
from types import SimpleNamespace

# 延迟导入torch和nanodet模块以避免循环导入问题
# 这些模块将在实际使用时才导入


class NanoDetConfig:
    """
    NanoDet配置类 - 硬编码所有必要的配置参数
    
    这个类运用了面向对象编程的封装概念：
    1. 将所有配置参数封装在一个类中
    2. 提供清晰的结构和易于维护的配置管理
    3. 消除了对外部配置文件的依赖
    """
    
    def __init__(self):
        """初始化配置参数 - 基于nanodet-plus-m-1.5x_416-glass-detect.yml"""
        
        # 类别信息配置
        self.class_names = ['object']  # 检测的类别名称列表
        
        # 设备配置
        self.device = 'cpu'
        self.use_gpu = False  # 添加use_gpu属性
        self.confidence_threshold = 0.35
        
        # 模型架构配置
        self.model = SimpleNamespace()
        
        # 模型主体架构配置（使用字典格式，因为NanoDet框架需要pop操作）
        self.model.arch = {
            'name': 'NanoDetPlus',
            'detach_epoch': 10,
            
            # Backbone网络配置（特征提取网络）
            'backbone': {
                'name': 'ShuffleNetV2',
                'model_size': '1.5x',
                'out_stages': [2, 3, 4],  # 输出的特征层
                'activation': 'LeakyReLU'
            },
            
            # FPN网络配置（特征金字塔网络）
            'fpn': {
                'name': 'GhostPAN',
                'in_channels': [176, 352, 704],  # 输入通道数
                'out_channels': 128,  # 输出通道数
                'kernel_size': 5,
                'num_extra_level': 1,
                'use_depthwise': True,
                'activation': 'LeakyReLU'
            },
            
            # 检测头配置
             'head': {
                 'name': 'NanoDetPlusHead',
                 'num_classes': 1,  # 类别数量（不包括背景）
                 'input_channel': 128,
                 'feat_channels': 128,
                 'stacked_convs': 2,
                 'kernel_size': 5,
                 'strides': [8, 16, 32, 64],  # 多尺度检测的步长
                 'activation': 'LeakyReLU',
                 'reg_max': 7,  # 回归最大值
                 
                 # 归一化配置
                 'norm_cfg': {
                     'type': 'BN'
                 },
                 
                 # 损失函数配置（推理时不需要，但模型构建时可能需要）
                  # 注意：这里需要使用SimpleNamespace，因为NanoDetPlusHead期望对象属性访问
                  'loss': SimpleNamespace(
                      loss_qfl=SimpleNamespace(
                          name='QualityFocalLoss',
                          use_sigmoid=True,
                          beta=2.0,
                          loss_weight=1.0
                      ),
                      loss_dfl=SimpleNamespace(
                          name='DistributionFocalLoss',
                          loss_weight=0.25
                      ),
                      loss_bbox=SimpleNamespace(
                          name='GIoULoss',
                          loss_weight=2.0
                      )
                  )
             },
             
             # 辅助检测头配置（NanoDetPlus需要的必需参数）
             'aux_head': {
                 'name': 'SimpleConvHead',
                 'num_classes': 1,
                 'input_channel': 128,
                 'feat_channels': 128,
                 'stacked_convs': 4,
                 'strides': [8, 16, 32, 64],
                 'activation': 'LeakyReLU',
                 'norm_cfg': {
                     'type': 'BN'
                 }
             }
        }
        
        # 数据处理配置
        self.data = SimpleNamespace()
        self.data.val = SimpleNamespace()
        self.data.val.input_size = [416, 416]  # 输入图像尺寸 [width, height]
        self.data.val.keep_ratio = True  # 保持宽高比
        
        # 数据预处理管道配置（推理时只需要归一化）
        self.data.val.pipeline = SimpleNamespace()
        self.data.val.pipeline.normalize = [[0, 0, 0], [1, 1, 1]]  # 归一化参数 [mean, std]


class NanoDetInference:
    """
    NanoDet推理类 - 基于成功的独立版本实现
    
    这个类运用了面向对象编程的核心概念：
    1. 封装：将模型加载、推理等功能封装在一个类中
    2. 属性：存储模型、配置、设备等状态信息
    3. 方法：提供不同功能的接口，如推理等
    4. 依赖注入：通过构造函数注入必要的依赖
    """
    
    def __init__(self, model_path, device="cpu", confidence_threshold=0.35, config=None):
        """
        初始化NanoDet推理器 - 基于独立版本，支持config参数
        
        参数说明：
        - model_path: 训练好的模型权重文件路径  
        - device: 推理设备，默认CPU（兼容性最好）
        - confidence_threshold: 置信度阈值，默认0.35（适合标注任务）
        - config: 可选的配置对象，如果提供则使用其中的参数
        
        注意：支持两种初始化方式：
        1. 直接传参：NanoDetInference(model_path, device, confidence_threshold)
        2. 使用config：NanoDetInference(model_path, config=config_obj)
        """
        # 延迟导入torch和nanodet模块
        try:
            import torch
            from nanodet.data.batch_process import stack_batch_img
            from nanodet.data.collate import naive_collate
            from nanodet.data.transform import Pipeline
            from nanodet.model.arch import build_model
            from nanodet.util import Logger, load_model_weight
            
            # 将导入的模块保存为实例变量
            self.torch = torch
            self.stack_batch_img = stack_batch_img
            self.naive_collate = naive_collate
            self.Pipeline = Pipeline
            self.build_model = build_model
            self.Logger = Logger
            self.load_model_weight = load_model_weight
            
        except ImportError as e:
            raise ImportError(f"无法导入必要的模块: {e}")
        
        # 如果提供了config对象，优先使用config中的参数
        if config is not None:
            self.device = getattr(config, 'device', device)
            self.confidence_threshold = getattr(config, 'confidence_threshold', confidence_threshold)
        else:
            self.device = device
            self.confidence_threshold = confidence_threshold
        
        # 创建内置配置对象
        self.cfg = NanoDetConfig()
        
        # 创建日志器（用于模型加载时的信息输出）
        # 添加异常处理，防止日志系统初始化失败影响核心功能
        try:
            self.logger = self.Logger(0, use_tensorboard=False)
        except Exception as e:
            print(f"警告: 日志系统初始化失败 - {str(e)}")
            print("使用安全的日志替代对象，不影响核心功能")
            self.logger = self._create_safe_logger()
        
        # 构建并加载模型
        self.model = self._load_model(model_path)
        
        # 创建数据预处理管道
        self.pipeline = self._create_pipeline()
        
        # 获取类别名称列表
        self.class_names = self.cfg.class_names
        
        # 记录模型加载成功信息到日志文件
        try:
            self.logger.log(f"NanoDet模型加载成功！")
            self.logger.log(f"检测类别: {self.class_names}")
            self.logger.log(f"置信度阈值: {self.confidence_threshold}")
            self.logger.log(f"推理设备: {self.device}")
            self.logger.log(f"输入尺寸: {self.cfg.data.val.input_size}")
        except Exception as e:
            print(f"日志写入失败: {e}")
        
        print(f"NanoDet模型加载成功！")
        print(f"检测类别: {self.class_names}")
        print(f"置信度阈值: {self.confidence_threshold}")
        print(f"推理设备: {self.device}")
        print(f"输入尺寸: {self.cfg.data.val.input_size}")
    
    def _create_safe_logger(self):
        """
        创建安全的日志替代对象
        
        当原始日志系统初始化失败时，提供一个安全的替代方案
        这个方法体现了面向对象编程中的封装和容错设计原则：
        1. 封装：将日志功能封装在一个内部类中
        2. 容错：提供备用的日志实现，确保程序不会因日志问题而崩溃
        3. 接口一致性：保持与原始Logger相同的方法接口
        
        返回:
            SafeLogger: 安全的日志对象
        """
        class SafeLogger:
            """
            安全日志类 - 当原始日志系统失败时的备用方案
            
            这个内部类提供了基本的日志功能，通过print语句输出信息
            确保即使日志系统受损，程序也能继续运行
            """
            def __init__(self):
                self.name = "SafeLogger"
                
            def info(self, message):
                """输出信息级别日志"""
                print(f"[INFO] {message}")
                
            def warning(self, message):
                """输出警告级别日志"""
                print(f"[WARNING] {message}")
                
            def error(self, message):
                """输出错误级别日志"""
                print(f"[ERROR] {message}")
                
            def debug(self, message):
                """输出调试级别日志"""
                print(f"[DEBUG] {message}")
                
            def __call__(self, *args, **kwargs):
                """使对象可调用，兼容某些日志使用方式"""
                if args:
                    self.info(str(args[0]))
                    
        return SafeLogger()
    
    def _create_pipeline(self):
        """
        创建数据预处理管道
        
        基于YAML配置文件的格式，pipeline只包含normalize参数
        这与独立版本demo-get-roi-glass.py中的实现保持一致
        """
        # 构建预处理管道配置 - 与YAML文件格式保持一致
        pipeline_config = {
            'normalize': self.cfg.data.val.pipeline.normalize
        }
        
        return self.Pipeline(pipeline_config, True)
    
    def _namespace_to_dict(self, namespace_obj):
        """
        私有方法：将SimpleNamespace对象递归转换为字典
        
        这个方法解决了NanoDet build_model函数期望字典格式配置的问题：
        - 递归处理嵌套的SimpleNamespace对象
        - 保持原有的数据结构和层次关系
        - 确保与NanoDet框架的兼容性
        
        参数：
        - namespace_obj: SimpleNamespace对象或其他类型的对象
        
        返回值：转换后的字典或原始对象
        """
        if isinstance(namespace_obj, SimpleNamespace):
            # 如果是SimpleNamespace对象，转换为字典并递归处理每个值
            result = {}
            for key, value in namespace_obj.__dict__.items():
                result[key] = self._namespace_to_dict(value)  # 递归处理嵌套对象
            return result
        elif isinstance(namespace_obj, list):
            # 如果是列表，递归处理列表中的每个元素
            return [self._namespace_to_dict(item) for item in namespace_obj]
        else:
            # 其他类型直接返回
            return namespace_obj
    
    def _load_model(self, model_path):
        """
        私有方法：加载训练好的模型
        
        这是一个私有方法（以_开头），体现了封装的概念：
        - 外部不需要直接调用此方法
        - 内部实现细节被隐藏
        - 提高了代码的安全性和可维护性
        
        返回值：加载好的模型对象
        """
        # 直接使用SimpleNamespace配置对象（NanoDet框架要求）
        # NanoDet的build_model函数期望接收具有属性访问的对象，而不是字典
        model = self.build_model(self.cfg.model)
        
        # 加载预训练权重
        checkpoint = self.torch.load(model_path, map_location=lambda storage, loc: storage)
        self.load_model_weight(model, checkpoint, self.logger)
        
        # 将模型移动到指定设备并设置为评估模式
        model = model.to(self.device).eval()
        
        return model
    
    def infer_single_image(self, image_path):
        """
        对单张图片进行推理检测
        
        参数：
        - image_path: 图片文件路径
        
        返回值：
        - detections: 检测结果列表，每个元素包含[类别ID, x1, y1, x2, y2, 置信度]
        - image_info: 图片信息字典，包含宽度、高度等
        
        基于独立版本demo-get-roi-glass.py的成功实现
        """
        try:
            # 读取图片 - 使用支持中文路径的方法
            try:
                # 方法1: 使用numpy和cv2.imdecode处理中文路径
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                image_array = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                if image is None:
                    raise ValueError(f"无法解码图片: {image_path}")
            except Exception as e:
                # 如果上述方法失败，尝试使用PIL转换
                try:
                    from PIL import Image
                    pil_image = Image.open(image_path)
                    # 转换为RGB格式（PIL默认RGB，OpenCV默认BGR）
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    # 转换为numpy数组并调整颜色通道顺序（RGB -> BGR）
                    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                except Exception as e2:
                    raise ValueError(f"无法读取图片: {image_path}，错误: {str(e2)}")
            
            # 获取图片基本信息 - 按照独立版本的格式
            height, width = image.shape[:2]
            img_info = {
                "id": 0,
                "file_name": os.path.basename(image_path),
                "height": height,
                "width": width,
            }
            
            # 构建meta数据 - 完全按照独立版本的格式
            meta = dict(img_info=img_info, raw_img=image, img=image)
            
            # 数据预处理 - 按照独立版本的调用方式
            meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
            
            # 转换为张量格式 - 使用延迟导入的模块
            meta["img"] = self.torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
            meta = self.naive_collate([meta])
            meta["img"] = self.stack_batch_img(meta["img"], divisible=32)
            
            print(f"开始推理图片: {os.path.basename(image_path)}")
            print(f"原始尺寸: {width}x{height}")
            print(f"预处理后张量形状: {meta['img'].shape}")
            
            # 模型推理 - 按照独立版本的调用方式
            with self.torch.no_grad():
                results = self.model.inference(meta)
            
            print(f"模型输出结果数量: {len(results)}")
            if len(results) > 0:
                print(f"第一个结果形状: {len(results[0]) if results[0] is not None else 'None'}")
            
            # 解析检测结果 - 使用原始图片信息
            image_info = {
                "file_name": os.path.basename(image_path),
                "width": width,
                "height": height,
            }
            detections = self._parse_detections(results)
            
            print(f"检测完成，找到 {len(detections)} 个目标")
            
            return detections, image_info
            
        except Exception as e:
            print(f"推理过程中发生错误: {str(e)}")
            import traceback
            print(f"详细错误信息:\n{traceback.format_exc()}")
            return [], {"file_name": os.path.basename(image_path), "width": 0, "height": 0}
    
    def _parse_detections(self, raw_results):
        """
        解析模型的原始输出结果
        
        参数：
        - raw_results: 模型的原始输出，格式为[{label: [bbox1, bbox2, ...]}, ...]
        
        返回值：
        - detections: 过滤后的检测结果列表
        
        基于独立版本demo-get-roi-glass.py的结果处理方式
        """
        detections = []
        
        if raw_results is None or len(raw_results) == 0:
            print("模型输出为空或无检测结果")
            return detections
        
        print(f"解析检测结果，原始结果数量: {len(raw_results)}")
        
        # 按照独立版本的处理方式：res[0][label]
        if len(raw_results) > 0 and isinstance(raw_results[0], dict):
            result_dict = raw_results[0]
            print(f"检测到的类别数量: {len(result_dict)}")
            
            # 遍历每个类别的检测结果
            for label, bboxes in result_dict.items():
                print(f"  类别 {label}: {len(bboxes)} 个检测框")
                
                # 遍历该类别的所有检测框
                for i, bbox in enumerate(bboxes):
                    # 安全地检查bbox类型和长度
                    try:
                        # 首先检查bbox是否为None
                        if bbox is None:
                            print(f"    检测框 {i+1} 为None，跳过")
                            continue
                            
                        # 检查bbox是否为可索引对象（列表、元组、numpy数组等）
                        if not hasattr(bbox, '__len__') or not hasattr(bbox, '__getitem__'):
                            print(f"    检测框 {i+1} 不是可索引对象: {type(bbox)}, 值: {bbox}")
                            continue
                        
                        # 检查bbox长度是否足够
                        try:
                            bbox_len = len(bbox)
                        except TypeError:
                            print(f"    检测框 {i+1} 无法获取长度: {type(bbox)}, 值: {bbox}")
                            continue
                            
                        if bbox_len >= 5:  # [x0, y0, x1, y1, score]
                            try:
                                x0, y0, x1, y1, score = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]
                            except (IndexError, TypeError) as slice_error:
                                print(f"    检测框 {i+1} 索引访问失败: {slice_error}, bbox类型: {type(bbox)}, 值: {bbox}")
                                continue
                            
                            print(f"    检测框 {i+1}: bbox=[{x0:.2f}, {y0:.2f}, {x1:.2f}, {y1:.2f}], score={score:.4f}")
                        else:
                            print(f"    检测框 {i+1} 格式不正确，长度不足: {bbox_len}, 内容: {bbox}")
                            continue
                    except Exception as e:
                        print(f"    处理检测框 {i+1} 时发生错误: {e}, bbox类型: {type(bbox)}, 值: {bbox}")
                        continue
                    
                    # 置信度过滤 - 只有在成功解析bbox后才进行
                    if score >= self.confidence_threshold:
                        # 安全地处理label类型转换
                        try:
                            label_id = int(label) if not isinstance(label, int) else label
                            # 增加对class_names的安全检查
                            if not isinstance(self.class_names, (list, tuple)):
                                print(f"      class_names类型错误: {type(self.class_names)}，应为列表或元组")
                                class_name = f'class_{label_id}'
                            elif not isinstance(label_id, int):
                                print(f"      label_id类型错误: {type(label_id)}，应为整数")
                                class_name = f'class_{label_id}'
                            elif label_id < 0 or label_id >= len(self.class_names):
                                print(f"      label_id超出范围: {label_id}，class_names长度: {len(self.class_names)}")
                                class_name = f'class_{label_id}'
                            else:
                                class_name = self.class_names[label_id]
                        except (ValueError, TypeError, IndexError) as e:
                            print(f"      标签处理错误: {e}, label={label}, type={type(label)}")
                            label_id = 0
                            class_name = 'unknown'
                        
                        detections.append({
                            'class_id': label_id,
                            'class_name': class_name,
                            'confidence': float(score),
                            'bbox': [float(x0), float(y0), float(x1), float(y1)]
                        })
                        print(f"      通过置信度过滤 (>= {self.confidence_threshold})")
                    else:
                        print(f"      置信度过低 ({score:.4f} < {self.confidence_threshold})")
        else:
            print(f"结果格式不符合预期: {type(raw_results[0]) if len(raw_results) > 0 else 'empty'}")
        
        print(f"最终检测结果数量: {len(detections)}")
        return detections
    
    def process_images(self, input_dir, output_dir, output_format="YOLO", progress_callback=None):
        """
        批量处理图像目录中的所有图像
        
        这个方法实现了批量图像处理的核心逻辑：
        1. 遍历输入目录中的所有图像文件
        2. 对每个图像进行推理检测
        3. 将检测结果保存为指定格式的标注文件
        4. 支持进度回调函数，用于GUI进度显示
        
        参数：
        - input_dir: 输入图像目录路径
        - output_dir: 输出标注文件目录路径
        - output_format: 输出格式，支持"YOLO"、"XML"等
        - progress_callback: 进度回调函数，接收(current, total)参数
        
        返回值：
        - 处理成功的图像数量
        
        这个方法运用了以下Python概念：
        1. 文件系统操作：使用pathlib进行路径处理
        2. 异常处理：try-except确保单个文件错误不影响整体处理
        3. 回调函数：支持外部进度监控
        4. 字符串处理：文件扩展名判断和路径操作
        """
        import os
        from pathlib import Path
        
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有图像文件
        input_path = Path(input_dir)
        image_files = []
        
        for file_path in input_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)
        
        total_files = len(image_files)
        processed_count = 0
        
        # 记录批量处理开始信息到日志文件
        try:
            self.logger.log(f"开始批量处理: 找到 {total_files} 个图像文件")
            self.logger.log(f"输出目录: {output_dir}")
            self.logger.log(f"输出格式: {output_format}")
        except Exception as e:
            print(f"日志写入失败: {e}")
        
        print(f"找到 {total_files} 个图像文件")
        print(f"输出目录: {output_dir}")
        print(f"输出格式: {output_format}")
        
        # 处理每个图像文件
        for i, image_file in enumerate(image_files):
            try:
                print(f"\n处理图像 {i+1}/{total_files}: {image_file.name}")
                
                # 进行推理检测
                detections, image_info = self.infer_single_image(str(image_file))
                
                # 生成输出文件路径
                output_file = Path(output_dir) / f"{image_file.stem}.txt"
                
                # 保存检测结果
                if output_format.upper() == "YOLO":
                    self._save_yolo_format(detections, image_info, str(output_file))
                elif output_format.upper() == "XML":
                    # XML格式保存（如果需要的话）
                    xml_file = Path(output_dir) / f"{image_file.stem}.xml"
                    self._save_xml_format(detections, image_info, str(xml_file), str(image_file))
                
                processed_count += 1
                
                # 记录单个图像处理结果到日志文件
                try:
                    self.logger.log(f"处理完成 {image_file.name}: 检测到 {len(detections)} 个目标")
                except Exception as e:
                    print(f"日志写入失败: {e}")
                
                print(f"处理完成，检测到 {len(detections)} 个目标")
                
                # 调用进度回调函数 - 计算百分比进度
                if progress_callback:
                    progress_percent = int(((i + 1) / total_files) * 100)
                    progress_callback(progress_percent)
                    
            except Exception as e:
                error_msg = f"处理图像 {image_file.name} 时发生错误: {str(e)}"
                # 记录错误到日志文件
                try:
                    self.logger.log(error_msg)
                except Exception as log_e:
                    print(f"日志写入失败: {log_e}")
                print(error_msg)
                continue
        
        # 记录批量处理完成信息到日志文件
        try:
            self.logger.log(f"批量处理完成！成功处理: {processed_count}/{total_files} 个图像")
        except Exception as e:
            print(f"日志写入失败: {e}")
        
        print(f"\n批量处理完成！成功处理 {processed_count}/{total_files} 个图像")
        
        # 记录最终统计信息到日志文件
        try:
            self.logger.log(f"处理统计 - 总计: {total_files}, 成功: {processed_count}, 失败: {total_files - processed_count}")
        except Exception as e:
            print(f"日志写入失败: {e}")
        
        # 返回统计信息字典，与GUI期望的格式匹配
        statistics = {
            'total': total_files,
            'processed': processed_count,
            'failed': total_files - processed_count
        }
        return statistics
    
    def _save_yolo_format(self, detections, image_info, output_file):
        """
        保存检测结果为YOLO格式
        
        YOLO格式说明：
        - 每行一个检测结果
        - 格式：class_id center_x center_y width height
        - 所有坐标都是相对于图像尺寸的归一化值（0-1之间）
        
        参数：
        - detections: 检测结果列表
        - image_info: 图像信息字典
        - output_file: 输出文件路径
        """
        with open(output_file, 'w') as f:
            for detection in detections:
                # 获取边界框坐标
                x1, y1, x2, y2 = detection['bbox']
                
                # 计算中心点和宽高
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                width = x2 - x1
                height = y2 - y1
                
                # 归一化到0-1范围
                img_width = image_info['width']
                img_height = image_info['height']
                
                norm_center_x = center_x / img_width
                norm_center_y = center_y / img_height
                norm_width = width / img_width
                norm_height = height / img_height
                
                # 写入YOLO格式
                class_id = detection['class_id']
                f.write(f"{class_id} {norm_center_x:.6f} {norm_center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")
    
    def _save_xml_format(self, detections, image_info, output_file, image_file):
        """
        保存检测结果为XML格式（Pascal VOC格式）
        
        参数：
        - detections: 检测结果列表
        - image_info: 图像信息字典
        - output_file: 输出XML文件路径
        - image_file: 原始图像文件路径
        """
        import xml.etree.ElementTree as ET
        from pathlib import Path
        
        # 创建XML根元素
        annotation = ET.Element('annotation')
        
        # 添加文件信息
        folder = ET.SubElement(annotation, 'folder')
        folder.text = Path(image_file).parent.name
        
        filename = ET.SubElement(annotation, 'filename')
        filename.text = Path(image_file).name
        
        # 添加图像尺寸信息
        size = ET.SubElement(annotation, 'size')
        width = ET.SubElement(size, 'width')
        width.text = str(image_info['width'])
        height = ET.SubElement(size, 'height')
        height.text = str(image_info['height'])
        depth = ET.SubElement(size, 'depth')
        depth.text = '3'
        
        # 添加检测结果
        for detection in detections:
            obj = ET.SubElement(annotation, 'object')
            
            name = ET.SubElement(obj, 'name')
            name.text = detection['class_name']
            
            bndbox = ET.SubElement(obj, 'bndbox')
            xmin = ET.SubElement(bndbox, 'xmin')
            xmin.text = str(int(detection['bbox'][0]))
            ymin = ET.SubElement(bndbox, 'ymin')
            ymin.text = str(int(detection['bbox'][1]))
            xmax = ET.SubElement(bndbox, 'xmax')
            xmax.text = str(int(detection['bbox'][2]))
            ymax = ET.SubElement(bndbox, 'ymax')
            ymax.text = str(int(detection['bbox'][3]))
        
        # 保存XML文件
        tree = ET.ElementTree(annotation)
        tree.write(output_file, encoding='utf-8', xml_declaration=True)


# 为了保持向后兼容性，提供一个工厂函数
def create_nanodet_inference(model_path, device="cpu", confidence_threshold=0.35):
    """
    创建NanoDet推理器实例的工厂函数
    
    参数：
    - model_path: 模型文件路径
    - device: 推理设备
    - confidence_threshold: 置信度阈值
    
    返回值：
    - NanoDetInference实例
    """
    return NanoDetInference(model_path, device, confidence_threshold)


if __name__ == "__main__":
    # 测试代码
    print("NanoDet推理模块 - 基于成功的独立版本实现")
    print("使用方法:")
    print("  from libs.nanodet_inference import NanoDetInference")
    print("  inference = NanoDetInference('model.pth', 'cpu', 0.35)")
    print("  detections, image_info = inference.infer_single_image('image.jpg')")