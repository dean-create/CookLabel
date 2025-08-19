#!/usr/bin/env python
# -*- coding: utf8 -*-
import codecs
import os

from libs.constants import DEFAULT_ENCODING

TXT_EXT = '.txt'
ENCODE_METHOD = DEFAULT_ENCODING

class YOLOWriter:

    def __init__(self, folder_name, filename, img_size, database_src='Unknown', local_img_path=None):
        self.folder_name = folder_name
        self.filename = filename
        self.database_src = database_src
        self.img_size = img_size
        self.box_list = []
        self.local_img_path = local_img_path
        self.verified = False

    def add_bnd_box(self, x_min, y_min, x_max, y_max, name, difficult):
        bnd_box = {'xmin': x_min, 'ymin': y_min, 'xmax': x_max, 'ymax': y_max}
        bnd_box['name'] = name
        bnd_box['difficult'] = difficult
        self.box_list.append(bnd_box)

    def bnd_box_to_yolo_line(self, box, class_list=[]):
        x_min = box['xmin']
        x_max = box['xmax']
        y_min = box['ymin']
        y_max = box['ymax']

        x_center = float((x_min + x_max)) / 2 / self.img_size[1]
        y_center = float((y_min + y_max)) / 2 / self.img_size[0]

        w = float((x_max - x_min)) / self.img_size[1]
        h = float((y_max - y_min)) / self.img_size[0]

        # PR387
        box_name = box['name']
        if box_name not in class_list:
            class_list.append(box_name)

        class_index = class_list.index(box_name)

        return class_index, x_center, y_center, w, h

    def save(self, class_list=[], target_file=None):

        out_file = None  # Update yolo .txt
        out_class_file = None   # Update class list .txt

        if target_file is None:
            out_file = open(
            self.filename + TXT_EXT, 'w', encoding=ENCODE_METHOD)
            classes_file = os.path.join(os.path.dirname(os.path.abspath(self.filename)), "classes.txt")
            out_class_file = open(classes_file, 'w')

        else:
            out_file = codecs.open(target_file, 'w', encoding=ENCODE_METHOD)
            classes_file = os.path.join(os.path.dirname(os.path.abspath(target_file)), "classes.txt")
            out_class_file = open(classes_file, 'w')


        for box in self.box_list:
            class_index, x_center, y_center, w, h = self.bnd_box_to_yolo_line(box, class_list)
            # print (classIndex, x_center, y_center, w, h)
            out_file.write("%d %.6f %.6f %.6f %.6f\n" % (class_index, x_center, y_center, w, h))

        # print (classList)
        # print (out_class_file)
        for c in class_list:
            out_class_file.write(c+'\n')

        out_class_file.close()
        out_file.close()



class YoloReader:

    def __init__(self, file_path, image, class_list_path=None):
        # shapes type:
        # [labbel, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], color, color, difficult]
        self.shapes = []
        self.file_path = file_path

        if class_list_path is None:
            dir_path = os.path.dirname(os.path.realpath(self.file_path))
            self.class_list_path = os.path.join(dir_path, "classes.txt")
        else:
            self.class_list_path = class_list_path

        # print (file_path, self.class_list_path)

        # 安全地读取classes.txt文件
        self.classes = []
        try:
            if os.path.exists(self.class_list_path):
                with open(self.class_list_path, 'r', encoding='utf-8') as classes_file:
                    content = classes_file.read().strip()
                    if content:
                        self.classes = content.split('\n')
                        # 过滤空行
                        self.classes = [cls.strip() for cls in self.classes if cls.strip()]
            
            # 如果classes.txt不存在或为空，使用默认类别
            if not self.classes:
                print(f"警告: classes.txt文件不存在或为空，使用默认类别")
                self.classes = ['class_0']  # 默认类别
                
        except Exception as e:
            print(f"警告: 读取classes.txt文件失败: {e}，使用默认类别")
            self.classes = ['class_0']  # 默认类别

        # print (self.classes)

        img_size = [image.height(), image.width(),
                    1 if image.isGrayscale() else 3]

        self.img_size = img_size

        self.verified = False
        # try:
        self.parse_yolo_format()
        # except:
        #     pass

    def get_shapes(self):
        return self.shapes

    def add_shape(self, label, x_min, y_min, x_max, y_max, difficult):

        points = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        self.shapes.append((label, points, None, None, difficult))

    def yolo_line_to_shape(self, class_index, x_center, y_center, w, h):
        """
        将YOLO格式的标注转换为形状对象
        如果类别索引超出范围，返回None跳过该标注框
        """
        # 将类别索引转换为整数
        try:
            class_idx = int(class_index)
        except ValueError:
            print(f"警告: 无效的类别索引格式 '{class_index}'，跳过该标注框")
            return None
        
        # 检查类别索引是否在有效范围内
        if class_idx < 0 or class_idx >= len(self.classes):
            print(f"警告: 类别索引 {class_idx} 超出范围 [0, {len(self.classes)-1}]，跳过该标注框")
            print(f"提示: 请检查classes.txt文件是否包含所有需要的类别")
            return None
        
        # 获取类别标签
        label = self.classes[class_idx]

        # 计算边界框坐标
        x_min = max(float(x_center) - float(w) / 2, 0)
        x_max = min(float(x_center) + float(w) / 2, 1)
        y_min = max(float(y_center) - float(h) / 2, 0)
        y_max = min(float(y_center) + float(h) / 2, 1)

        # 转换为像素坐标
        x_min = round(self.img_size[1] * x_min)
        x_max = round(self.img_size[1] * x_max)
        y_min = round(self.img_size[0] * y_min)
        y_max = round(self.img_size[0] * y_max)

        return label, x_min, y_min, x_max, y_max

    def parse_yolo_format(self):
        """
        解析YOLO格式的标注文件
        跳过类别索引超出范围的标注框
        """
        bnd_box_file = open(self.file_path, 'r')
        valid_boxes = 0  # 有效标注框计数
        skipped_boxes = 0  # 跳过的标注框计数
        
        for line_num, bndBox in enumerate(bnd_box_file, 1):
            try:
                # 解析YOLO格式的一行数据
                parts = bndBox.strip().split(' ')
                if len(parts) != 5:
                    print(f"警告: 第{line_num}行格式错误，应包含5个数值，实际包含{len(parts)}个")
                    skipped_boxes += 1
                    continue
                
                class_index, x_center, y_center, w, h = parts
                
                # 转换为形状对象
                result = self.yolo_line_to_shape(class_index, x_center, y_center, w, h)
                
                # 如果返回None，说明类别索引超出范围，跳过该标注框
                if result is None:
                    skipped_boxes += 1
                    continue
                
                label, x_min, y_min, x_max, y_max = result
                
                # 添加有效的标注框
                self.add_shape(label, x_min, y_min, x_max, y_max, False)
                valid_boxes += 1
                
            except Exception as e:
                print(f"警告: 第{line_num}行解析失败: {e}")
                skipped_boxes += 1
                continue
        
        bnd_box_file.close()
        
        # 输出解析结果统计
        if skipped_boxes > 0:
            print(f"标注解析完成: 成功加载 {valid_boxes} 个标注框，跳过 {skipped_boxes} 个无效标注框")
        elif valid_boxes > 0:
            print(f"标注解析完成: 成功加载 {valid_boxes} 个标注框")
