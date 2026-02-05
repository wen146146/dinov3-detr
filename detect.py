# !/usr/bin/env python3
"""
改进版DINOv3+DETR单图片检测脚本
降低识别严格性，提高检测率
"""

import torch
import torch.nn as nn
from PIL import Image, ImageDraw
import numpy as np
import os

# 导入模型
from mymodels.dinov3_detr import DINOv3DETR


class  SimpleDetector: #11111111111
    def __init__(self):
        # 选择设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # ✅ 修改类别数量（根据你的训练设置）
        self.num_classes = 10  # 修改为你的实际类别数

        # ✅ 修改模型权重路径
        self.model_path = "your_model.pth"  # 修改为你的模型文件路径

        # 加载模型
        self.load_model()

    def load_model(self):
        """加载模型"""
        print("加载模型...")

        # 创建模型
        self.model = DINOv3DETR(num_classes=self.num_classes,)

        # 加载训练好的检测头权重
        if os.path.exists(self.model_path):
            print(f"加载权重: {self.model_path}")
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.detr_head.load_state_dict(state_dict)
            print("✅ 检测头权重加载完成")
        else:
            # 尝试寻找其他可能的模型文件
            print(f"⚠️ 警告: 权重文件不存在 {self.model_path}")
            pth_files = [f for f in os.listdir('.') if f.endswith('.pth')]
            if pth_files:
                print(f"找到其他pth文件: {pth_files}")
                # 尝试加载第一个找到的pth文件
                self.model_path = pth_files[0]
                print(f"尝试加载: {self.model_path}")
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.detr_head.load_state_dict(state_dict)
            else:
                print("❌ 没有找到任何模型文件")
                print("将使用随机初始化的模型")

        # 移到设备并设置为评估模式
        self.model.to(self.device)
        self.model.eval()
        print("✅ 模型加载完成")

    def preprocess_image(self, image_path, img_size=224):
        """预处理图像"""
        print(f"处理图像: {image_path}")

        # 打开图像
        image = Image.open(image_path).convert('RGB')
        orig_width, orig_height = image.size
        print(f"原始尺寸: {orig_width} x {orig_height}")

        # 调整大小
        img_resized = image.resize((img_size, img_size), Image.BILINEAR)

        # 转换为numpy数组并归一化
        img_np = np.array(img_resized) / 255.0

        # 转换为PyTorch张量 [H, W, C] -> [C, H, W]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()

        # 添加批次维度 [C, H, W] -> [1, C, H, W]
        img_tensor = img_tensor.unsqueeze(0)

        return img_tensor, image, (orig_width, orig_height)

    def predict(self, image_tensor, confidence_threshold=0.1):  # ✅ 降低阈值到0.1
        """执行预测"""
        print("执行推理...")

        # 将图像移到设备
        image_tensor = image_tensor.to(self.device)

        # 前向传播
        with torch.no_grad():
            pred_logits, pred_boxes = self.model(image_tensor)

        # 处理预测结果
        pred_logits = pred_logits[0]  # [100, num_classes+1]
        pred_boxes = pred_boxes[0]  # [100, 4]

        # 获取类别概率
        pred_probs = torch.softmax(pred_logits, dim=-1)

        # 获取每个预测的最高置信度和类别
        max_probs, max_indices = torch.max(pred_probs, dim=-1)

        # ✅ 添加调试信息：显示置信度分布
        print(f"\n置信度统计:")
        print(f"  最高置信度: {max_probs.max().item():.4f}")
        print(f"  平均置信度: {max_probs.mean().item():.4f}")
        print(f"  置信度 > 0.1 的数量: {(max_probs > 0.1).sum().item()}")
        print(f"  置信度 > 0.3 的数量: {(max_probs > 0.3).sum().item()}")
        print(f"  置信度 > 0.5 的数量: {(max_probs > 0.5).sum().item()}")

        # ✅ 显示前10个预测的详细信息
        print(f"\n前10个预测详情:")
        for i in range(min(10, len(max_probs))):
            class_id = max_indices[i].item()
            confidence = max_probs[i].item()
            is_target = class_id < self.num_classes
            status = "✓ 目标" if is_target else "✗ 背景"
            print(f"  查询{i:2d}: {status} (类{class_id}), 置信度={confidence:.4f}")

        # ✅ 修改检测逻辑：更宽松的条件
        detections = []
        for i in range(len(max_probs)):
            confidence = max_probs[i].item()
            class_id = max_indices[i].item()

            # ✅ 修改1: 降低置信度阈值 (0.5 -> 0.1)
            # ✅ 修改2: 即使是背景类，如果置信度很高也考虑
            if confidence >= confidence_threshold:
                bbox = pred_boxes[i].cpu().numpy().tolist()  # [cx, cy, w, h]

                # 如果是目标类别
                if class_id < self.num_classes:
                    detections.append({
                        'class_id': class_id,
                        'confidence': confidence,
                        'bbox_cxcywh': bbox,
                        'is_target': True
                    })
                # ✅ 修改3: 即使被分类为背景，但置信度很高，也显示（用不同颜色）
                elif confidence > 0.7:  # 背景类但非常确信
                    detections.append({
                        'class_id': self.num_classes,  # 标记为背景类
                        'confidence': confidence,
                        'bbox_cxcywh': bbox,
                        'is_target': False
                    })

        print(f"\n检测到 {len(detections)} 个目标 (阈值={confidence_threshold})")
        return detections

    def draw_boxes(self, image, detections, orig_size, img_size=224):
        """在图像上绘制边界框"""
        draw = ImageDraw.Draw(image)
        orig_width, orig_height = orig_size

        # 颜色列表：目标用彩色，背景用灰色
        target_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                         (255, 255, 0), (255, 0, 255)]
        bg_color = (128, 128, 128)  # 背景框用灰色

        for i, det in enumerate(detections):
            # 获取边界框坐标
            cx, cy, w, h = det['bbox_cxcywh']

            # 转换到像素坐标
            cx = cx * img_size
            cy = cy * img_size
            w = w * img_size
            h = h * img_size

            # 转换为xyxy格式
            x_min = cx - w / 2
            y_min = cy - h / 2
            x_max = cx + w / 2
            y_max = cy + h / 2

            # 缩放到原始图像尺寸
            scale_x = orig_width / img_size
            scale_y = orig_height / img_size

            x_min = int(x_min * scale_x)
            y_min = int(y_min * scale_y)
            x_max = int(x_max * scale_x)
            y_max = int(y_max * scale_y)

            # 确保在图像范围内
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(orig_width, x_max)
            y_max = min(orig_height, y_max)

            # 选择颜色：目标用彩色，背景用灰色
            if det.get('is_target', True):
                color = target_colors[i % len(target_colors)]
                label_prefix = "目标"
            else:
                color = bg_color
                label_prefix = "背景"

            # 绘制边界框（目标用实线，背景用虚线）
            if det.get('is_target', True):
                draw.rectangle([x_min, y_min, x_max, y_max],
                               outline=color, width=3)
            else:
                # 背景用虚线（通过绘制多个小线段实现）
                dash_length = 5
                # 上边
                for dx in range(x_min, x_max, dash_length * 2):
                    draw.line([dx, y_min, min(dx + dash_length, x_max), y_min],
                              fill=color, width=2)
                # 下边
                for dx in range(x_min, x_max, dash_length * 2):
                    draw.line([dx, y_max, min(dx + dash_length, x_max), y_max],
                              fill=color, width=2)
                # 左边
                for dy in range(y_min, y_max, dash_length * 2):
                    draw.line([x_min, dy, x_min, min(dy + dash_length, y_max)],
                              fill=color, width=2)
                # 右边
                for dy in range(y_min, y_max, dash_length * 2):
                    draw.line([x_max, dy, x_max, min(dy + dash_length, y_max)],
                              fill=color, width=2)

            # 绘制标签
            label = f"{label_prefix}:{det['confidence']:.2f}"
            draw.text((x_min + 5, y_min + 5), label, fill=color)

        return image

    def draw_no_detection(self, image):
        """绘制'未检测到目标'的提示"""
        draw = ImageDraw.Draw(image)
        width, height = image.size

        # 绘制提示文字
        text = "⚠️ 未检测到高置信度目标"
        # 简单文本（如果没有字体）
        draw.text((10, 10), text, fill=(255, 0, 0))

        # 绘制建议
        suggestion = "尝试: 1.降低阈值 2.检查模型 3.使用训练集图片"
        draw.text((10, 30), suggestion, fill=(255, 0, 0))

        return image

    def detect_image(self, image_path, output_path=None, confidence_threshold=0.1):
        """检测单张图像"""
        print("=" * 60)
        print(f"开始检测: {image_path}")
        print("=" * 60)

        if not os.path.exists(image_path):
            print(f"❌ 图像不存在: {image_path}")
            return None

        # 1. 预处理
        img_tensor, original_image, orig_size = self.preprocess_image(image_path)

        # 2. 预测（使用更低的阈值）
        detections = self.predict(img_tensor, confidence_threshold)

        # 3. 确定输出路径
        if output_path is None:
            base_name = os.path.splitext(image_path)[0]
            output_path = f"{base_name}_detected.jpg"

        # ✅ 修改：无论如何都保存图片
        if not detections:
            print("\n⚠️ 未检测到高置信度目标")
            print("尝试降低阈值或使用训练集中的图片测试")

            # 即使没有检测到，也保存原图并添加提示
            result_image = self.draw_no_detection(original_image.copy())
        else:
            # 有检测结果：绘制边界框
            result_image = self.draw_boxes(original_image.copy(), detections, orig_size)

            # 显示检测结果统计
            print(f"\n📊 检测结果统计:")
            target_count = sum(1 for d in detections if d.get('is_target', True))
            bg_count = len(detections) - target_count
            print(f"  目标检测数: {target_count}")
            print(f"  背景高置信数: {bg_count}")

            print(f"\n🔍 检测详情:")
            for i, det in enumerate(detections):
                if det.get('is_target', True):
                    print(f"  目标{i + 1}: 类别={det['class_id']}, "
                          f"置信度={det['confidence']:.3f}")
                else:
                    print(f"  背景{i + 1}: 置信度={det['confidence']:.3f}")

        # 4. 保存结果
        result_image.save(output_path)
        print(f"\n✅ 结果保存到: {output_path}")

        # 5. 尝试打开图片（可选）
        try:
            import webbrowser
            webbrowser.open(output_path)
            print(f"📸 正在打开图片...")
        except:
            pass

        return result_image


# 主函数
def main():
    """测试方法"""
    print("=" * 60)
    print("DINOv3+DETR 改进版检测测试")
    print("特点: 降低阈值、显示更多检测、无论如何都绘图")
    print("=" * 60)

    # 创建检测器实例
    detector = SimpleDetector()

    # ✅ 设置测试图像路径
    test_image_path = "test.jpg"  # 修改为你的测试图片路径

    # ✅ 执行检测（使用更低的阈值）
    print(f"\n🔧 测试配置:")
    print(f"  图片: {test_image_path}")
    print(f"  模型: {detector.model_path}")
    print(f"  类别数: {detector.num_classes}")
    print(f"  置信度阈值: 0.1 (较低)")

    # 尝试多个阈值
    thresholds = [0.1, 0.05, 0.01]

    for threshold in thresholds:
        print(f"\n{'=' * 60}")
        print(f"测试阈值: {threshold}")
        print(f"{'=' * 60}")

        # 生成不同的输出文件名
        base_name = os.path.splitext(test_image_path)[0]
        output_path = f"{base_name}_detected_th{threshold}.jpg"

        detector.detect_image(
            image_path=test_image_path,
            output_path=output_path,
            confidence_threshold=threshold
        )

    print("\n" + "=" * 60)
    print("✅ 所有测试完成!")
    print(f"生成了 {len(thresholds)} 个不同阈值的检测结果")
    print("建议检查 threshold=0.01 的结果")
    print("=" * 60)


if __name__ == "__main__":
    main()