#!/usr/bin/env python3
"""
最简化的DINOv3 + DETR检测头训练脚本
包含所有必要功能
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import numpy as np
from tqdm import tqdm
import warnings

import math  # 在文件开头添加
import matplotlib.pyplot as plt  # 可选


def adjust_learning_rate(optimizer, epoch, warmup_epochs, base_lr, num_epochs, method='cosine'):
    """
    学习率调整策略（预热 + 衰减）

    Args:
        optimizer: 优化器
        epoch: 当前epoch（从0开始）
        warmup_epochs: 预热epoch数
        base_lr: 基础学习率
        num_epochs: 总epoch数
        method: 衰减方法 ('cosine', 'step', 'linear')

    Returns:
        当前学习率
    """
    if epoch < warmup_epochs:
        # 🔥 线性预热：从0.1倍学习率逐渐增加到1倍
        warmup_factor = (epoch + 1) / warmup_epochs
        lr = base_lr * warmup_factor

    else:
        if method == 'cosine':
            # 🔥 余弦衰减（最平滑）
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            lr = 0.5 * base_lr * (1 + math.cos(math.pi * progress))

        elif method == 'step':
            # 阶梯衰减
            decay_factor = 0.5 ** ((epoch - warmup_epochs) // 3)
            lr = base_lr * decay_factor

        elif method == 'linear':
            # 线性衰减
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            lr = base_lr * (1 - progress)
        else:
            lr = base_lr

    # 设置学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def plot_training_history(history):
    """
    绘制训练历史图表（可选）
    """
    if not history['epoch']:
        return

    plt.figure(figsize=(12, 8))

    # 1. 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(history['epoch'], history['train_loss'], 'b-', label='训练损失', linewidth=2)
    plt.plot(history['epoch'], history['val_loss'], 'r-', label='验证损失', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('训练和验证损失')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. 学习率曲线
    plt.subplot(2, 2, 2)
    plt.plot(history['epoch'], history['lr'], 'g-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('学习率')
    plt.title('学习率变化（预热+衰减）')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # 3. 验证损失放大
    plt.subplot(2, 2, 3)
    plt.plot(history['epoch'], history['val_loss'], 'r-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('验证损失')
    plt.title('验证损失变化（早停监控）')
    plt.grid(True, alpha=0.3)

    # 找到最小值点
    min_idx = np.argmin(history['val_loss'])
    plt.scatter(history['epoch'][min_idx], history['val_loss'][min_idx],
                color='green', s=100, zorder=5, label=f'最佳: {history["val_loss"][min_idx]:.4f}')
    plt.legend()

    # 4. 损失差值（过拟合程度）
    plt.subplot(2, 2, 4)
    diff = [t - v for t, v in zip(history['train_loss'], history['val_loss'])]
    plt.plot(history['epoch'], diff, 'm-', linewidth=2)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Epoch')
    plt.ylabel('训练损失 - 验证损失')
    plt.title('过拟合程度（正值可能过拟合）')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print(f"📈 训练历史已保存到: training_history.png")
    plt.show()

# 导入自定义模型
from mymodels.dinov3_detr import DINOv3DETR


# ==================== 2. 匈牙利匹配器 ====================
class HungarianMatcher(nn.Module):
    """匈牙利匹配器，用于找到最佳预测-目标匹配"""

    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, pred_logits, pred_boxes, targets):
        """
        Args:
            pred_logits: [batch_size, num_queries, num_classes+1]
            pred_boxes: [batch_size, num_queries, 4] (cx, cy, w, h)
            targets: list of dict with keys 'boxes', 'labels'
        Returns:
            indices: list of tuples (pred_idx, target_idx) for each batch
        """
        batch_size = pred_logits.shape[0]
        num_queries = pred_logits.shape[1]

        # 存储每张图片的匹配结果
        indices = []

        for batch_idx in range(batch_size):
            # 获取当前batch的目标
            target_boxes = targets[batch_idx]['boxes']  # [num_targets, 4]
            target_labels = targets[batch_idx]['labels']  # [num_targets]
            num_targets = len(target_boxes)

            # 如果没有目标，所有查询都匹配到"无对象"
            if num_targets == 0:
                indices.append((torch.tensor([], dtype=torch.int64),
                                torch.tensor([], dtype=torch.int64)))
                continue

            # 计算分类损失成本
            pred_logit = pred_logits[batch_idx]  # [num_queries, num_classes+1]

            # 获取目标类别的概率（负值，因为匈牙利算法找最小成本）
            cost_class = -pred_logit[:, target_labels]  # [num_queries, num_targets]

            # 计算L1边界框损失成本
            pred_box = pred_boxes[batch_idx]  # [num_queries, 4]
            target_boxes = target_boxes.to(pred_box.device)
            cost_bbox = torch.cdist(pred_box, target_boxes, p=1)  # [num_queries, num_targets]

            # 计算GIoU损失成本
            cost_giou = -self.generalized_box_iou(
                self.box_cxcywh_to_xyxy(pred_box),
                self.box_cxcywh_to_xyxy(target_boxes)
            )

            # 组合成本矩阵
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.cpu()  # 匈牙利算法在CPU上运行

            # 执行匈牙利匹配
            C = C.reshape(num_queries, -1).detach()

            if num_targets < num_queries:
                # 如果目标数小于查询数，填充虚拟目标
                C = torch.cat([C, torch.zeros(num_queries, num_queries - num_targets)], dim=1)

            # 使用scipy的线性分配（匈牙利算法）
            from scipy.optimize import linear_sum_assignment
            indices_i = linear_sum_assignment(C)

            # 转换为PyTorch张量
            indices_i = (torch.as_tensor(indices_i[0], dtype=torch.int64),
                         torch.as_tensor(indices_i[1], dtype=torch.int64))

            # 过滤掉虚拟目标的匹配
            if num_targets < num_queries:
                mask = indices_i[1] < num_targets
                indices_i = (indices_i[0][mask], indices_i[1][mask])

            indices.append(indices_i)

        return indices

    @staticmethod
    def box_cxcywh_to_xyxy(x):
        """将(cx, cy, w, h)转换为(x1, y1, x2, y2)"""
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)

    @staticmethod
    def generalized_box_iou(boxes1, boxes2):
        """
        计算广义IoU
        boxes1: [N, 4] (x1, y1, x2, y2)
        boxes2: [M, 4] (x1, y1, x2, y2)
        返回: [N, M] GIoU值
        """
        # 确保boxes2在boxes1的设备上
        boxes2 = boxes2.to(boxes1.device)

        # 计算交集面积
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
        wh = (rb - lt).clamp(min=0)  # [N, M, 2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

        # 计算各自的面积
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [N]
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [M]

        union = area1[:, None] + area2[None, :] - inter

        iou = inter / union

        # 计算最小包围框
        lt_min = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rb_max = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
        wh_min = (rb_max - lt_min).clamp(min=0)
        area_min = wh_min[:, :, 0] * wh_min[:, :, 1]

        # 计算GIoU
        giou = iou - (area_min - union) / area_min

        return giou


# ==================== 3. 匈牙利损失函数 ====================
class HungarianLoss(nn.Module):
    """使用匈牙利匹配的DETR损失函数"""

    def __init__(self, num_classes, matcher=None):
        super().__init__()
        self.num_classes = num_classes

        # 使用默认匈牙利匹配器
        self.matcher = matcher if matcher is not None else HungarianMatcher()

        # 损失权重
        self.weight_class = 1.0
        self.weight_bbox = 5.0
        self.weight_giou = 2.0

        # 基础损失函数
        self.loss_class = nn.CrossEntropyLoss()
        self.loss_bbox = nn.L1Loss()

    def forward(self, pred_logits, pred_boxes, targets):
        """
        Args:
            pred_logits: [batch, num_queries, num_classes+1]
            pred_boxes: [batch, num_queries, 4]
            targets: list of dict with 'boxes' and 'labels'
        """
        # 第一步：匈牙利匹配找到最佳配对
        indices = self.matcher(pred_logits, pred_boxes, targets)

        total_loss = 0
        batch_size = pred_logits.shape[0]

        for batch_idx in range(batch_size):
            # 获取匹配结果
            idx_pred, idx_target = indices[batch_idx]

            # 如果没有匹配的目标，跳过
            if len(idx_pred) == 0:
                continue

            # 提取匹配的预测和目标
            matched_pred_logits = pred_logits[batch_idx, idx_pred]  # [num_matched, num_classes+1]
            matched_pred_boxes = pred_boxes[batch_idx, idx_pred]  # [num_matched, 4]

            target_boxes = targets[batch_idx]['boxes'][idx_target]  # [num_matched, 4]
            target_labels = targets[batch_idx]['labels'][idx_target]  # [num_matched]
            device = pred_logits.device
            target_boxes = target_boxes.to(device)
            target_labels = target_labels.to(device)
            # 分类损失
            loss_class = self.loss_class(matched_pred_logits, target_labels)

            # 边界框L1损失
            loss_bbox = self.loss_bbox(matched_pred_boxes, target_boxes)

            # GIoU损失
            loss_giou = 1.0 - self.matcher.generalized_box_iou(
                self.matcher.box_cxcywh_to_xyxy(matched_pred_boxes),
                self.matcher.box_cxcywh_to_xyxy(target_boxes)
            ).diag().mean()

            # 组合损失
            loss = (self.weight_class * loss_class +
                    self.weight_bbox * loss_bbox +
                    self.weight_giou * loss_giou)

            total_loss += loss

        # 如果没有匹配的目标，计算"无对象"的分类损失
        if total_loss == 0:
            for batch_idx in range(batch_size):
                if len(targets[batch_idx]['boxes']) == 0:
                    # 所有预测都应该是"无对象"
                    cls_target = torch.full((pred_logits.shape[1],),
                                            self.num_classes,
                                            dtype=torch.long,
                                            device=pred_logits.device)
                    loss = self.loss_class(pred_logits[batch_idx], cls_target)
                    total_loss += loss

        return total_loss / max(1, batch_size)

# ==================== 1. 自定义数据集（内联实现） ====================
class SimpleCOCODataset(Dataset):
    """简化的COCO格式数据集"""

    def __init__(self, data_root, split='train', image_size=224):
        """
        Args:
            data_root: 数据根目录 (如: ./datasets/coco)
            split: 数据集划分 ('train', 'valid', 'test')
            image_size: 图像尺寸
        """
        self.data_root = data_root
        self.split = split
        self.image_size = image_size

        # 图像目录
        self.image_dir = os.path.join(data_root, split)

        # 标注文件路径 - 现在在对应的图片文件夹中
        self.annotation_file = os.path.join(self.image_dir, f"{split}.json")
        #标注文件目录

        print(f"图像目录: {self.image_dir}")
        print(f"标注文件: {self.annotation_file}")

        # 检查文件是否存在
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"图像目录不存在: {self.image_dir}")
        if not os.path.exists(self.annotation_file):
            raise FileNotFoundError(f"标注文件不存在: {self.annotation_file}")

        # 加载标注
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 准备数据
        self.images = [] # 图像路径
        self.targets = [] # 标注信息

        # 建立图像映射
        image_dict = {img['id']: img for img in data['images']}
        # 建立标注映射 images中存放的图片的id和图片的信息现在就可以用id找到信息
        #创建图像# 结果：{0: 第一张图片信息, 1: 第二张图片信息, 2: 第三张图片信息}
        ann_dict = {}

        # 组织标注
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in ann_dict:
                ann_dict[img_id] = []
            ann_dict[img_id].append(ann)
        #annotations中存放这所以图片的所以标注的点坐标，通过遍历，来分类，ann_dict就是存储图片id，
        # 如果还没存储某一张的图片就将下标为该id的数组创建一个列表
        #用来将该图片的所以点坐标分类
        # 结果：{0: [标注1-7], 1: [标注8-11], 2: [标注12-17]}
        # 创建样本列表
        images_found = 0
        for img_id, img_info in image_dict.items():
            # 图像文件名
            filename = img_info['file_name']

            # 图像路径 - 现在在对应的split目录中
            img_path = os.path.join(self.image_dir, filename)

            # 检查文件是否存在
            if not os.path.exists(img_path):
                print(f"⚠️ 警告: 图像文件不存在: {img_path}")
                # 尝试使用绝对路径
                img_path = os.path.abspath(img_path)
                if not os.path.exists(img_path):
                    print(f"  绝对路径也不存在: {img_path}")
                    continue

            images_found += 1

            # 获取该图像的所有标注
            anns = ann_dict.get(img_id, [])
            boxes = [] #边界框
            labels = []#类别

            for ann in anns:
                # 边界框 [x, y, w, h] 归一化
                bbox = ann['bbox']
                x, y, w, h = bbox

                # 归一化到0-1
                x = x / img_info['width']
                y = y / img_info['height']
                w = w / img_info['width'] #宽度百分比，加上x百分比就等于右边x的百分比
                h = h / img_info['height']

                boxes.append([x, y, w, h])
                labels.append(ann['category_id'] - 1)  # 0-based
                #类别标签id，结构是从1开始，我们改为从零开始

            self.images.append(img_path)
            self.targets.append({
                'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else# torch.Size([3, 4])
                torch.zeros((0, 4), dtype=torch.float32),#图片没有标注时
                # tensor([[0.2000, 0.3000, 0.1000, 0.1500],
                #         [0.5000, 0.4000, 0.2000, 0.1000],
                #         [0.7000, 0.6000, 0.1500, 0.2000]])
                'labels': torch.tensor(labels, dtype=torch.long) if labels else# torch.Size([3])
                # tensor([2, 0, 1])
                torch.zeros((0,), dtype=torch.long)
            })

        print(f"数据集加载完成: {len(self.images)} 张图片")
        print(f"  标注文件中图像: {len(image_dict)} 张")
        print(f"  实际找到图像: {images_found} 张")

        if self.images:
            print(f"示例图像路径: {self.images[0]}")
            print(f"示例目标: {len(self.targets[0]['boxes'])} 个边界框")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):#通过对象[？]来将该图片转换成模型能看到的数据，从而给到模型
        # 加载图像
        try:
            img = Image.open(self.images[idx]).convert('RGB')# 🔥 这里转换成RGB三通道
        except Exception as e:
            print(f"❌ 无法加载图像 {self.images[idx]}: {e}")
            # 返回一个白色占位图像
            img = Image.new('RGB', (self.image_size, self.image_size), color='white')

        img = img.resize((self.image_size, self.image_size))

        # 转换为tensor并归一化
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        # 🔥 这里：array(img) → [H, W, 3] 三通道
        # 🔥 permute(2, 0, 1) → [3, H, W] 通道在前
        # img_tensor = torch.tensor([
        #     # 红色通道 (R)
        #     [[0.9, 0.2, 0.5],  # 第1行：3个像素的红色值
        #      [0.3, 0.8, 0.1],  # 第2行
        #      [0.7, 0.4, 0.6]],  # 第3行
        #
        #     # 绿色通道 (G)
        #     [[0.1, 0.8, 0.3],
        #      [0.6, 0.2, 0.9],
        #      [0.4, 0.7, 0.5]],
        #
        #     # 蓝色通道 (B)
        #     [[0.5, 0.3, 0.9],
        #      [0.2, 0.7, 0.4],
        #      [0.8, 0.1, 0.6]]
        # ])  # 形状：[3, 3, 3] = [通道, 高度, 宽度]
        return img_tensor, self.targets[idx]


def collate_fn(batch):   #将图片变为张量然后堆叠
    """自定义批次处理"""
    images = []
    targets = []

    for img, target in batch:
        images.append(img)  #存放图片 现在是链表形式存放
        targets.append(target) #存放标签

    images = torch.stack(images, dim=0) #将图像的像素值连续存储起来，堆叠成一个张量（模型一次训练一个张量），dim=0在零（最前面）维度添加维度
    return images, targets


# ==================== 2. 旧的损失函数（内联实现） ====================
class SimpleDetrLoss(nn.Module):
    """简化的DETR损失函数"""

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes#需要检测种类
        # 分类损失
        self.cls_loss = nn.CrossEntropyLoss()#交叉熵损失函数
        # 边界框损失
        self.bbox_loss = nn.L1Loss()#均方误差损失函数
        #给这个函数定义他损失所以的运算函数

    def forward(self, pred_logits, pred_boxes, targets):
        """
        pred_logits: [batch, num_queries, num_classes+1],识别出的种类百分比
        pred_boxes: [batch, num_queries, 4] 识别出的坐标
        targets: list of dict with 'boxes' and 'labels'  答案
        """
        batch_size = pred_logits.shape[0]#样本数量
        total_loss = 0

        for i in range(batch_size):
            # 获取目标
            target_boxes = targets[i]['boxes']  # [num_objects, 4]
            target_labels = targets[i]['labels']  # [num_objects]

            # 如果没有目标，只计算"无对象"的分类损失
            if len(target_boxes) == 0:
                # 所有预测都应该是"无对象"
                cls_target = torch.full((pred_logits.shape[1],),
                                        self.num_classes,  # "无对象"类别 这里的num_classes不是代表有多少类，而是代表他是第几类，刚好是无类型
                                        dtype=torch.long,
                                        device=pred_logits.device)#与预测张量同一设备
                loss = self.cls_loss(pred_logits[i], cls_target)#将第n张图片中的100个预测目标的预测种类，和答题卡
                total_loss += loss
                continue

            # 简化的匹配：每个目标分配给一个查询
            num_objects = min(len(target_boxes), pred_logits.shape[1]) #答案中的目标数和查询到的目标数的最小值

            # 分类损失
            cls_target = torch.full((pred_logits.shape[1],),
                                    self.num_classes,  # 默认"无对象"
                                    dtype=torch.long,
                                    device=pred_logits.device)
            cls_target[:num_objects] = target_labels[:num_objects] #将答案中的目标数，和查询到的目标数的最小值，作为答案(非常低级)
            loss_cls = self.cls_loss(pred_logits[i], cls_target)
        #，老师先将做一份全空的答案，表明全部没有答案，然后在有答案的下标数组中填入真确答案，这样到时候检测时，空的就表明一定错误，不空的话，再进行比对
            # 边界框损失
            loss_bbox = self.bbox_loss(pred_boxes[i, :num_objects],target_boxes[:num_objects])#第i张图的所有检测目标和标准答案

            total_loss += loss_cls + 5.0 * loss_bbox  # 给bbox损失更高权重（边框损失看的更重）

        return total_loss / batch_size


# ==================== 3. 训练函数 ====================
def train_detr_head():
    """主训练函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # ========== 数据集路径 ==========
    data_root = "./datasets/coco"
    train_split = "train"
    val_split = "valid"
    # ===============================

    # 创建数据集
    print("加载数据集...")
    try:
        train_dataset = SimpleCOCODataset(data_root, split=train_split, image_size=224)
        print(f"✅ 训练集加载成功: {len(train_dataset)} 张图片")
    except Exception as e:
        print(f"❌ 训练集加载失败: {e}")
        return

    try:
        val_dataset = SimpleCOCODataset(data_root, split=val_split, image_size=224)
        print(f"✅ 验证集加载成功: {len(val_dataset)} 张图片")
    except Exception as e:
        print(f"⚠️ 验证集加载失败: {e}")
        print("⚠️ 将使用训练集的一部分作为验证集")

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,  # 🔥 注意：这里还是2，但通过梯度累积等效放大
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    print(f"训练集: {len(train_dataset)} 张图片")
    print(f"验证集: {len(val_dataset)} 张图片")
    print(f"训练批次大小: {train_loader.batch_size}")
    print(f"训练步数/epoch: {len(train_loader)}")

    # 创建模型
    print("\n创建模型...")
    try:
        model = DINOv3DETR(num_classes=10).to(device)
        print("✅ 模型创建成功")
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return

    # 创建损失函数
    criterion = HungarianLoss(num_classes=10).to(device)
    print("✅ 使用匈牙利匹配损失函数")

    # ========== 🔥 优化1: 梯度累积设置 ==========
    accumulation_steps = 2  # 累积8个batch，相当于batch_size=16
    effective_batch_size = train_loader.batch_size * accumulation_steps

    # 根据梯度累积调整学习率（经验公式）
    base_lr = 1e-4
    adjusted_lr = base_lr * (effective_batch_size / train_loader.batch_size) ** 0.5

    print(f"\n🎯 梯度累积配置:")
    print(f"  实际batch_size: {train_loader.batch_size}")
    print(f"  累积步数: {accumulation_steps}")
    print(f"  等效batch_size: {effective_batch_size}")
    print(f"  调整后学习率: {adjusted_lr:.2e} (原: {base_lr:.2e})")
    # ==========================================

    # ========== 🔥 优化2: 学习率预热参数 ==========
    warmup_epochs = 3  # 预热3个epoch
    lr_schedule_method = 'cosine'  # 余弦衰减（比阶梯衰减更平滑）

    print(f"\n🎯 学习率预热配置:")
    print(f"  预热epoch数: {warmup_epochs}")
    print(f"  衰减策略: {lr_schedule_method}")
    # ============================================

    # ========== 🔥 优化3: 早停策略参数 ==========
    patience = 5  # 容忍连续5个epoch验证损失不下降
    best_val_loss = float('inf')  # 最佳验证损失
    patience_counter = 0  # 当前连续不下降次数
    best_model_path = 'best_detr_model.pth'  # 最佳模型保存路径

    # 记录训练历史
    train_history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }

    print(f"\n🎯 早停策略配置:")
    print(f"  容忍连续不下降epoch数: {patience}")
    print(f"  最佳模型保存路径: {best_model_path}")
    # ===========================================

    # 创建优化器（使用调整后的学习率）
    optimizer = optim.AdamW(
        model.detr_head.parameters(),
        lr=adjusted_lr,  # 🔥 使用梯度累积调整后的学习率
        weight_decay=1e-4
    )

    # 🔥 注意：删除原来的StepLR调度器，改为手动控制
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # 训练循环
    num_epochs = 100
    print(f"\n开始训练DETR检测头 ({num_epochs}个epoch)...")
    print("=" * 60)

    for epoch in range(num_epochs):
        # 🔥 优化2: 在每个epoch开始调整学习率（预热+衰减）
        current_lr = adjust_learning_rate(
            optimizer, epoch, warmup_epochs,
            base_lr=adjusted_lr,  # 使用调整后的基础学习率
            num_epochs=num_epochs,
            method=lr_schedule_method
        )

        # 训练模式
        model.train()
        train_loss = 0
        train_items = 0

        # 🔥 优化1: 梯度累积计数器
        accumulation_counter = 0

        # 训练一个epoch
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(device)

            # 前向传播
            pred_logits, pred_boxes = model(images)

            # 计算损失
            loss = criterion(pred_logits, pred_boxes, targets)

            # 🔥 优化1: 梯度累积 - 缩放损失
            loss = loss / accumulation_steps

            # 🔥 优化1: 梯度累积 - 反向传播（累积梯度）
            loss.backward()

            # 🔥 优化1: 梯度累积 - 计数器+1
            accumulation_counter += 1

            # 🔥 优化1: 梯度累积 - 判断是否达到累积步数
            if accumulation_counter % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # 更新权重
                optimizer.step()

                # 清空梯度
                optimizer.zero_grad()

                # 更新进度条显示（注意还原真实损失值）
                real_loss = loss.item() * accumulation_steps
                progress_bar.set_postfix({
                    'loss': f'{real_loss:.4f}',
                    'avg_loss': f'{train_loss / (batch_idx + 1):.4f}',
                    'lr': f'{current_lr:.2e}',  # 🔥 显示当前学习率
                    'step': f'{(batch_idx + 1) // accumulation_steps}/{(len(train_loader) + accumulation_steps - 1) // accumulation_steps}'
                })

            # 🔥 记录损失（注意：乘回accumulation_steps得到真实损失）
            real_loss = loss.item() * accumulation_steps
            train_loss += real_loss
            train_items += images.shape[0]

        # 🔥 优化1: 确保最后一批也更新（如果还有未更新的梯度）
        if accumulation_counter % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        # 验证
        model.eval()
        val_loss = 0
        val_items = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                pred_logits, pred_boxes = model(images)
                loss = criterion(pred_logits, pred_boxes, targets)
                val_loss += loss.item()
                val_items += images.shape[0]

        # 🔥 注意：删除原来的scheduler.step()调用
        # scheduler.step()

        # 计算平均损失
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0

        # 🔥 优化3: 记录训练历史
        train_history['epoch'].append(epoch + 1)
        train_history['train_loss'].append(avg_train_loss)
        train_history['val_loss'].append(avg_val_loss)
        train_history['lr'].append(current_lr)

        # 🔥 优化3: 早停判断
        if avg_val_loss < best_val_loss:
            # 有改善：保存最佳模型，重置计数器
            improvement = best_val_loss - avg_val_loss
            best_val_loss = avg_val_loss
            patience_counter = 0

            # 保存最佳模型
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss,
                'learning_rate': current_lr,
            }, best_model_path)

            print(f"\nEpoch {epoch + 1}/{num_epochs} 完成:")
            print(f"  🏆 最佳模型更新！验证损失: {avg_val_loss:.6f} (提升: {improvement:.6f})")

        else:
            # 没有改善：计数器+1
            patience_counter += 1

            print(f"\nEpoch {epoch + 1}/{num_epochs} 完成:")
            print(f"  ⚠️ 验证损失未改善 ({patience_counter}/{patience})")

        print(f"  训练损失: {avg_train_loss:.6f}")
        print(f"  验证损失: {avg_val_loss:.6f}")
        print(f"  学习率: {current_lr:.2e}")

        # 🔥 优化3: 检查是否需要早停
        if patience_counter >= patience:
            print(f"\n🚫 早停触发！连续{patience}个epoch验证损失未改善")
            print(f"   最佳验证损失: {best_val_loss:.6f}")
            break  # 跳出训练循环

        # 保存检查点（每2个epoch或最后）
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            save_path = f'detr_head_epoch_{epoch + 1}.pth'
            torch.save(model.detr_head.state_dict(), save_path)
            print(f"  💾 当前模型保存到: {save_path}")

           # checkpoint_path = f'checkpoint_epoch_{epoch + 1}.pth'
           #  torch.save({
           #      'epoch': epoch + 1,
           #      'model_state_dict': model.state_dict(),
           #      'optimizer_state_dict': optimizer.state_dict(),
           #      'train_loss': avg_train_loss,
           #      'val_loss': avg_val_loss,
           #      'learning_rate': current_lr,
           #  }, checkpoint_path)
           # print(f"  📦 检查点保存到: {checkpoint_path}")

    print("\n" + "=" * 60)

    # 🔥 优化3: 训练结束后加载最佳模型
    if os.path.exists(best_model_path):
        print(f"🔄 加载最佳模型: {best_model_path}")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        print(f"   最佳验证损失: {checkpoint['val_loss']:.6f}")
        print(f"   对应训练损失: {checkpoint['train_loss']:.6f}")
        print(f"   对应epoch: {checkpoint['epoch']}")
    else:
        print("⚠️ 未找到最佳模型，使用最后epoch的模型")

    # 保存最终模型（现在保存的是最佳模型）
    final_path = 'your_model.pth'
    torch.save(model.detr_head.state_dict(), final_path)

    print("🎉 训练完成！")
    print(f"最终模型保存到: {final_path}")

    # 显示训练总结
    print("\n📊 训练总结:")
    print(f"  总epoch数: {min(epoch + 1, num_epochs)}")  # 考虑早停可能提前结束
    print(f"  训练集大小: {len(train_dataset)}")
    print(f"  验证集大小: {len(val_dataset)}")
    print(f"  批次大小: {train_loader.batch_size}")
    print(f"  梯度累积步数: {accumulation_steps}")
    print(f"  等效batch_size: {effective_batch_size}")
    print(f"  最佳验证损失: {best_val_loss:.6f}")
    print("=" * 60)

    # 🔥 可选：绘制训练历史
    try:
        plot_training_history(train_history)
    except Exception as e:
        print(f"⚠️ 训练历史绘图失败: {e}")


# ==================== 4. 主程序入口 ====================
if __name__ == '__main__':
    #设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)

    # 忽略警告
    warnings.filterwarnings('ignore')

    # 运行训练
    train_detr_head()