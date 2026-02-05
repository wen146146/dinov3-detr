import math

import torch
import torch.nn as nn
# 特征图展平后丢失了空间位置信息，必须添加位置编码
class SimpleDETRHead(nn.Module):
    def __init__(self, num_classes=80, num_queries=10,feat_height=14, feat_width=14):#设置八十个类别，最多查询100个目标
        super().__init__()
        self.num_queries = num_queries

        # 特征投影
        #self.input_proj = nn.Conv2d(768, 256, kernel_size=1)
        #降维：将768维特征降到256维，减少计算量  卷积核: 1x1 - 只改变通道数，不改变空间尺寸
        self.input_proj = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=3, padding=1),  # 3×3卷积
            nn.BatchNorm2d(512),# 批归一化
            nn.ReLU(inplace=True),# 激活函数
            nn.Conv2d(512, 256, kernel_size=1),  # 再用1×1降维
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # 查询向量
        # 🔥 第1步：创建100张地图，和
        self.query_embed = nn.Embedding(num_queries, 256)#这里的256和上述的特征值无关
        # 🔥 第2步：给每个侦探培训专业技能，告诉侦探应该找什么
        self.content_queries = nn.Embedding(num_queries, 256)
        #给100个侦探分配任务
        self._init_queries(num_queries)
        #  查询数量: num_queries=100 - DETR标准设置
        # 每个查询对应一个可能的检测结果
        # 100个查询最多检测100个目标
        # 维度: 256 - 与Transformer隐藏维度一致
        # Transformer解码器（1层简化版）
        decoder_layer = nn.TransformerDecoderLayer(d_model=256, # d_model=256: 特征维度
                                                   nhead=8,# nhead=8: 注意力头数（8个头并行处理）# 每个头维度 = 256 / 8 = 32(八个维度，每个注意32个特征，可以更专业)
                                                   batch_first=True,
                                                   dim_feedforward=2048,  # 增加FFN维度FFN维度=2048就是将原本的256个特征值扩展成2048个，然后进行分析，分析结束后再转回256，这样可以使得到的256个特征值更精准
                                                   dropout=0.1  # 添加dropout关闭0.1的神经元防止过拟合
                                                   )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        # 层归一化（用于稳定训练）
        self.norm = nn.LayerNorm(256)
        # 预测头
        # 改为：
        self.class_head = nn.Sequential(
            nn.Linear(256, 256),  # 第一层：特征深化（将特征值中的特征组合，和去除噪声等，使得表现更加清晰）
            nn.LayerNorm(256),  # 将特征值稳定再一定范围，稳定训练
            nn.ReLU(inplace=True),  #将负特征值固定为0 非线性激活
            nn.Dropout(0.1),  # 防止过拟合
            nn.Linear(256, num_classes + 1)  # 最终分类
        )

        self.bbox_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 4)  # [cx, cy, w, h]
        )

        self.pos_encoding = PositionalEncoding2D(256, feat_height, feat_width)

    def forward(self, feature_map):
        batch_size = feature_map.shape[0]

        # 特征投影
        memory = self.input_proj(feature_map)
        # 输入: [batch, 768, H, W]
        # 输出: [batch, 256, H, W]
        memory = self.pos_encoding(memory)  # 🔥 在这里添加2D位置编码
        # memory: [batch, 256, H, W]，已经包含位置信息
        # 假设输入: [2, 256, 14, 14]
        memory = (memory.flatten(2).  # [2, 256, 196]  # 将14×14=196个空间位置展平（像素坐标丢失，需要位置编码）
                  permute(0, 2, 1))  # [2, 196, 256]  # 交换维度

        # 🔥 1. 位置查询（地图）
        query_pos = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        # 🔥 2. 内容查询（大脑）
        tgt = self.content_queries.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        # ✅ 正确的解码器调用（将位置信息加到内容中）：
        # 方法1：直接相加（最简单）
        tgt_with_pos = tgt + query_pos  # 将位置信息加到内容中
        # 解码
        output = self.decoder(tgt_with_pos, memory)
        # output: [batch, 100, 256]  # 更新后的查询表示 通过memory的数据解码，获得的tgt
        # 每张图像有100个查询，每个查询有256维的表示，这个表示包含了该查询关注的物体信息（正常需要六个）
        # 层归一化
        output = self.norm(output)#平话权重值
        # 预测
        pred_logits = self.class_head(output)  # [batch, 100, 81]  将256个特征值，转化成81个类的相似度百分比
        pred_boxes = torch.sigmoid(self.bbox_head(output))  # [batch, 100, 4] 给出这100个预测的坐标

        return pred_logits, pred_boxes  # 返回预测的类别和边界框

    def _init_queries(self, num_queries):
        """智能初始化查询向量"""
        # 方法1：更好的权重初始化
        # nn.init.xavier_uniform_(self.query_embed.weight)  # 位置查询初始化
        # nn.init.xavier_uniform_(self.content_queries.weight)  # 内容查询初始化
        #他第一次生成，是比较有经验的分配了一下，以后可以考自己训练
        nn.init.uniform_(self.query_embed.weight, -1.0, 1.0)

        # 内容查询（content_queries）：学习物体特征
        # 使用正态分布，标准差0.02（小值避免初始激活过大）
        nn.init.normal_(self.content_queries.weight, mean=0.0, std=0.02)


class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height, width):
        super().__init__()
        if d_model % 4 != 0:
            d_model = (d_model // 4) * 4

        pe = torch.zeros(1, d_model, height, width)

        # 每个位置编码的维度（x和y各占一半）
        d_model_half = d_model // 2

        # 为高度和宽度分别创建位置编码
        pos_h = torch.arange(height).float().unsqueeze(1)  # [height, 1]
        pos_w = torch.arange(width).float().unsqueeze(0)  # [1, width]

        # 生成不同频率
        div_term = torch.exp(
            torch.arange(0, d_model_half, 2).float() *
            -(math.log(10000.0) / d_model_half)
        )  # [d_model_half/2]

        # 高度编码（前d_model_half个通道）
        for i in range(0, d_model_half, 2):
            freq = div_term[i // 2]
            pe[0, i, :, :] = torch.sin(pos_h * freq).expand(-1, width)
            pe[0, i + 1, :, :] = torch.cos(pos_h * freq).expand(-1, width)

        # 宽度编码（后d_model_half个通道）
        for i in range(0, d_model_half, 2):
            freq = div_term[i // 2]
            pe[0, d_model_half + i, :, :] = torch.sin(pos_w * freq).expand(height, -1)
            pe[0, d_model_half + i + 1, :, :] = torch.cos(pos_w * freq).expand(height, -1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe
