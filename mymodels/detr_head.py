import math

import torch
import torch.nn as nn
# 特征图展平后丢失了空间位置信息，必须添加位置编码
class SimpleDETRHead(nn.Module):
    def __init__(self, num_classes=80, num_queries=10,feat_height=14, feat_width=14):#设置八十个类别，最多查询100个目标
        super().__init__()
        self.num_queries = num_queries #传递种类

        # 特征投影
        #降维：将768维特征降到256维，减少计算量  卷积核: 1x1 - 只改变通道数，不改变空间尺寸
        # 输入: [batch, 768, H, W]
        # 输出: [batch, 256, H, W] 🕵️
        self.input_proj = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=3, padding=1),  # 3×3卷积
            nn.BatchNorm2d(512),# 批归一化
            nn.ReLU(inplace=True),# 激活函数
            nn.Conv2d(512, 256, kernel_size=1),  # 再用1×1降维
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # 查询向量
        # 🔥 第1步：创建100张随机区域，和256的区域描述列表
        self.query_embed = nn.Embedding(num_queries, 256)#这里的256和上述的特征值无关
        # 🔥 第2步：创建100个搜索器，和256的搜索器描述列表
        self.content_queries = nn.Embedding(num_queries, 256)
        # 🔥 第3步：给创建的区域和搜索器随机初始化权重列表🕵️
        self._init_queries(num_queries)


        # Transformer解码器（6层）
        decoder_layer = nn.TransformerDecoderLayer(d_model=256, # d_model=256: 特征维度
                                                   nhead=8,# nhead=8: 注意力头数（8个头并行处理）# 每个头维度 = 256 / 8 = 32(八个维度，每个注意32个特征，可以更专业)
                                                   batch_first=True,
                                                   dim_feedforward=2048,  # 增加FFN维度FFN维度=2048就是将原本的256个特征值扩展成2048个，然后进行分析，分析结束后再转回256，这样可以使得到的256个特征值更精准
                                                   dropout=0.1, # 添加dropout关闭0.1的神经元防止过拟合
                                                   activation = 'gelu'  # ✓ 使用GELU激活函数（比ReLU更平滑
                                                   )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)

        # 层归一化维度设置（用于稳定训练）
        self.norm = nn.LayerNorm(256)

        # 预测头（类别）
        self.class_head = nn.Sequential(
            nn.Linear(256, 512),  # ✓ 增加中间维度
            nn.LayerNorm(512),  # ✓ 对应新维度
            #nn.Linear(256, 256),  # 第一层：特征深化（将特征值中的特征组合，和去除噪声等，使得表现更加清晰）
            #nn.LayerNorm(256),  # 将特征值稳定再一定范围，稳定训练
            nn.ReLU(inplace=True),  #将负特征值固定为0 非线性激活
            nn.Dropout(0.2),  # ✓ 提高dropout
            nn.Linear(512, 256),  # ✓ 增加一层
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),  # 防止过拟合
            nn.Linear(256, num_classes + 1)  # 最终分类
        )

        # 预测头（坐标）
        self.bbox_head = nn.Sequential(
            nn.Linear(256, 512),  # ✓ 增加容量
            nn.LayerNorm(512),
            #nn.Linear(256, 256),
            #nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),  # ✓ 适当dropout
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 4)  # [cx, cy, w, h]
        )
        #给每个模块固定坐标
        self.pos_encoding = LearnablePositionalEncoding2D(256, feat_height, feat_width)

        #设置位置和类别的权重，告诉他更应该注重位置还是物种
        self.query_fusion = ResidualQueryFusion(d_model=256)

    def forward(self, feature_map):
        #查看dinov3传输过来的数据 [batch, 768, 14, 14]
        batch_size = feature_map.shape[0]

        # 特征投影
        # 输入: [batch, 768, H, W]
        # 输出: [batch, 256, H, W]
        memory = self.input_proj(feature_map)

        #添加2D位置编码
        memory = self.pos_encoding(memory)

        # 转换维度
        # 假设输入: [2, 256, 14, 14]
        memory = (memory.flatten(2).  # [2, 256, 196]  # 将14×14=196个空间位置展平（像素坐标丢失，需要位置编码）
                  permute(0, 2, 1))  # [2, 196, 256]  # 交换维度


        # 🔥 1. 位置查询（地图）
        query_pos = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        # 🔥 2. 内容查询（大脑）
        tgt = self.content_queries.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        # 将位置信息加到内容中


        #tgt_with_pos = tgt + query_pos
        tgt_with_pos =self.query_fusion(tgt,query_pos)

        # 解码
        output = self.decoder(tgt_with_pos, memory)

        # 层归一化
        output = self.norm(output)#平话权重值

        # 预测
        pred_logits = self.class_head(output)  # [batch, 100, 81]  将256个特征值，转化成81个类的相似度百分比
        pred_boxes = torch.sigmoid(self.bbox_head(output))  # [batch, 100, 4] 给出这100个预测的坐标

        return pred_logits, pred_boxes  # 返回预测的类别和边界框

    def _init_queries(self, num_queries):
        """智能初始化查询向量"""
        #他第一次生成，是比较有经验的分配了一下，以后可以考自己训练
        nn.init.normal_(self.query_embed.weight, mean=0.0, std=0.02)
        # 内容查询（content_queries）：学习物体特征
        # 使用正态分布，标准差0.02（小值避免初始激活过大）
        nn.init.normal_(self.content_queries.weight, mean=0.0, std=0.01)


class LearnablePositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height, width):
        super().__init__()
        # 当前：随机初始化
        # self.pos_encoding = nn.Parameter(torch.randn(1, d_model, height, width))

        # 改进：使用更小的初始化
        self.pos_encoding = nn.Parameter(
            torch.randn(1, d_model, height, width) * 0.02  # ✓ 缩小初始化
        )

    def forward(self, x):
        return x + self.pos_encoding


class ResidualQueryFusion(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        # 轻量级的自适应模块
        self.content_proj = nn.Linear(d_model, d_model, bias=False)
        self.position_proj = nn.Linear(d_model, d_model, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习的缩放因子

    def forward(self, content, position):
        # 分别投影
        content_proj = self.content_proj(content)
        position_proj = self.position_proj(position)

        # 自适应融合
        fused = content_proj + position_proj

        # 残差连接 + 可学习缩放
        output = content + self.gamma * fused

        return output