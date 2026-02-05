import torch
import torch.nn as nn
from transformers import AutoModel


class FrozenDINOv3(nn.Module):
    def __init__(self):
        super().__init__()
        # 从本地加载DINOv3
        self.model = AutoModel.from_pretrained("./weights/facebook/dinov3-base")

        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False

        self.feature_dim = 768

    def forward(self, x):
        with torch.no_grad():
            outputs = self.model(x)#给模型的数据
            features = outputs.last_hidden_state[:, 5:, :]  # 去掉[CLS] token
            batch_size = x.shape[0]
            feature_map = features.permute(0, 2, 1).view(batch_size, self.feature_dim, 14, 14)
            return feature_map
    #可以正常返回特征矩阵  输入 torch.Size([1, 3, 224, 224]) -> 输出 torch.Size([1, 768, 14, 14])
