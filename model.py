import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class selfAttention(nn.Module):
    def __init__(self, num_heads, input_dim, hidden_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads  # 每个头的维度

        # 线性层：明确输入维度为input_dim（128）
        self.key = nn.Linear(input_dim, hidden_dim)
        self.query = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x):
        # x形状：(batch_size, seq_len, input_dim) = (32, 195, 128)
        batch_size, seq_len, _ = x.shape

        # 线性变换 + 多头拆分
        key = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        query = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力计算
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = F.dropout(attn_probs, p=0.1, training=self.training)

        # 聚合结果
        output = torch.matmul(attn_probs, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return output


class convATTnet(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层（保持不变）
        self.conv1 = nn.Conv1d(21, 64, 16, stride=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 8, stride=1)  # 输出通道=128
        self.bn2 = nn.BatchNorm1d(128)
        self.maxpool = nn.MaxPool1d(5, stride=5)  # 池化序列长度
        self.dropout = nn.Dropout(0.2)

        # 自注意力（input_dim=128，保持不变）
        self.attention = selfAttention(
            num_heads=8,
            input_dim=128,
            hidden_dim=32
        )

        # 全连接层（序列长度修正为195，保持不变）
        self.fc1 = nn.Linear(195 * 32, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # 输入处理（保持不变）
        x = F.one_hot(x, num_classes=21).float()  # (batch, seq_len, 21)
        x = x.permute(0, 2, 1)  # (batch, 21, seq_len)

        # 卷积特征提取（保持不变）
        x = F.relu(self.bn1(self.conv1(x)))  # (batch, 64, 985)
        x = F.relu(self.bn2(self.conv2(x)))  # (batch, 128, 978)

        # 修正：先池化序列长度，再调整维度顺序
        x = self.maxpool(x)  # 对序列长度（最后一维）池化：(batch, 128, 195)
        x = x.permute(0, 2, 1)  # 转为 (batch, 195, 128)（序列长度=195，通道=128）
        x = self.dropout(x)

        # 自注意力（输入维度正确为128）
        x = self.attention(x)  # (batch, 195, 32)

        # 分类头（保持不变）
        x = x.flatten(1)  # (batch, 195×32)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x