# Adapted from SenuaLab/EEG-IED-Detection (MIT); see LICENSE and NOTICE.
import torch
from torch import nn

NUM_CHANNELS = 19

class SqueezeExcitation1D(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(channels, channels // reduction, bias=False), nn.ReLU(inplace=True), nn.Linear(channels // reduction, channels, bias=False), nn.Sigmoid())

    def forward(self, x):
        (b, c, _) = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResidualBlock1D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None, use_se=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.se = SqueezeExcitation1D(out_channels) if use_se else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class TemporalAttention(nn.Module):

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        (B, T, C) = x.shape
        residual = x
        x = self.norm(x)
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        (q, k, v) = (qkv[0], qkv[1], qkv[2])
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        x = x + residual
        return x.permute(0, 2, 1)

class MultiScaleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv1d(in_channels, out_channels // 4, kernel_size=3, padding=1, bias=False), nn.BatchNorm1d(out_channels // 4), nn.ReLU(inplace=True))
        self.branch2 = nn.Sequential(nn.Conv1d(in_channels, out_channels // 4, kernel_size=5, padding=2, bias=False), nn.BatchNorm1d(out_channels // 4), nn.ReLU(inplace=True))
        self.branch3 = nn.Sequential(nn.Conv1d(in_channels, out_channels // 4, kernel_size=7, padding=3, bias=False), nn.BatchNorm1d(out_channels // 4), nn.ReLU(inplace=True))
        self.branch4 = nn.Sequential(nn.Conv1d(in_channels, out_channels // 4, kernel_size=11, padding=5, bias=False), nn.BatchNorm1d(out_channels // 4), nn.ReLU(inplace=True))

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)

class EEGResNetAttention(nn.Module):

    def __init__(self, num_channels=NUM_CHANNELS, num_classes=2, input_length=2000):
        super().__init__()
        self.num_channels = num_channels
        self.input_length = input_length
        self.stem = MultiScaleConv(num_channels, 64)
        self.stem_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)
        self.temporal_attn = TemporalAttention(512, num_heads=8)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)
        self._init_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm1d(out_channels))
        layers = [ResidualBlock1D(in_channels, out_channels, stride=stride, downsample=downsample)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stem_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.temporal_attn(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
