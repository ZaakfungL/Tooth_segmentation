# models/nas_unet.py
import torch
import torch.nn as nn
from typing import Dict, Any, List


class NASUNet(nn.Module):
    def __init__(self, initial_channels: int, encoder_blocks: List[Dict],
                 decoder_blocks: List[Dict], bottleneck: Dict, **kwargs):  # 添加**kwargs接收额外参数
        super(NASUNet, self).__init__()
        self.initial_channels = initial_channels

        # 初始卷积层
        self.input_conv = nn.Conv2d(1, initial_channels, kernel_size=3, padding=1)

        # 构建编码器
        self.encoder_stages = nn.ModuleList()
        self.downsample = nn.ModuleList()

        for block in encoder_blocks:
            stage = self._create_block(block)
            self.encoder_stages.append(stage)
            self.downsample.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # 构建bottleneck
        self.bottleneck = self._create_block(bottleneck)

        # 构建解码器
        self.decoder_stages = nn.ModuleList()
        self.upsample = nn.ModuleList()

        for block in decoder_blocks:
            self.upsample.append(
                nn.ConvTranspose2d(
                    block['in_channels'] // 2,
                    block['in_channels'] // 2,
                    kernel_size=2,
                    stride=2
                )
            )
            stage = self._create_block(block)
            self.decoder_stages.append(stage)

        # 最终输出层
        self.output_conv = nn.Conv2d(initial_channels, 1, kernel_size=1)

    def _create_block(self, block: Dict[str, Any]) -> nn.Module:
        """根据block配置创建一个块"""
        layers = []
        in_channels = block['in_channels']
        out_channels = block['out_channels']

        # 创建操作序列
        for op in block['operations']:
            if op == 'conv3x3':
                layers.extend([
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    self._get_norm_layer(block['normalization'], out_channels),
                    self._get_activation(block['activation'])
                ])
            elif op == 'conv5x5':
                layers.extend([
                    nn.Conv2d(in_channels, out_channels, 5, padding=2),
                    self._get_norm_layer(block['normalization'], out_channels),
                    self._get_activation(block['activation'])
                ])
            elif op == 'dilated_conv3x3':
                layers.extend([
                    nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2),
                    self._get_norm_layer(block['normalization'], out_channels),
                    self._get_activation(block['activation'])
                ])
            elif op == 'separable_conv3x3':
                layers.extend([
                    nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
                    nn.Conv2d(in_channels, out_channels, 1),
                    self._get_norm_layer(block['normalization'], out_channels),
                    self._get_activation(block['activation'])
                ])

            in_channels = out_channels

        # 添加注意力机制（如果需要）
        if block.get('use_attention', False):
            layers.append(self._create_attention(out_channels))

        return nn.Sequential(*layers)

    def _get_norm_layer(self, norm_type: str, channels: int) -> nn.Module:
        """获取归一化层"""
        if norm_type == 'batch':
            return nn.BatchNorm2d(channels)
        elif norm_type == 'instance':
            return nn.InstanceNorm2d(channels)
        elif norm_type == 'group':
            return nn.GroupNorm(8, channels)
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")

    def _get_activation(self, activation_type: str) -> nn.Module:
        """获取激活函数"""
        if activation_type == 'relu':
            return nn.ReLU(inplace=True)
        elif activation_type == 'leaky_relu':
            return nn.LeakyReLU(0.2, inplace=True)
        elif activation_type == 'silu':
            return nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unknown activation type: {activation_type}")

    def _create_attention(self, channels: int) -> nn.Module:
        """创建注意力模块"""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 初始特征提取
        x = self.input_conv(x)

        # 编码器路径
        skip_connections = []
        for encoder, down in zip(self.encoder_stages, self.downsample):
            x = encoder(x)
            skip_connections.append(x)
            x = down(x)

        # Bottleneck
        x = self.bottleneck(x)

        # 解码器路径
        for decoder, up, skip in zip(self.decoder_stages, self.upsample,
                                     reversed(skip_connections)):
            x = up(x)
            # 处理skip connection
            if isinstance(decoder, dict) and decoder.get('skip_fusion') == 'concat':
                x = torch.cat([x, skip], dim=1)
            else:
                x = x + skip
            x = decoder(x)

        # 最终输出
        return self.output_conv(x)