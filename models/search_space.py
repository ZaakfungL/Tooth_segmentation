# models/search_space.py
from typing import Dict, Any, List
import random
import json
import torch.nn as nn
from config import search_space_config as cfg


class SearchSpace:
    def __init__(self):
        self.encoder_options = cfg['encoder_block']
        self.decoder_options = cfg['decoder_block']
        self.network_options = cfg['network']

    def sample_block_operations(self, is_encoder: bool = True) -> Dict[str, Any]:
        """采样block的操作序列和其他特性，不包含通道数"""
        options = self.encoder_options if is_encoder else self.decoder_options
        num_ops = random.choice(options['num_ops_per_block'])

        block = {
            'operations': [],
            'activation': random.choice(options['activation']),
            'normalization': random.choice(options['normalization']),
            'use_attention': random.choice(options['use_attention'])
        }

        # 添加特定选项
        if is_encoder:
            block['skip_connection'] = random.choice(options['skip_connection'])
        else:
            block['skip_fusion'] = random.choice(options['skip_fusion'])

        # 采样操作序列
        for _ in range(num_ops):
            op = random.choice(options['operations'])
            block['operations'].append(op)

        return block

    def assign_channels(self, block: Dict[str, Any], in_channels: int, out_channels: int,
                        is_encoder: bool = True) -> Dict[str, Any]:
        """为已有操作序列分配通道数，确保通道数渐进变化"""
        block = block.copy()
        num_ops = len(block['operations'])
        block['in_channels'] = in_channels
        block['out_channels'] = out_channels
        block['channels'] = []

        if num_ops == 1:
            # 只有一个操作时直接连接输入输出
            block['channels'].append((in_channels, out_channels))
            return block

        # 计算每步通道数的变化量
        if is_encoder:
            # 编码器：通道数递增
            step = (out_channels - in_channels) / (num_ops - 1)
        else:
            # 解码器：通道数递减
            step = (out_channels - in_channels) / (num_ops - 1)

        current_channels = in_channels
        for i in range(num_ops):
            if i == num_ops - 1:
                # 最后一个操作使用目标通道数
                next_channels = out_channels
            else:
                # 计算下一个操作的通道数
                next_channels = int(in_channels + step * (i + 1))
                # 确保通道数是8的倍数
                next_channels = max(8, (next_channels + 7) // 8 * 8)

            block['channels'].append((int(current_channels), next_channels))
            current_channels = next_channels

        return block

    def sample_architecture(self) -> Dict[str, Any]:
        architecture = {
            'initial_channels': random.choice(self.network_options['initial_channels']),
            'channel_multiplier': random.choice(self.network_options['channel_multiplier']),
            'num_stages': self.network_options['num_stages']
        }

        # 采样基本块
        encoder_base_block = self.sample_block_operations(is_encoder=True)
        decoder_base_block = self.sample_block_operations(is_encoder=False)
        bottleneck_block = self.sample_block_operations(is_encoder=True)

        # 计算每个阶段的通道数
        current_channels = architecture['initial_channels']
        encoder_channels = []

        # 构建编码器blocks
        architecture['encoder_blocks'] = []
        for _ in range(architecture['num_stages']):
            out_channels = current_channels * architecture['channel_multiplier']
            stage_block = self.assign_channels(
                encoder_base_block.copy(),
                current_channels,
                out_channels,
                is_encoder=True
            )
            architecture['encoder_blocks'].append(stage_block)
            encoder_channels.append(out_channels)
            current_channels = out_channels

        # Bottleneck使用与最后一个编码器stage相同的通道数
        architecture['bottleneck'] = self.assign_channels(
            bottleneck_block,
            current_channels,
            current_channels,  # 输出通道数与输入相同
            is_encoder=True
        )

        # 构建解码器blocks
        architecture['decoder_blocks'] = []
        for i in range(architecture['num_stages']):
            out_channels = encoder_channels[-i - 1]
            stage_block = self.assign_channels(
                decoder_base_block.copy(),
                current_channels * 2,  # 因为有skip connection
                out_channels,
                is_encoder=False
            )
            architecture['decoder_blocks'].append(stage_block)
            current_channels = out_channels

        print("\nGenerated architecture:")
        print(f"Encoder base block: {encoder_base_block}")
        print(f"Decoder base block: {decoder_base_block}")
        print(f"Channel progression: {encoder_channels}")
        print("Full architecture structure:")
        print(f"- Initial channels: {architecture['initial_channels']}")
        print(f"- Channel multiplier: {architecture['channel_multiplier']}")
        print(f"- Number of stages: {architecture['num_stages']}")
        print(f"- Bottleneck channels: {current_channels}")

        return architecture

    def mutate_architecture(self, architecture: Dict[str, Any], mutation_rate: float = 0.1) -> Dict[str, Any]:
        """变异一个现有架构，只变异基本块的结构"""
        if random.random() < mutation_rate:
            # 重新采样一个编码器基本块
            encoder_base_block = self.sample_block_operations(is_encoder=True)
            # 更新所有编码器阶段
            for i, block in enumerate(architecture['encoder_blocks']):
                new_block = self.assign_channels(
                    encoder_base_block.copy(),
                    block['in_channels'],
                    block['out_channels'],
                    is_encoder=True
                )
                architecture['encoder_blocks'][i] = new_block

        if random.random() < mutation_rate:
            # 重新采样一个解码器基本块
            decoder_base_block = self.sample_block_operations(is_encoder=False)
            # 更新所有解码器阶段
            for i, block in enumerate(architecture['decoder_blocks']):
                new_block = self.assign_channels(
                    decoder_base_block.copy(),
                    block['in_channels'],
                    block['out_channels'],
                    is_encoder=False
                )
                architecture['decoder_blocks'][i] = new_block

        if random.random() < mutation_rate:
            # 变异bottleneck
            architecture['bottleneck'] = self.sample_block_operations(is_encoder=True)
            architecture['bottleneck'] = self.assign_channels(
                architecture['bottleneck'],
                architecture['bottleneck']['in_channels'],
                architecture['bottleneck']['out_channels'],
                is_encoder=True
            )

        return architecture

    def crossover_architecture(self, arch1: Dict[str, Any], arch2: Dict[str, Any]) -> Dict[str, Any]:
        """对两个架构进行交叉操作"""
        child = {
            'initial_channels': arch1['initial_channels'],
            'encoder_blocks': [],
            'decoder_blocks': [],
            'bottleneck': None
        }

        # 交叉编码器blocks
        for block1, block2 in zip(arch1['encoder_blocks'], arch2['encoder_blocks']):
            child['encoder_blocks'].append(
                block1 if random.random() < 0.5 else block2
            )

        # 交叉bottleneck
        child['bottleneck'] = (
            arch1['bottleneck'] if random.random() < 0.5
            else arch2['bottleneck']
        )

        # 交叉解码器blocks
        for block1, block2 in zip(arch1['decoder_blocks'], arch2['decoder_blocks']):
            child['decoder_blocks'].append(
                block1 if random.random() < 0.5 else block2
            )

        return child