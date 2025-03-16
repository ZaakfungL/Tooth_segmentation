import torch
import torch.nn as nn
import numpy as np
from typing import Dict
from torch.autograd.functional import jacobian
import torch.nn.functional as F


class ZeroCostEvaluator:
    def __init__(self, device='cuda'):
        self.device = device

    def compute_ntk(self, model: nn.Module, input_shape: tuple, batch_size: int = 8):
        """计算神经切线核"""
        model.eval()

        # 生成随机输入
        x = torch.randn((batch_size,) + input_shape).to(self.device)
        x.requires_grad_(True)

        # 获取模型参数
        params = []
        names = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params.append(param)
                names.append(name)

        def get_jacobian(x):
            """计算雅可比矩阵"""
            output = model(x)
            jac = torch.zeros(output.shape[0], output.numel() // output.shape[0],
                              sum(p.numel() for p in params)).to(self.device)

            for i in range(output.numel()):
                model.zero_grad()
                scalar = torch.zeros_like(output)
                scalar.view(-1)[i] = 1
                output.backward(scalar, retain_graph=True)

                col_idx = 0
                for param in params:
                    if param.grad is not None:
                        jac[:, i // output.shape[0], col_idx:col_idx + param.grad.numel()] = \
                            param.grad.view(-1)
                        col_idx += param.grad.numel()

            return jac

        # 计算神经切线核
        J = get_jacobian(x)
        K = torch.bmm(J, J.transpose(1, 2))

        # 计算核矩阵的条件数
        eigenvals = torch.linalg.eigvalsh(K)
        condition_number = eigenvals.max() / eigenvals.min()

        return {
            'ntk_score': condition_number.item(),
            'ntk_trace': torch.trace(K).item() / batch_size
        }

    def compute_linear_regions(self, model: nn.Module, input_shape: tuple, num_samples: int = 100):
        """估计线性区域数量"""
        model.eval()

        def count_linear_regions(x1, x2):
            """在两点之间采样并计算激活模式变化"""
            points = torch.linspace(0, 1, steps=10).to(self.device)
            interpolated = torch.stack([x1 + (x2 - x1) * t for t in points])

            with torch.no_grad():
                features = []
                for p in interpolated:
                    # 收集所有ReLU层的激活状态
                    activation_patterns = []

                    def hook_fn(module, input, output):
                        if isinstance(module, nn.ReLU):
                            activation_patterns.append((output > 0).float())

                    handles = []
                    for module in model.modules():
                        if isinstance(module, nn.ReLU):
                            handles.append(module.register_forward_hook(hook_fn))

                    _ = model(p.unsqueeze(0))

                    for handle in handles:
                        handle.remove()

                    # 将所有激活模式连接成一个向量
                    pattern = torch.cat([p.flatten() for p in activation_patterns])
                    features.append(pattern)

            # 计算相邻点之间激活模式的变化
            features = torch.stack(features)
            changes = (features[1:] != features[:-1]).float().sum(1)
            return (changes > 0).float().sum().item()

        # 生成随机样本对
        x = torch.randn((num_samples,) + input_shape).to(self.device)
        total_regions = 0

        # 计算每对样本之间的线性区域
        for i in range(num_samples - 1):
            total_regions += count_linear_regions(x[i], x[i + 1])

        return {
            'linear_regions': total_regions / (num_samples - 1)
        }

    def evaluate_architecture(self, model: nn.Module, input_shape: tuple) -> Dict[str, float]:
        """综合评估架构"""
        ntk_metrics = self.compute_ntk(model, input_shape)
        linear_region_metrics = self.compute_linear_regions(model, input_shape)

        # 组合评估指标
        score = (1 / ntk_metrics['ntk_score']) * linear_region_metrics['linear_regions']

        return {
            'zero_cost_score': score,
            **ntk_metrics,
            **linear_region_metrics
        }