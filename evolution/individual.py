import torch
import torch.nn as nn
from typing import Dict, Any
from models.nas_unet import NASUNet


class Individual:
    def __init__(self, architecture: Dict[str, Any]):
        self.architecture = architecture
        self.fitness = None
        self.model = None
        self.zero_cost_metrics = None

    def evaluate(self, train_loader, val_loader, device):
        """仅使用零成本方法评估个体的适应度"""
        if self.fitness is not None:
            return self.fitness

        # 创建模型
        self.model = NASUNet(**self.architecture).to(device)

        # 使用零成本评估
        from utils.zero_cost_metrics import ZeroCostEvaluator
        evaluator = ZeroCostEvaluator(device)
        self.zero_cost_metrics = evaluator.evaluate_architecture(
            self.model,
            input_shape=(1, 256, 256)
        )

        # 直接使用零成本评估分数作为适应度
        self.fitness = self.zero_cost_metrics['zero_cost_score']

        # 释放显存
        del self.model
        torch.cuda.empty_cache()

        return self.fitness