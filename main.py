import os

import argparse
import torch
import torch.nn as nn
from torchvision import transforms
import json
from datetime import datetime

from config import (
    train_params,
    evolution_params,
    zero_cost_params
)
from utils.dataset import get_dataloaders
from evolution.evolution import Evolution, Individual
from models.search_space import SearchSpace
from utils.zero_cost_metrics import ZeroCostEvaluator


class NASRunner:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.search_space = SearchSpace()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Create output directory
        self.output_dir = os.path.join('nas_results', datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize zero-cost evaluator
        self.evaluator = ZeroCostEvaluator(self.device)

    def search_architecture(self):
        """只执行架构搜索阶段"""
        train_loader, val_loader = get_dataloaders(
            image_dirs=[train_params['image_dirs']],
            mask_dirs=[train_params['mask_dirs']],
            batch_size=zero_cost_params['ntk_batch_size'],
            transform=self.transform,
            val_split=train_params['val_split']
        )

        evolution = Evolution(
            population_size=evolution_params['population_size'],
            generations=evolution_params['generations'],
            search_space=self.search_space,
            mutation_rate=evolution_params['mutation_rate'],
            crossover_rate=evolution_params['crossover_rate']
        )

        best_individual = evolution.run(
            train_loader=train_loader,
            val_loader=val_loader,
            device=self.device
        )

        self.save_results(best_individual)
        return best_individual

    def save_results(self, best_individual):
        """保存搜索结果"""
        results = {
            'best_architecture': best_individual.architecture,
            'best_fitness': best_individual.fitness,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # 保存结果到JSON文件
        results_path = os.path.join(self.output_dir, 'search_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"Search results saved to {results_path}")

    def train_best_model(self, best_individual):
        """只执行最终训练阶段"""
        train_loader, val_loader = get_dataloaders(
            image_dirs=[train_params['image_dirs']],
            mask_dirs=[train_params['mask_dirs']],
            batch_size=train_params['batch_size'],  # 使用训练专用的batch_size
            transform=self.transform,
            val_split=train_params['val_split']
        )
        """Train the best model found by NAS"""
        from models.nas_unet import UNet

        # Initialize model
        model = UNet(**best_individual.architecture).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_params['learning_rate'],
            weight_decay=train_params.get('weight_decay', 1e-4)
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        best_val_loss = float('inf')
        best_epoch = 0

        # Training loop
        for epoch in range(train_params['num_epochs']):
            model.train()
            train_loss = 0
            for data, target in train_loader:
                data = data.to(self.device)
                target = target.to(self.device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation loop
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data = data.to(self.device)
                    target = target.to(self.device)
                    output = model(data)
                    loss = criterion(output, target)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'architecture': best_individual.architecture
                }
                torch.save(
                    checkpoint,
                    os.path.join(checkpoint_dir, 'best_model.pth')
                )

            print(f"Epoch {epoch + 1}/{train_params['num_epochs']} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Best Val Loss: {best_val_loss:.4f} (epoch {best_epoch + 1})")


def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--search-only', action='store_true', help='Only perform architecture search')
    parser.add_argument('--train-only', action='store_true', help='Only train the best architecture')
    parser.add_argument('--architecture-path', type=str, help='Path to the saved architecture JSON file')
    args = parser.parse_args()

    runner = NASRunner()

    if args.train_only:
        if not args.architecture_path:
            raise ValueError("Please provide --architecture-path when using --train-only")

        # 从保存的结果加载最佳架构
        with open(args.architecture_path, 'r') as f:
            saved_results = json.load(f)
        best_individual = Individual(
            architecture=saved_results['best_architecture'],
            fitness=saved_results.get('best_fitness')  # 可能不存在
        )
        runner.train_best_model(best_individual)

    elif args.search_only:
        runner.search_architecture()

    else:
        # 运行完整流程
        best_individual = runner.search_architecture()
        runner.train_best_model(best_individual)


if __name__ == "__main__":
    main()