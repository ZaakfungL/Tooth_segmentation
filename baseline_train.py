import os
import torch
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import glob
from config import train_params

from utils.dataset import get_dataloaders
from utils.losses import CustomLoss
from utils.metrics import dice_metric, iou_metric, hausdorff_distance
from models.model_factory import select_model


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device,
                checkpoint_dir, use_pretrained, model_name, pretrained_file):
    """
    训练模型的主函数
    """
    # 加载预训练权重
    if use_pretrained and os.path.exists(pretrained_file):
        model.load_state_dict(torch.load(pretrained_file))
        print(f"Loaded pretrained weights from {pretrained_file}")

    model = model.to(device)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_dice = 0
        train_iou = 0
        # 跟踪最佳模型文件
        best_model_path = None

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Training]')
        for images, masks in train_pbar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            # 计算指标
            train_loss += loss.item()
            train_dice += dice_metric(outputs, masks).item()
            train_iou += iou_metric(outputs, masks).item()

            # 更新进度条
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice_metric(outputs, masks).item():.4f}',
                'iou': f'{iou_metric(outputs, masks).item():.4f}'
            })

        # 计算训练集平均指标
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        train_iou /= len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0
        val_dice = 0
        val_iou = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Validation]')
            for images, masks in val_pbar:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                # 计算指标
                val_loss += loss.item()
                val_dice += dice_metric(outputs, masks).item()
                val_iou += iou_metric(outputs, masks).item()

                # 更新进度条
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'dice': f'{dice_metric(outputs, masks).item():.4f}',
                    'iou': f'{iou_metric(outputs, masks).item():.4f}'
                })

        # 计算验证集平均指标
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)

        # 打印epoch结果
        print(f'\nEpoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, Train IoU: {train_iou:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            # 删除之前的最佳模型文件
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)

            # 使用新格式创建文件名
            best_model_path = os.path.join(
                checkpoint_dir,
                f'{model_name}_dice_{val_dice:.4f}_iou_{val_iou:.4f}.pth'
            )

            # 保存新的最佳模型
            torch.save(model.state_dict(), best_model_path)
            print(f'Saved best model to {best_model_path}')

    return model

if __name__ == "__main__":
    os.makedirs(train_params['checkpoint_dir'], exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_loader, val_loader = get_dataloaders(
        train_params['image_dirs'],
        train_params['mask_dirs'],
        train_params['batch_size'],
        train_params['val_split'],
        transform
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = select_model(train_params['model_name'], in_channels=3, classes=1)
    criterion = CustomLoss()
    optimizer = optim.AdamW(model.parameters(), lr=train_params['learning_rate'])

    train_model(model, train_loader, val_loader, criterion, optimizer,
                train_params['num_epochs'], device,
                train_params['checkpoint_dir'], train_params['use_pretrained'],
                train_params['model_name'], train_params['pretrained_file'])
