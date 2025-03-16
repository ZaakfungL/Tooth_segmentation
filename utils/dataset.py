from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import os


class ToothDataset(Dataset):
    def __init__(self, image_dirs, mask_dirs, transform=None):
        self.transform = transform
        self.image_list = []

        # 获取 image_dirs 路径下的所有图像文件
        files = os.listdir(image_dirs)
        # 只获取图像文件
        image_files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.image_list = [os.path.join(image_dirs, f) for f in image_files]

        if len(self.image_list) == 0:
            raise FileNotFoundError(f"No image files found in {image_dirs}")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        mask_path = img_path.replace('Image', 'Mask')

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

def get_dataloaders(image_dirs, mask_dirs, batch_size, val_split=0.2, transform=None):
    dataset = ToothDataset(image_dirs, mask_dirs, transform=transform)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader