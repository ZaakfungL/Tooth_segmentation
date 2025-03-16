import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class UNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.convd1 = DoubleConv(1, 64)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.convd2 = DoubleConv(64, 128)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.convd3 = DoubleConv(128, 256)
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.convd4 = DoubleConv(256, 512)
        self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(512, 1024)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.convu1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.convu2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.convu3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.convu4 = DoubleConv(128, 64)
        self.outconv = nn.Conv2d(64, num_classes, kernel_size=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.convd1(x)
        x = self.mp1(x1)
        x2 = self.convd2(x)
        x = self.mp2(x2)
        x3 = self.convd3(x)
        x = self.mp3(x3)
        x4 = self.convd4(x)
        x = self.mp4(x4)
        x = self.bottleneck(x)
        x = self.up1(x)
        x = torch.cat([x, x4], dim=1)
        x = self.convu1(x)
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.convu2(x)
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.convu3(x)
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.convu4(x)
        x = self.outconv(x)
        return x
