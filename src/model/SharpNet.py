import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.leaky_relu(x + residual)


class SharpNet(nn.Module):
    def __init__(self, channels):
        super(SharpNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # residual blocks
        self.resblock1 = ResidualBlock(64)
        self.resblock2 = ResidualBlock(64)
        self.resblock3 = ResidualBlock(64)
        
        # downsampling & upsampling
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        
        # dilated convolution
        self.dilated_conv = nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2)
        self.bn_dilated = nn.BatchNorm2d(64)
        
        # final convolution
        self.conv2 = nn.Conv2d(64, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = F.leaky_relu(self.bn1(self.conv1(x)))
        
        x = self.resblock1(x1)
        x = self.resblock2(x)
        x = self.downsample(x)
        
        x = F.leaky_relu(self.bn_dilated(self.dilated_conv(x)))
        x = self.upsample(x)
        
        x = self.resblock3(x)
        x = x + x1  # skip connection from initial input
        output = self.conv2(x)
        
        return output
