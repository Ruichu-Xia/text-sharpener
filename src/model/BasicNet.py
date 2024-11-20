import torch.nn as nn
import torch.nn.functional as F

class BasicNet(nn.Module):
    def __init__(self, channels):
        super(BasicNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=3, padding=1)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        
        self.residual_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=channels, kernel_size=3, padding=1)
       
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x1 = F.leaky_relu(self.bn1(self.conv1(x)))
        
        x2 = F.leaky_relu(self.bn2(self.conv2(x1)))
        
        x3 = F.leaky_relu(self.bn3(self.conv3(x2)))
        
        x3 = self.dropout(x3)  
        x3 = x3 + self.residual_conv(x1)
        
        output = self.conv4(x3)
        
        return output