import torch.nn as nn
import torch.nn.functional as F

class BetterNet(nn.Module):
    def __init__(self, input_channels, middle_channels, num_middle_layers):
        super(BetterNet, self).__init__()

        self.blocks = nn.ModuleList()
        self.blocks.append(
            nn.Sequential(
                nn.Conv2d(in_channels=input_channels, out_channels=middle_channels, kernel_size=3, padding=1), 
                nn.BatchNorm2d(middle_channels),
                nn.LeakyReLU()
            )
        )
        for _ in range(num_middle_layers): 
            block = nn.Sequential(
                nn.Conv2d(in_channels=middle_channels, out_channels=middle_channels, kernel_size=3, padding=1), 
                nn.BatchNorm2d(middle_channels),
                nn.LeakyReLU()
            )
            self.blocks.append(block)

    def forward(self, x):
        input_residual = x
        for block in self.blocks: 
            x = block(x)
        
        output = x + input_residual
        
        return output