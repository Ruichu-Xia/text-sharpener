import torch.nn as nn
import torch.nn.functional as F

class BetterNet(nn.Module):
    def __init__(self, channels_list):
        super(BetterNet, self).__init__()

        self.blocks = nn.ModuleList()
        self.input_channel = channels_list[0]

        for in_channels, out_channels in zip(channels_list[:-1], channels_list[1:]): 
            block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1), 
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU()
            )
            self.blocks.append(block)

    def forward(self, x):
        input_residual = x
        for block in self.blocks: 
            x = block(x)
        
        output = x + input_residual
        
        return output