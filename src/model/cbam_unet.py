import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module): 
    def __init__(self, num_features): 
        super(LayerNorm2d, self).__init__()
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, x): 
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2)
        return x.to(x.dtype)

class ChannelAttention(nn.Module): 
    def __init__(self, channels, reduction_ratio=8): 
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True), 
            nn.Linear(channels // reduction_ratio, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x): 
        batch_size, channels, _, _ = x.size() 
        avg_out = self.avg_pool(x).view(batch_size, channels)
        max_out = self.max_pool(x).view(batch_size, channels)
        
        avg_out = self.mlp(avg_out)
        max_out = self.mlp(max_out)

        out = avg_out + max_out
        out = self.sigmoid(out).view(batch_size, channels, 1, 1)
        return (x * out).to(x.dtype)
    
class SpatialAttention(nn.Module): 
    def __init__(self, kernel_size=7): 
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): 
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(concat)
        out = self.sigmoid(out)
        return (x * out).to(x.dtype)


class CBAMResidualBlock(nn.Module): 
    def __init__(self, channels, dw_expand=1, ffn_expand=2, dropout_rate=0.): 
        super(CBAMResidualBlock, self).__init__()
        dw_channels = channels * dw_expand
        ffn_channels = channels * ffn_expand

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(channels, dw_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(dw_channels, dw_channels, kernel_size=3, stride=1, padding=1, groups=dw_channels, bias=True),
            nn.GELU(),
            nn.Conv2d(dw_channels, channels, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.channel_attention = ChannelAttention(channels)

        self.ffn = nn.Sequential(
            nn.Conv2d(channels, ffn_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(ffn_channels, channels, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.layer_norm1 = LayerNorm2d(channels)
        self.layer_norm2 = LayerNorm2d(channels)

        self.dropout1 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1), requires_grad=True)

    def forward(self, x): 
        residual = x
        x = self.layer_norm1(x)
        x = self.depthwise_conv(x)
        x = self.channel_attention(x)
        x = self.dropout1(x)
        x = residual + x * self.beta

        residual = x
        x = self.layer_norm2(x)
        x = self.ffn(x)
        x = self.dropout2(x)
        x = residual + x * self.gamma

        return x
    

class UNetWithCBAM(nn.Module): 
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], dw_expand=1, ffn_expand=2, dropout_rate=0.):
        super(UNetWithCBAM, self).__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[CBAMResidualBlock(chan, dw_expand, ffn_expand, dropout_rate) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, kernel_size=2, stride=2)
            )
            chan *= 2

        self.middle_blks = nn.Sequential(
            *[CBAMResidualBlock(chan, dw_expand, ffn_expand, dropout_rate) for _ in range(middle_blk_num)]
        )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, kernel_size=1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan //= 2
            self.decoders.append(
                nn.Sequential(
                    *[CBAMResidualBlock(chan, dw_expand, ffn_expand, dropout_rate) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
    
