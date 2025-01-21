"""
Basic building blocks for the GAN architecture.
"""
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class DeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.deconv(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.query_conv = nn.Conv2d(channels, channels//8, 1)
        self.key_conv = nn.Conv2d(channels, channels//8, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        # Query
        proj_query = self.query_conv(x).view(batch_size, -1, width*height).permute(0, 2, 1)
        # Key
        proj_key = self.key_conv(x).view(batch_size, -1, width*height)
        # Value
        proj_value = self.value_conv(x).view(batch_size, -1, width*height)
        
        # Attention Map
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        
        # Output
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        
        return self.gamma*out + x
