"""
Generator model for the inpainting GAN.
"""
import torch
import torch.nn as nn
from .blocks import ConvBlock, DeConvBlock, AttentionBlock

class Generator(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):  # 4 channels: RGB + mask
        super().__init__()
        
        # Encoder
        self.e1 = ConvBlock(in_channels, 64, stride=2)  # 128x128
        self.e2 = ConvBlock(64, 128, stride=2)         # 64x64
        self.e3 = ConvBlock(128, 256, stride=2)        # 32x32
        self.e4 = ConvBlock(256, 512, stride=2)        # 16x16
        
        # Bottleneck with attention
        self.bottleneck = nn.Sequential(
            ConvBlock(512, 512),
            AttentionBlock(512),
            ConvBlock(512, 512)
        )
        
        # Decoder
        self.d4 = DeConvBlock(1024, 256)  # Skip connection from e3
        self.d3 = DeConvBlock(512, 128)   # Skip connection from e2
        self.d2 = DeConvBlock(256, 64)    # Skip connection from e1
        self.d1 = DeConvBlock(128, 32)    # Skip connection from input
        
        # Output layer
        self.out = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x, mask):
        # Combine input image and mask
        x = torch.cat([x, mask], dim=1)
        
        # Encoder
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        
        # Bottleneck
        b = self.bottleneck(e4)
        
        # Decoder with skip connections
        d4 = self.d4(torch.cat([b, e4], dim=1))
        d3 = self.d3(torch.cat([d4, e3], dim=1))
        d2 = self.d2(torch.cat([d3, e2], dim=1))
        d1 = self.d1(torch.cat([d2, e1], dim=1))
        
        return self.out(d1)
