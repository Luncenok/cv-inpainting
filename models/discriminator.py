"""
PatchGAN discriminator for the inpainting GAN.
"""
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        
        def discriminator_block(in_channels, out_channels, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),  # 128x128
            *discriminator_block(64, 128),                          # 64x64
            *discriminator_block(128, 256),                         # 32x32
            *discriminator_block(256, 512),                         # 16x16
            nn.Conv2d(512, 1, 4, padding=1)                        # 16x16 -> 1 channel
        )
    
    def forward(self, x):
        return self.model(x)
