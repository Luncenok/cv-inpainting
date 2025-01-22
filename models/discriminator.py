"""
Discriminator model for the inpainting GAN.
"""
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        """Initialize discriminator.
        
        Args:
            in_channels (int): Number of input channels. Should be 4 for RGB + mask.
                             Default is 3 for backward compatibility.
        """
        super().__init__()
        self.model = nn.Sequential(
            # Input: [batch_size, in_channels, 256, 256]
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),  # [batch_size, 64, 128, 128]
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # [batch_size, 128, 64, 64]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # [batch_size, 256, 32, 32]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # [batch_size, 512, 16, 16]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1),  # [batch_size, 1, 8, 8]
        )
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, channels, height, width].
                            Expects 4 channels: RGB + mask
            
        Returns:
            torch.Tensor: Discriminator output [batch_size, 1]
        """
        
        # Get model output and ensure it's contiguous
        out = self.model(x)  # [batch_size, 1, H', W']
        out = out.contiguous()
        
        # Global average pooling instead of reshape
        out = out.mean(dim=[2, 3], keepdim=True)  # [batch_size, 1, 1, 1]
        out = out.reshape(out.size(0), 1)  # [batch_size, 1]
        
        return out.contiguous()  # Ensure final output is contiguous
