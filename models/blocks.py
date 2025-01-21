"""
Basic building blocks for the GAN architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        """Self-attention mechanism using unfold/fold operations.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Output tensor with same shape as input
        """
        
        batch_size, C, H, W = x.size()
        expected_shape = (batch_size, C, H, W)
        
        # Compute query, key, value tensors
        query = self.query_conv(x)  # [B, C//8, H, W]
        key = self.key_conv(x)      # [B, C//8, H, W]
        value = self.value_conv(x)  # [B, C, H, W]
        
        # Unfold the spatial dimensions
        query_unf = F.unfold(query, kernel_size=1, padding=0)  # [B, C//8, H*W]
        key_unf = F.unfold(key, kernel_size=1, padding=0)      # [B, C//8, H*W]
        value_unf = F.unfold(value, kernel_size=1, padding=0)  # [B, C, H*W]
        
        # Compute attention scores
        query_unf = query_unf.transpose(1, 2)  # [B, H*W, C//8]
        key_unf = key_unf.transpose(1, 2)      # [B, H*W, C//8]
        
        attention = torch.bmm(query_unf, key_unf.transpose(1, 2))  # [B, H*W, H*W]
        attention = self.softmax(attention)
        
        # Apply attention to values
        value_unf = value_unf.transpose(1, 2)  # [B, H*W, C]
        out_unf = torch.bmm(attention, value_unf)  # [B, H*W, C]
        out_unf = out_unf.transpose(1, 2)  # [B, C, H*W]
        
        # Fold back to spatial dimensions
        out = F.fold(out_unf, output_size=(H, W), kernel_size=1, padding=0)  # [B, C, H, W]

        if out.shape != expected_shape:
            raise ValueError(f"Output shape {out.shape} does not match expected shape {expected_shape}")
        
        # Apply residual connection
        out = self.gamma * out + x
        
        # Final shape check
        if out.shape != x.shape:
            raise ValueError(f"Final output shape {out.shape} does not match input shape {x.shape}")
        
        return out
