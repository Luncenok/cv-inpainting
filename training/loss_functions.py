"""
Loss functions for training the inpainting GAN.
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

class GANLoss(nn.Module):
    def __init__(self, gan_mode='vanilla', target_real_label=1.0, target_fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
    
    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input."""
        target_tensor = self.real_label if target_is_real else self.fake_label
        target_tensor = target_tensor.expand(prediction.shape[0], 1)
        return target_tensor.to(prediction.device).contiguous()
    
    def forward(self, prediction, target_is_real):
        """Calculate GAN loss.
        
        Args:
            prediction (torch.Tensor): Discriminator output, shape [batch_size, 1]
            target_is_real (bool): Whether target should be real (1) or fake (0)
            
        Returns:
            torch.Tensor: Scalar loss value
        """
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # Register normalization parameters as buffers
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))
    
    def normalize(self, x):
        """Normalize input images for VGG."""
        # Convert from [-1, 1] to [0, 1]
        x = (x + 1) / 2
        # Normalize using ImageNet statistics
        x = (x - self.mean) / self.std
        return x.contiguous()
    
    def extract_features(self, x):
        """Extract VGG features and ensure they're properly handled."""
        features = []
        x = x.contiguous()
        
        for layer in self.feature_extractor:
            x = layer(x)
            if isinstance(layer, nn.MaxPool2d):
                x = x.contiguous()
        
        return x.contiguous()
    
    def forward(self, x, y):
        """Calculate perceptual loss between two images.
        
        Args:
            x (torch.Tensor): First image tensor [batch_size, channels, height, width]
            y (torch.Tensor): Second image tensor [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Scalar loss value
        """
        
        # Normalize inputs
        x = self.normalize(x)
        y = self.normalize(y)
        
        # Extract features with proper handling
        features_x = self.extract_features(x)
        features_y = self.extract_features(y)
        return nn.functional.l1_loss(features_x, features_y)

def get_loss_functions():
    """Return dictionary of loss functions."""
    return {
        'adversarial': GANLoss(),
        'perceptual': PerceptualLoss(),
        'l1': nn.L1Loss()
    }
