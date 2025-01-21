"""
Loss functions for training the inpainting GAN.
"""
import torch
import torch.nn as nn
import torchvision.models as models

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
    
    def forward(self, prediction, target_is_real):
        target_tensor = self.real_label if target_is_real else self.fake_label
        target_tensor = target_tensor.expand_as(prediction)
        return self.loss(prediction, target_tensor)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def forward(self, x, y):
        features_x = self.feature_extractor(x)
        features_y = self.feature_extractor(y)
        return nn.functional.l1_loss(features_x, features_y)

def get_loss_functions():
    """Return dictionary of loss functions."""
    return {
        'adversarial': GANLoss(),
        'perceptual': PerceptualLoss(),
        'l1': nn.L1Loss()
    }
