"""
Evaluation metrics for the inpainting model.
"""
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from pytorch_fid import fid_score

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images."""
    if torch.is_tensor(img1):
        img1 = img1.cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.cpu().numpy()
    
    # Convert to range [0, 1]
    img1 = (img1 + 1) / 2
    img2 = (img2 + 1) / 2
    
    return peak_signal_noise_ratio(img1, img2)

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images."""
    if torch.is_tensor(img1):
        img1 = img1.cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.cpu().numpy()
    
    # Convert to range [0, 1]
    img1 = (img1 + 1) / 2
    img2 = (img2 + 1) / 2
    
    # Move channels to last dimension for skimage
    if img1.shape[0] == 3:  # If channels first
        img1 = np.transpose(img1, (1, 2, 0))
        img2 = np.transpose(img2, (1, 2, 0))
    
    return structural_similarity(
        img1, img2,
        data_range=1,
        channel_axis=2  # Specify which axis contains the channels
    )

def calculate_fid(path1, path2):
    """Calculate FID between two sets of images."""
    return fid_score.calculate_fid_given_paths(
        [path1, path2],
        batch_size=50,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        dims=2048
    )

class MetricTracker:
    def __init__(self):
        self.metrics = {
            'psnr': [],
            'ssim': [],
            'fid': None
        }
    
    def update(self, real_images, fake_images):
        """Update PSNR and SSIM metrics."""
        for i in range(len(real_images)):
            self.metrics['psnr'].append(
                calculate_psnr(real_images[i], fake_images[i])
            )
            self.metrics['ssim'].append(
                calculate_ssim(real_images[i], fake_images[i])
            )
    
    def compute_fid(self, real_path, fake_path):
        """Compute FID score."""
        self.metrics['fid'] = calculate_fid(real_path, fake_path)
    
    def get_metrics(self):
        """Return average metrics."""
        return {
            'psnr': np.mean(self.metrics['psnr']),
            'ssim': np.mean(self.metrics['ssim']),
            'fid': self.metrics['fid']
        }
