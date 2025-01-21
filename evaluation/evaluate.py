"""
Evaluation script for the inpainting model.
"""
import torch
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from models.generator import Generator
from utils.data_loader import get_data_loaders
from .metrics import MetricTracker

def evaluate(config):
    # Initialize model
    generator = Generator(in_channels=4, out_channels=3)
    
    # Load checkpoint
    checkpoint = torch.load(config['checkpoint_path'])
    generator.load_state_dict(checkpoint['generator_state_dict'])
    
    if torch.cuda.is_available():
        generator = generator.cuda()
    
    generator.eval()
    
    # Get data loader
    _, val_loader = get_data_loaders(
        config['data_dir'],
        batch_size=config['batch_size']
    )
    
    # Initialize metric tracker
    metric_tracker = MetricTracker()
    
    # Create directories for saving results
    os.makedirs(config['results_dir'], exist_ok=True)
    real_dir = os.path.join(config['results_dir'], 'real')
    fake_dir = os.path.join(config['results_dir'], 'fake')
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader)):
            real_images = batch['image'].cuda()
            masked_images = batch['masked_image'].cuda()
            masks = batch['mask'].cuda()
            
            # Generate fake images
            fake_images = generator(masked_images, masks)
            
            # Update metrics
            metric_tracker.update(real_images, fake_images)
            
            # Save images
            for j in range(real_images.size(0)):
                save_image(
                    real_images[j],
                    os.path.join(real_dir, f'real_{i}_{j}.png'),
                    normalize=True
                )
                save_image(
                    fake_images[j],
                    os.path.join(fake_dir, f'fake_{i}_{j}.png'),
                    normalize=True
                )
    
    # Calculate FID
    metric_tracker.compute_fid(real_dir, fake_dir)
    
    # Get final metrics
    metrics = metric_tracker.get_metrics()
    
    # Plot and save metrics
    plot_metrics(metrics, config['results_dir'])
    
    return metrics

def plot_metrics(metrics, save_dir):
    """Plot and save metrics."""
    plt.figure(figsize=(10, 5))
    
    # Bar plot for PSNR and SSIM
    plt.subplot(1, 2, 1)
    plt.bar(['PSNR', 'SSIM'], [metrics['psnr'], metrics['ssim']])
    plt.title('PSNR and SSIM Metrics')
    
    # Text for FID
    plt.subplot(1, 2, 2)
    plt.text(0.5, 0.5, f'FID Score: {metrics["fid"]:.2f}',
             horizontalalignment='center',
             verticalalignment='center',
             fontsize=12)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics.png'))
    plt.close()

if __name__ == '__main__':
    config = {
        'checkpoint_path': 'checkpoints/latest.pth',
        'data_dir': 'data/celeba',
        'results_dir': 'evaluation/results',
        'batch_size': 8
    }
    
    metrics = evaluate(config)
    print('Evaluation Results:')
    print(f'PSNR: {metrics["psnr"]:.2f}')
    print(f'SSIM: {metrics["ssim"]:.2f}')
    print(f'FID: {metrics["fid"]:.2f}')
