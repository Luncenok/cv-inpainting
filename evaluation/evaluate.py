"""
Evaluation script for the inpainting model.
"""
import torch
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import argparse

from models.generator import Generator
from utils.data_loader import get_data_loaders
from evaluation.metrics import MetricTracker

def evaluate(config):
    # Initialize model
    generator = Generator(in_channels=4, out_channels=3)
    generator = generator.to(config['device'])
    
    # Load checkpoint
    checkpoint = torch.load(config['checkpoint_path'], map_location=config['device'])
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    # Get data loader
    _, val_loader = get_data_loaders(
        config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        debug=config['debug']
    )
    
    # Initialize metric tracker
    metric_tracker = MetricTracker()
    
    # Create directories for saving results
    os.makedirs(config['output_dir'], exist_ok=True)
    real_dir = os.path.join(config['output_dir'], 'real')
    fake_dir = os.path.join(config['output_dir'], 'fake')
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader)):
            real_images = batch['image'].to(config['device'])
            masked_images = batch['masked_image'].to(config['device'])
            masks = batch['mask'].to(config['device'])
            
            # Generate fake images
            fake_images = generator(masked_images, masks)
            
            # Update metrics
            metric_tracker.update(real_images, fake_images)
            
            # Save images
            # for j in range(real_images.size(0)):
            save_image(
                real_images[0],
                os.path.join(real_dir, f'real_{i}.png'),
                normalize=True
            )
            save_image(
                fake_images[0],
                os.path.join(fake_dir, f'fake_{i}.png'),
                normalize=True
            )
    
    # Calculate FID
    metric_tracker.compute_fid(real_dir, fake_dir)
    
    # Get final metrics
    metrics = metric_tracker.get_metrics()
    
    # Plot and save metrics
    plot_metrics(metrics, config['output_dir'])
    
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
    parser = argparse.ArgumentParser(description='Evaluate the inpainting model')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with limited dataset')
    args = parser.parse_args()
    
    config = {
        'checkpoint_path': 'checkpoints/latest.pt',
        'data_dir': 'data/celeba',
        'batch_size': 8,
        'num_workers': 4,
        'output_dir': 'evaluation_results',
        'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
        'debug': args.debug
    }
    
    metrics = evaluate(config)
    print('Evaluation Results:')
    print(f'PSNR: {metrics["psnr"]:.2f}')
    print(f'SSIM: {metrics["ssim"]:.2f}')
    print(f'FID: {metrics["fid"]:.2f}')
