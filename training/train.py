"""
Training script for the inpainting GAN.
"""
import torch
import torch.optim as optim
from tqdm import tqdm
import mlflow
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import argparse

from models.generator import Generator
from models.discriminator import Discriminator
from utils.data_loader import get_data_loaders
from training.loss_functions import get_loss_functions
from training.config import TrainingConfig

def validate_tensor_shapes(name, tensor, expected_shape=None):
    """Validate tensor shapes and memory layout."""
    
    if expected_shape and tensor.shape != expected_shape:
        print(f"\n=== Validating {name} ===")
        print(f"- Shape: {tensor.shape}")
        print(f"- Stride: {tensor.stride()}")
        print(f"- Expected shape: {expected_shape if expected_shape else 'not specified'}")
        print(f"- Is contiguous: {tensor.is_contiguous()}")
        print(f"- Device: {tensor.device}")
        print(f"- Dtype: {tensor.dtype}")
        raise ValueError(f"Invalid shape for {name}. Expected {expected_shape}, got {tensor.shape}")
    if not tensor.is_contiguous():
        print(f"Warning: {name} is not contiguous, making it contiguous...")
        tensor = tensor.contiguous()
    return tensor

def validate(generator, val_loader, loss_functions, writer, epoch, device):
    """Validate the model."""
    generator.eval()
    val_l1_loss = 0
    val_percep_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            real_images = batch['image'].to(device)
            masked_images = batch['masked_image'].to(device)
            masks = batch['mask'].to(device)
            
            # Generate images
            fake_images = generator(masked_images, masks)
            
            # Calculate losses
            l1_loss = loss_functions['l1'](fake_images, real_images)
            percep_loss = loss_functions['perceptual'](fake_images, real_images)
            
            val_l1_loss += l1_loss.item()
            val_percep_loss += percep_loss.item()
    
    # Average losses
    val_l1_loss /= len(val_loader)
    val_percep_loss /= len(val_loader)
    
    # Log validation losses
    writer.add_scalar('val_l1_loss', val_l1_loss, epoch)
    writer.add_scalar('val_perceptual_loss', val_percep_loss, epoch)
    
    return val_l1_loss, val_percep_loss

def train(config):
    """Training function."""
    # Set random seed for reproducibility
    torch.manual_seed(config.seed)
    device = config.device
    
    # Set up MLflow and tensorboard directories
    os.makedirs('mlruns', exist_ok=True)
    os.makedirs('runs', exist_ok=True)
    
    # Set up MLflow
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)
    
    # Create data loaders
    train_loader, val_loader = get_data_loaders(
        config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        debug=config.debug,
        debug_images_count=config.debug_images_count
    )
    
    try:
        # Initialize models
        generator = Generator(config.in_channels, config.out_channels).to(config.device)
        discriminator = Discriminator(config.in_channels).to(config.device)
        
        # Initialize optimizers
        g_optimizer = optim.Adam(generator.parameters(), lr=config.g_lr, betas=(config.beta1, config.beta2))
        d_optimizer = optim.Adam(discriminator.parameters(), lr=config.d_lr, betas=(config.beta1, config.beta2))
        
        # Get loss functions and move them to device
        loss_functions = get_loss_functions()
        for name, loss_fn in loss_functions.items():
            if isinstance(loss_fn, torch.nn.Module):
                loss_functions[name] = loss_fn.to(device)
        
        # Create tensorboard writer
        writer = SummaryWriter(log_dir=f'runs/inpainting_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        
        # Start MLflow run
        with mlflow.start_run(run_name=config.mlflow_run_name):
            # Log parameters
            mlflow.log_params({
                'batch_size': config.batch_size,
                'g_lr': config.g_lr,
                'd_lr': config.d_lr,
                'num_epochs': config.get_debug_epochs(),
                'debug': config.debug,
                'debug_images_count': config.debug_images_count if config.debug else None,
                'lambda_l1': config.lambda_l1,
                'lambda_perceptual': config.lambda_perceptual
            })
            
            for epoch in range(config.get_debug_epochs()):
                generator.train()
                discriminator.train()
                
                train_g_loss = 0
                train_d_loss = 0
                
                progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.get_debug_epochs()}')
                for batch in progress_bar:
                    real_images = batch['image'].to(device)
                    masked_images = batch['masked_image'].to(device)
                    masks = batch['mask'].to(device)
                    
                    # Train discriminator
                    d_optimizer.zero_grad()
                    
                    # Generate fake images
                    fake_images = generator(masked_images, masks)
                    fake_images = validate_tensor_shapes("Generator output", fake_images, real_images.shape)
                    
                    # Prepare inputs for discriminator (concatenate with mask)
                    real_with_mask = torch.cat([real_images, masks], dim=1)
                    fake_with_mask = torch.cat([fake_images.detach(), masks], dim=1)
                    
                    # Get predictions
                    real_pred = discriminator(real_with_mask)
                    fake_pred = discriminator(fake_with_mask)
                    
                    # Calculate discriminator loss
                    d_real_loss = loss_functions['adversarial'](real_pred, True)
                    d_fake_loss = loss_functions['adversarial'](fake_pred, False)
                    d_loss = (d_real_loss + d_fake_loss) / 2
                    
                    # Backpropagate and update discriminator
                    d_loss.backward()
                    d_optimizer.step()
                    
                    train_d_loss += d_loss.item()
                    
                    # Train generator
                    g_optimizer.zero_grad()
                    
                    # Get new predictions for generator (need new forward pass)
                    fake_with_mask = torch.cat([fake_images, masks], dim=1)
                    fake_pred = discriminator(fake_with_mask)
                    
                    # Calculate generator losses
                    g_adv_loss = loss_functions['adversarial'](fake_pred, True)
                    g_l1_loss = loss_functions['l1'](fake_images, real_images) * config.lambda_l1
                    g_percep_loss = loss_functions['perceptual'](fake_images, real_images) * config.lambda_perceptual
                    
                    g_loss = g_adv_loss + g_l1_loss + g_percep_loss
                    
                    # Backpropagate and update generator
                    g_loss.backward()
                    g_optimizer.step()
                    
                    # Update progress bar
                    train_g_loss += g_loss.item()
                    progress_bar.set_postfix({
                        'g_loss': g_loss.item(),
                        'd_loss': d_loss.item()
                    })
                
                # Average training losses
                train_g_loss /= len(train_loader)
                train_d_loss /= len(train_loader)
                
                # Log training losses
                writer.add_scalar('train_generator_loss', train_g_loss, epoch)
                writer.add_scalar('train_discriminator_loss', train_d_loss, epoch)
                
                # Log metrics
                mlflow.log_metrics({
                    'train_generator_loss': train_g_loss,
                    'train_discriminator_loss': train_d_loss
                }, step=epoch)
                
                # Validate
                val_l1_loss, val_percep_loss = validate(generator, val_loader, loss_functions, writer, epoch, device)
                
                # Log validation metrics
                mlflow.log_metrics({
                    'val_l1_loss': val_l1_loss,
                    'val_perceptual_loss': val_percep_loss
                }, step=epoch)
                
                # Save checkpoint
                if (epoch + 1) % config.save_freq == 0:
                    checkpoint = {
                        'epoch': epoch,
                        'generator_state_dict': generator.state_dict(),
                        'discriminator_state_dict': discriminator.state_dict(),
                        'g_optimizer_state_dict': g_optimizer.state_dict(),
                        'd_optimizer_state_dict': d_optimizer.state_dict(),
                    }
                    save_checkpoint(checkpoint, config.checkpoint_dir, epoch)
        
        writer.close()
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise e

def save_checkpoint(state, checkpoint_dir, epoch):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(state, path)
    
    # Also save as latest
    latest_path = os.path.join(checkpoint_dir, 'latest.pt')
    torch.save(state, latest_path)
    
    return path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the inpainting model')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with limited dataset')
    args = parser.parse_args()
    
    config = TrainingConfig()
    config.debug = args.debug
    config.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    train(config)
