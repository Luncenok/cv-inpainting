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

from models.generator import Generator
from models.discriminator import Discriminator
from utils.data_loader import get_data_loaders
from .loss_functions import get_loss_functions
from .config import TrainingConfig

def train(config: TrainingConfig):
    # Set up MLflow
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)
    
    # Start MLflow run
    with mlflow.start_run(run_name=config.mlflow_run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_params(config.to_dict())
        
        # Initialize models
        generator = Generator(in_channels=config.in_channels, out_channels=config.out_channels)
        discriminator = Discriminator(in_channels=config.out_channels)
        
        if torch.cuda.is_available():
            generator = generator.cuda()
            discriminator = discriminator.cuda()
        
        # Get data loaders
        train_loader, val_loader = get_data_loaders(
            config.data_dir,
            batch_size=config.batch_size,
            num_workers=config.num_workers
        )
        
        # Initialize optimizers
        g_optimizer = optim.Adam(
            generator.parameters(),
            lr=config.g_lr,
            betas=(config.beta1, config.beta2)
        )
        d_optimizer = optim.Adam(
            discriminator.parameters(),
            lr=config.d_lr,
            betas=(config.beta1, config.beta2)
        )
        
        # Initialize loss functions
        loss_functions = get_loss_functions()
        
        # Initialize tensorboard
        writer = SummaryWriter(os.path.join(config.mlflow_tracking_uri, 'runs'))
        
        # Training loop
        for epoch in range(config.num_epochs):
            generator.train()
            discriminator.train()
            
            # Training metrics for this epoch
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            num_batches = 0
            
            with tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}') as pbar:
                for batch_idx, batch in enumerate(pbar):
                    real_images = batch['image'].cuda()
                    masked_images = batch['masked_image'].cuda()
                    masks = batch['mask'].cuda()
                    
                    # Train Discriminator
                    d_optimizer.zero_grad()
                    
                    # Generate fake images
                    fake_images = generator(masked_images, masks)
                    
                    # Real images
                    real_pred = discriminator(real_images)
                    d_real_loss = loss_functions['adversarial'](real_pred, True)
                    
                    # Fake images
                    fake_pred = discriminator(fake_images.detach())
                    d_fake_loss = loss_functions['adversarial'](fake_pred, False)
                    
                    # Combined discriminator loss
                    d_loss = (d_real_loss + d_fake_loss) * 0.5
                    d_loss.backward()
                    d_optimizer.step()
                    
                    # Train Generator
                    g_optimizer.zero_grad()
                    
                    # Adversarial loss
                    fake_pred = discriminator(fake_images)
                    g_adv_loss = loss_functions['adversarial'](fake_pred, True)
                    
                    # L1 loss
                    g_l1_loss = loss_functions['l1'](fake_images, real_images) * config.lambda_l1
                    
                    # Perceptual loss
                    g_percep_loss = loss_functions['perceptual'](fake_images, real_images) * config.lambda_perceptual
                    
                    # Combined generator loss
                    g_loss = g_adv_loss + g_l1_loss + g_percep_loss
                    g_loss.backward()
                    g_optimizer.step()
                    
                    # Update metrics
                    epoch_g_loss += g_loss.item()
                    epoch_d_loss += d_loss.item()
                    num_batches += 1
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'D_loss': f'{d_loss.item():.4f}',
                        'G_loss': f'{g_loss.item():.4f}'
                    })
                    
                    # Log to MLflow and tensorboard
                    if batch_idx % config.log_interval == 0:
                        step = epoch * len(train_loader) + batch_idx
                        mlflow.log_metrics({
                            'batch_g_loss': g_loss.item(),
                            'batch_d_loss': d_loss.item()
                        }, step=step)
                        writer.add_scalar('Batch/G_Loss', g_loss.item(), step)
                        writer.add_scalar('Batch/D_Loss', d_loss.item(), step)
            
            # Log epoch metrics
            epoch_g_loss /= num_batches
            epoch_d_loss /= num_batches
            mlflow.log_metrics({
                'epoch_g_loss': epoch_g_loss,
                'epoch_d_loss': epoch_d_loss
            }, step=epoch)
            writer.add_scalar('Epoch/G_Loss', epoch_g_loss, epoch)
            writer.add_scalar('Epoch/D_Loss', epoch_d_loss, epoch)
            
            # Validation
            if (epoch + 1) % config.val_freq == 0:
                val_metrics = validate(generator, val_loader, loss_functions, writer, epoch)
                mlflow.log_metrics(val_metrics, step=epoch)
            
            # Save checkpoint
            if (epoch + 1) % config.save_freq == 0:
                checkpoint_path = save_checkpoint({
                    'epoch': epoch + 1,
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'g_optimizer_state_dict': g_optimizer.state_dict(),
                    'd_optimizer_state_dict': d_optimizer.state_dict(),
                }, config.checkpoint_dir, epoch + 1)
                mlflow.log_artifact(checkpoint_path)

def validate(generator, val_loader, loss_functions, writer, epoch):
    generator.eval()
    val_l1_loss = 0
    val_percep_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            real_images = batch['image'].cuda()
            masked_images = batch['masked_image'].cuda()
            masks = batch['mask'].cuda()
            
            fake_images = generator(masked_images, masks)
            
            val_l1_loss += loss_functions['l1'](fake_images, real_images).item()
            val_percep_loss += loss_functions['perceptual'](fake_images, real_images).item()
            num_batches += 1
    
    val_l1_loss /= num_batches
    val_percep_loss /= num_batches
    
    writer.add_scalar('Validation/L1_Loss', val_l1_loss, epoch)
    writer.add_scalar('Validation/Perceptual_Loss', val_percep_loss, epoch)
    
    return {
        'val_l1_loss': val_l1_loss,
        'val_perceptual_loss': val_percep_loss
    }

def save_checkpoint(state, checkpoint_dir, epoch):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(state, checkpoint_path)
    
    # Also save as latest
    latest_path = os.path.join(checkpoint_dir, 'latest.pth')
    torch.save(state, latest_path)
    
    return checkpoint_path

if __name__ == '__main__':
    config = TrainingConfig()
    train(config)
