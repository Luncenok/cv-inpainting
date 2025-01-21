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
        
        # Move models to device
        generator = generator.to(config.device)
        discriminator = discriminator.to(config.device)
        
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
        
        # Get loss functions
        loss_functions = get_loss_functions()
        for loss_fn in loss_functions.values():
            if isinstance(loss_fn, torch.nn.Module):
                loss_fn.to(config.device)
        
        # Set up tensorboard
        writer = SummaryWriter(os.path.join(config.mlflow_tracking_uri, 'tensorboard'))
        
        # Training loop
        for epoch in range(config.num_epochs):
            generator.train()
            discriminator.train()
            
            # Training metrics
            train_g_loss = 0
            train_d_loss = 0
            num_batches = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}"):
                real_images = batch['image'].to(config.device)
                masked_images = batch['masked_image'].to(config.device)
                masks = batch['mask'].to(config.device)
                
                # Train discriminator
                d_optimizer.zero_grad()
                
                fake_images = generator(masked_images, masks)
                fake_pred = discriminator(fake_images.detach())
                real_pred = discriminator(real_images)
                
                d_loss = (loss_functions['adversarial'](fake_pred, False) + 
                         loss_functions['adversarial'](real_pred, True)) * 0.5
                
                d_loss.backward()
                d_optimizer.step()
                
                # Train generator
                g_optimizer.zero_grad()
                
                fake_pred = discriminator(fake_images)
                
                g_adv_loss = loss_functions['adversarial'](fake_pred, True)
                g_l1_loss = loss_functions['l1'](fake_images, real_images) * config.lambda_l1
                g_percep_loss = loss_functions['perceptual'](fake_images, real_images) * config.lambda_perceptual
                
                g_loss = g_adv_loss + g_l1_loss + g_percep_loss
                
                g_loss.backward()
                g_optimizer.step()
                
                # Update metrics
                train_g_loss += g_loss.item()
                train_d_loss += d_loss.item()
                num_batches += 1
            
            # Calculate average losses
            train_g_loss /= num_batches
            train_d_loss /= num_batches
            
            # Log metrics
            writer.add_scalar('Training/Generator_Loss', train_g_loss, epoch)
            writer.add_scalar('Training/Discriminator_Loss', train_d_loss, epoch)
            mlflow.log_metrics({
                'train_generator_loss': train_g_loss,
                'train_discriminator_loss': train_d_loss
            }, step=epoch)
            
            # Validation
            if epoch % 5 == 0:
                val_metrics = validate(generator, val_loader, loss_functions, writer, epoch)
                mlflow.log_metrics(val_metrics, step=epoch)
                
                # Save checkpoint
                save_checkpoint({
                    'epoch': epoch,
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'g_optimizer_state_dict': g_optimizer.state_dict(),
                    'd_optimizer_state_dict': d_optimizer.state_dict(),
                }, config.checkpoint_dir, epoch)
        
        writer.close()

def validate(generator, val_loader, loss_functions, writer, epoch):
    generator.eval()
    val_l1_loss = 0
    val_percep_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            real_images = batch['image'].to(generator.device)
            masked_images = batch['masked_image'].to(generator.device)
            masks = batch['mask'].to(generator.device)
            
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
    path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(state, path)
    
    # Also save as latest
    latest_path = os.path.join(checkpoint_dir, 'latest.pt')
    torch.save(state, latest_path)
    
    return path

if __name__ == '__main__':
    config = TrainingConfig()
    config.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    train(config)
