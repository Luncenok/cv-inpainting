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

def train(config: TrainingConfig):
    # Set up MLflow and tensorboard directories
    os.makedirs('mlruns', exist_ok=True)
    os.makedirs('runs', exist_ok=True)
    
    # Set up MLflow
    mlflow.set_tracking_uri('file:mlruns')
    mlflow.set_experiment('image_inpainting')
    
    # Set up tensorboard
    writer = SummaryWriter(os.path.join('runs', datetime.now().strftime('%Y%m%d-%H%M%S')))
    
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
        
        # Training loop
        for epoch in range(config.num_epochs):
            generator.train()
            discriminator.train()
            
            # Training metrics
            train_g_loss = 0
            train_d_loss = 0
            num_batches = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}"):
                # Load and validate input tensors
                real_images = validate_tensor_shapes("Real images", batch['image'].to(config.device))
                masked_images = validate_tensor_shapes("Masked images", batch['masked_image'].to(config.device))
                masks = validate_tensor_shapes("Masks", batch['mask'].to(config.device))
                
                batch_size = real_images.size(0)
                expected_shape = (batch_size, 3, 256, 256)
                mask_shape = (batch_size, 1, 256, 256)
                
                if real_images.shape != expected_shape:
                    raise ValueError(f"Invalid real images shape. Expected {expected_shape}, got {real_images.shape}")
                if masked_images.shape != expected_shape:
                    raise ValueError(f"Invalid masked images shape. Expected {expected_shape}, got {masked_images.shape}")
                if masks.shape != mask_shape:
                    raise ValueError(f"Invalid masks shape. Expected {mask_shape}, got {masks.shape}")
                
                # Train discriminator
                d_optimizer.zero_grad()
                
                # Generate and validate fake images
                fake_images = generator(masked_images, masks)
                fake_images = validate_tensor_shapes("Generated images", fake_images, expected_shape)
                
                # Get discriminator predictions
                fake_pred = discriminator(fake_images.detach())
                real_pred = discriminator(real_images)
                
                # Validate discriminator outputs
                disc_output_shape = (batch_size, 1)
                fake_pred = validate_tensor_shapes("Fake predictions", fake_pred, disc_output_shape)
                real_pred = validate_tensor_shapes("Real predictions", real_pred, disc_output_shape)
                
                # Calculate discriminator loss
                d_loss = (loss_functions['adversarial'](fake_pred, False) + 
                            loss_functions['adversarial'](real_pred, True)) * 0.5
                
                d_loss.backward()
                d_optimizer.step()
                
                # Train generator
                g_optimizer.zero_grad()
                
                # Get new discriminator predictions for generator
                fake_pred = discriminator(fake_images)
                fake_pred = validate_tensor_shapes("Generator fake predictions", fake_pred, disc_output_shape)
                
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
