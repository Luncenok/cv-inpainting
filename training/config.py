"""
Training configuration and hyperparameter settings.
"""
import torch

class TrainingConfig:
    # Device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    seed = 42
    
    # Debug settings
    debug = False
    debug_images_count = 100  # Number of images to use in debug mode
    debug_epochs = 2  # Number of epochs to run in debug mode
    
    # Data
    data_dir = 'data/celeba'
    image_size = 256
    batch_size = 20
    num_workers = 4
    
    # Model
    in_channels = 4  # RGB + mask
    out_channels = 3  # RGB
    ngf = 64  # Number of generator filters
    ndf = 64  # Number of discriminator filters
    
    # Training
    num_epochs = 100
    g_lr = 0.0002
    d_lr = 0.0002
    beta1 = 0.5
    beta2 = 0.999
    
    # Loss weights
    lambda_l1 = 100
    lambda_perceptual = 10
    
    # Checkpoints
    checkpoint_dir = 'checkpoints'
    save_freq = 1
    
    # Validation
    val_freq = 5
    
    # MLflow
    mlflow_experiment_name = 'image_inpainting'
    mlflow_run_name = None  # Will be set based on timestamp
    mlflow_tracking_uri = 'mlruns'
    
    # Logging
    log_interval = 100  # Log every N batches
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('__') and not callable(getattr(self, k))}
    
    def update_from_args(self, args):
        """Update config from command line arguments."""
        for arg in vars(args):
            if hasattr(self, arg):
                setattr(self, arg, getattr(args, arg))
                
    def get_debug_epochs(self):
        """Get number of epochs based on debug mode."""
        return self.debug_epochs if self.debug else self.num_epochs
