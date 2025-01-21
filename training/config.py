"""
Training configuration and hyperparameter settings.
"""

class TrainingConfig:
    # Data
    data_dir = 'data/celeba'
    image_size = 256
    batch_size = 8
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
    save_freq = 10
    
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
                if not k.startswith('__') and not callable(v)}
