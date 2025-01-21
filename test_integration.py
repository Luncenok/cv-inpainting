"""
Test script to verify the integration of all components.
"""
import os
import torch
from pathlib import Path
from training.config import TrainingConfig
from models.generator import Generator
from models.discriminator import Discriminator
from utils.data_loader import get_data_loaders
from training.loss_functions import get_loss_functions
import mlflow

def test_data_loading(config):
    """Test data loading functionality."""
    print("\nTesting data loading...")
    try:
        train_loader, val_loader = get_data_loaders(
            config.data_dir,
            batch_size=2,
            num_workers=0
        )
        batch = next(iter(train_loader))
        print("✓ Data loading successful")
        print(f"✓ Batch shapes: {batch['image'].shape}, {batch['masked_image'].shape}, {batch['mask'].shape}")
        return True
    except Exception as e:
        print(f"✗ Data loading failed: {str(e)}")
        return False

def test_models(config):
    """Test model initialization and forward pass."""
    print("\nTesting models...")
    try:
        # Initialize models
        generator = Generator(in_channels=config.in_channels, out_channels=config.out_channels)
        discriminator = Discriminator(in_channels=config.out_channels)
        
        # Test forward pass
        batch_size = 2
        x = torch.randn(batch_size, 3, 256, 256)  # Image
        m = torch.randn(batch_size, 1, 256, 256)  # Mask
        
        with torch.no_grad():
            fake = generator(x, m)
            d_out = discriminator(fake)
        
        print("✓ Models initialized successfully")
        print(f"✓ Generator output shape: {fake.shape}")
        print(f"✓ Discriminator output shape: {d_out.shape}")
        return True
    except Exception as e:
        print(f"✗ Model testing failed: {str(e)}")
        return False

def test_loss_functions():
    """Test loss function initialization."""
    print("\nTesting loss functions...")
    try:
        loss_functions = get_loss_functions()
        print("✓ Loss functions initialized successfully")
        print(f"✓ Available loss functions: {list(loss_functions.keys())}")
        return True
    except Exception as e:
        print(f"✗ Loss function testing failed: {str(e)}")
        return False

def test_mlflow_integration(config):
    """Test MLflow integration."""
    print("\nTesting MLflow integration...")
    try:
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(config.mlflow_experiment_name)
        
        with mlflow.start_run():
            mlflow.log_params(config.to_dict())
            mlflow.log_metrics({'test_metric': 0.5})
        
        print("✓ MLflow integration successful")
        return True
    except Exception as e:
        print(f"✗ MLflow integration failed: {str(e)}")
        return False

def test_directories(config):
    """Test required directories exist."""
    print("\nTesting directory structure...")
    required_dirs = [
        Path(config.data_dir),
        Path(config.checkpoint_dir),
        Path(config.mlflow_tracking_uri)
    ]
    
    all_exist = True
    for d in required_dirs:
        if not d.exists():
            print(f"✗ Directory missing: {d}")
            all_exist = False
            os.makedirs(d, exist_ok=True)
            print(f"  Created directory: {d}")
    
    if all_exist:
        print("✓ All required directories exist")
    return all_exist

def main():
    print("Starting integration tests...")
    config = TrainingConfig()
    
    # Run tests
    tests = [
        ('Directory structure', lambda: test_directories(config)),
        ('Data loading', lambda: test_data_loading(config)),
        ('Models', lambda: test_models(config)),
        ('Loss functions', test_loss_functions),
        ('MLflow integration', lambda: test_mlflow_integration(config))
    ]
    
    results = []
    for test_name, test_fn in tests:
        try:
            success = test_fn()
            results.append((test_name, success))
        except Exception as e:
            print(f"Error in {test_name}: {str(e)}")
            results.append((test_name, False))
    
    # Print summary
    print("\nTest Summary:")
    print("-" * 40)
    all_passed = True
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name:.<30}{status}")
        all_passed = all_passed and success
    
    if all_passed:
        print("\n✓ All integration tests passed!")
    else:
        print("\n✗ Some tests failed. Please check the output above.")

if __name__ == '__main__':
    main()
