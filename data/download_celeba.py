"""
Script to download and organize the CelebA dataset.
"""
import os
import zipfile
import requests
from tqdm import tqdm

def download_celeba():
    """
    Instructions for downloading CelebA dataset.
    """
    # Create directories
    os.makedirs('celeba', exist_ok=True)
    
    print("""
Please download the CelebA dataset manually:

1. Go to Kaggle: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
2. Download the "img_align_celeba.zip" file
3. Place it in the 'celeba' directory
4. Run this script again to extract and organize the files

Alternative sources:
- Google Drive: https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view
- Academic Torrents: https://academictorrents.com/details/f1f6111d0963c12f34e2e7d6a7b0b7d16b3c2ab1
""")
    
    output = 'celeba/img_align_celeba.zip'
    if not os.path.exists(output):
        print("\nZip file not found. Please download it first.")
        return
        
    # Extract files
    print("Extracting files...")
    with zipfile.ZipFile(output, 'r') as zip_ref:
        for file in tqdm(zip_ref.namelist()):
            zip_ref.extract(file, 'celeba')
    
    # Remove zip file after successful extraction
    os.remove(output)
    print("Extraction complete!")

def organize_dataset():
    """
    Organize dataset into train, validation, and test sets.
    Following the official split from the CelebA paper.
    """
    # Create directories
    for split in ['train', 'val', 'test']:
        os.makedirs(f'celeba/{split}', exist_ok=True)
    
    # Create partition mapping (first 162770 images are training, next 19867 validation, rest test)
    partition_mapping = {}
    for i in range(1, 202599 + 1):  # Total number of images
        img_name = f"{i:06d}.jpg"
        if i <= 162770:
            partition_mapping[img_name] = 0  # train
        elif i <= 182637:  # 162770 + 19867
            partition_mapping[img_name] = 1  # validation
        else:
            partition_mapping[img_name] = 2  # test
    
    # Move images to appropriate directories
    print("Organizing dataset...")
    img_dir = 'celeba/img_align_celeba'
    if not os.path.exists(img_dir):
        print(f"Error: {img_dir} directory not found. Please make sure you have extracted the dataset.")
        return
        
    for img_name in tqdm(os.listdir(img_dir)):
        if img_name.endswith('.jpg'):
            partition = partition_mapping.get(img_name, 0)  # Default to train if unknown
            
            # Map partition number to split name
            split = {0: 'train', 1: 'val', 2: 'test'}[partition]
            
            # Move file to appropriate directory
            src = f'{img_dir}/{img_name}'
            dst = f'celeba/{split}/{img_name}'
            
            if os.path.exists(src):
                os.rename(src, dst)
    
    # Try to remove the empty directory
    try:
        os.rmdir(img_dir)
    except OSError:
        pass  # Directory might not be empty or might not exist
        
    print("Dataset organization complete!")

if __name__ == '__main__':
    download_celeba()
    organize_dataset()
