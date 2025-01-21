"""
Script to download and organize the CelebA dataset.
"""
import os
import gdown
import zipfile
from tqdm import tqdm

def download_celeba():
    """
    Download CelebA dataset from Google Drive.
    Note: Using Google Drive mirror due to slow download speeds from official source.
    """
    # Create directories
    os.makedirs('celeba', exist_ok=True)
    
    # Google Drive file ID for aligned images
    file_id = '0B7EVK8r0v71pZjFTYXZWM3FlRnM'
    
    # Download zip file
    print("Downloading CelebA dataset...")
    output = 'celeba/img_align_celeba.zip'
    gdown.download(f'https://drive.google.com/uc?id={file_id}', output, quiet=False)
    
    # Extract files
    print("Extracting files...")
    with zipfile.ZipFile(output, 'r') as zip_ref:
        for file in tqdm(zip_ref.namelist()):
            zip_ref.extract(file, 'celeba')
    
    # Remove zip file
    os.remove(output)
    print("Download and extraction complete!")

def organize_dataset():
    """
    Organize dataset into train, validation, and test sets.
    Following the official split from the CelebA paper.
    """
    # Create directories
    for split in ['train', 'val', 'test']:
        os.makedirs(f'celeba/{split}', exist_ok=True)
    
    # Download partition file if not exists
    if not os.path.exists('celeba/list_eval_partition.txt'):
        partition_id = '0B7EVK8r0v71pY0NSMzRuSXJEVkk'
        gdown.download(f'https://drive.google.com/uc?id={partition_id}',
                      'celeba/list_eval_partition.txt',
                      quiet=False)
    
    # Read partition file and move images
    print("Organizing dataset...")
    with open('celeba/list_eval_partition.txt', 'r') as f:
        for line in tqdm(f):
            img_name, partition = line.strip().split()
            partition = int(partition)
            
            # Map partition number to split name
            split = {0: 'train', 1: 'val', 2: 'test'}[partition]
            
            # Move file to appropriate directory
            src = f'celeba/img_align_celeba/{img_name}'
            dst = f'celeba/{split}/{img_name}'
            
            if os.path.exists(src):
                os.rename(src, dst)
    
    # Remove empty directory
    os.rmdir('celeba/img_align_celeba')
    print("Dataset organization complete!")

if __name__ == '__main__':
    download_celeba()
    organize_dataset()
