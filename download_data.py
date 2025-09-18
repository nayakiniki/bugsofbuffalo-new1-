import os
import zipfile
import requests
from tqdm import tqdm

def download_file(url, filename):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

# Create data directory
os.makedirs('ml-model/data', exist_ok=True)

# Download dataset (you'll need to find a direct download URL)
# Note: Kaggle datasets require authentication, so this might not work
# This is just a template - you might need to manually download
dataset_url = "YOUR_DIRECT_DOWNLOAD_LINK_HERE"  # You'll need to get this
zip_path = "ml-model/data/dataset.zip"

if dataset_url != "YOUR_DIRECT_DOWNLOAD_LINK_HERE":
    print("Downloading dataset...")
    download_file(dataset_url, zip_path)
    
    # Extract dataset
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("ml-model/data/")
    
    print("Dataset ready!")
else:
    print("Please manually download the dataset from Kaggle and place it in ml-model/data/")
