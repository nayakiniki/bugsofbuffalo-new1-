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

def main():
    # Create data directory
    os.makedirs('ml-model/data', exist_ok=True)
    
    # Direct download link for the dataset (you might need to update this)
    # This is a placeholder - you'll need to get the actual direct download URL
    dataset_url = "https://www.kaggle.com/datasets/lukex9442/indian-bovine-breeds"
    
    zip_path = "ml-model/data/dataset.zip"
    extract_path = "ml-model/data/"

    try:
        print("Downloading dataset...")
        download_file(dataset_url, zip_path)
        
        # Extract dataset
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        # Clean up zip file
        os.remove(zip_path)
        
        print("Dataset downloaded and extracted successfully!")
        print(f"Data located at: {os.path.abspath(extract_path)}")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nPlease manually download the dataset:")
        print("1. Go to: https://www.kaggle.com/datasets/lukex9442/indian-bovine-breeds")
        print("2. Click 'Download' (requires Kaggle login)")
        print("3. Extract the zip file to 'ml-model/data/indian-bovine-breeds/'")
        print("4. The structure should be: ml-model/data/indian-bovine-breeds/train/...")

if __name__ == "__main__":
    main()
