import os
import shutil
import kagglehub

def download_voc(save_dir: str):
    """
    Download Pascal VOC dataset with progress bar support
    url: "https://www.kaggle.com/datasets/huanghanchina/pascal-voc-2012"
    """

    os.makedirs(save_dir, exist_ok=True)
    
    # Using kagglehub to download the dataset
    print("Downloading VOC dataset using kagglehub...")
    downloaded_path = kagglehub.dataset_download("huanghanchina/pascal-voc-2012")
    
    shutil.move(downloaded_path, save_dir)
    print("VOC dataset downloaded successfully!")