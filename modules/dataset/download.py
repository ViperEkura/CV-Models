import os
import zipfile
import requests
import shutil
import kagglehub
from tqdm import tqdm


def download_coco(save_dir: str):
    # coco urls
    coco_urls = {
        'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
        'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    }
    dir_names = {
        "train_images": "train2017",
        "val_images": "val2017",
        "annotations": "annotations"
    }
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Download ZIP files
    for name, url in coco_urls.items():
        save_path = os.path.join(save_dir, os.path.basename(url))
        if os.path.exists(save_path):
            continue
        
        print(f"downloading {name} to {save_path}")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(save_path, 'wb') as f, tqdm(
            desc=name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                progress_bar.update(size)
    
    # Extract ZIP files
    for name, url in coco_urls.items():
        filename = dir_names[name]
        save_path = os.path.join(save_dir, filename)
        
        if os.path.exists(save_path):
            print(f"Zip file {save_path} does aready exist. Skipping extraction.")
            continue
        
        print(f"Extracting {name} to {save_path}")
        with zipfile.ZipFile(save_dir, 'r') as zip_ref:
            zip_ref.extractall(save_dir)
    
    print("finished downloading and extracting !")
    
    train_images_path = os.path.join(save_dir, 'train2017')
    val_images_path = os.path.join(save_dir, 'val2017')
    annotations_path = os.path.join(save_dir, 'annotations')
    
    return train_images_path, val_images_path, annotations_path



def download_voc(save_dir: str):
    """
    Download Pascal VOC dataset with progress bar support
    url: "https://www.kaggle.com/datasets/huanghanchina/pascal-voc-2012"
    """

    os.makedirs(save_dir, exist_ok=True)
    if os.path.exists(os.path.join(save_dir, "VOCdevkit")):
        print("VOC dataset already exists. Skipping download.")
        return
    # Using kagglehub to download the dataset
    print("Downloading VOC dataset using kagglehub...")
    downloaded_path = kagglehub.dataset_download("huanghanchina/pascal-voc-2012")
    
    shutil.move(downloaded_path, save_dir)
    print("VOC dataset downloaded successfully!")