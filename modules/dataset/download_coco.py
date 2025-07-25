import os
import sys
import requests
from tqdm import tqdm

def download_coco(save_dir: str):
    # coco urls
    coco_urls = {
        'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
        'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    }
    
    os.makedirs(save_dir, exist_ok=True)
    
    # download
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
    
    print("finished downloading !")
