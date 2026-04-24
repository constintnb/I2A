import io
import random
import os
import torch
import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils import extract_canny
import tqdm

def augment(tensor):

    if random.random() > 0.3:
        mask = torch.rand_like(tensor[0:1, :, :]) > 0.15
        tensor = tensor * mask
    
    if random.random() > 0.3:
        noise = torch.randn_like(tensor) * 0.05
        tensor = tensor + noise
        tensor = torch.clamp(tensor, -1.0, 1.0)

    return tensor

class parquetdata(Dataset):
    def __init__(self, parquet_path, ispair=False):
        super().__init__()
        self.data = pd.read_parquet(parquet_path)
        self.ispair = ispair

        self.transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        real_byte = row['real_image']['bytes']
        anime_byte = row['image']['bytes']

        real_img = Image.open(io.BytesIO(real_byte)).convert('RGB')
        anime_img = Image.open(io.BytesIO(anime_byte)).convert('RGB')
        canny_img = extract_canny(real_img)

        real_tensor = self.transform(real_img)
        anime_tensor = self.transform(anime_img)
        canny_tensor = self.transform(canny_img)

        return {
            "real_img": real_tensor,
            "anime_img": anime_tensor,
            "canny_img": canny_tensor
        }
    
class imagedata(Dataset):
    def __init__(self, root_dir, cache_mode=False):
        super().__init__()
        self.root_dir = root_dir
        
        self.filenames = sorted(os.listdir(root_dir))
        self.cache_mode = cache_mode
        self.cache = {}

        if self.cache_mode:
            print(f"Loading {len(self.filenames)} images into memory...")
            for i,name in tqdm(enumerate(self.filenames), total=len(self.filenames)):
                img = Image.open(os.path.join(root_dir, name)).convert('RGB')
                canny_img = extract_canny(img)
                self.cache[i] = (img, canny_img)


    def __len__(self):
        return len(self.filenames)
    
    
    def __getitem__(self, idx):
        if self.cache_mode:
            img, canny_img = self.cache[idx]
        else:
            name = self.filenames[idx]
            img = Image.open(os.path.join(self.root_dir, name)).convert('RGB')
            canny_img = extract_canny(img)

        img = TF.resize(img, 512)
        canny_img = TF.resize(canny_img, 512)
        img = TF.center_crop(img, 512)
        canny_img = TF.center_crop(canny_img, 512)

        if random.random() > 0.5:
            img = TF.hflip(img)
            canny_img = TF.hflip(canny_img)
        
        if random.random() > 0.5:
            img = TF.resize(img, 576)
            canny_img = TF.resize(canny_img, 576)

            i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(512, 512))
            img = TF.crop(img, i, j, h, w)
            canny_img = TF.crop(canny_img, i, j, h, w)

        img_tensor = TF.to_tensor(img)
        canny_tensor = TF.to_tensor(canny_img)
        canny_tensor = augment(canny_tensor)
        
        return {
            "img": img_tensor,
            "canny_img": canny_tensor
        }
    