import os
import torch.nn as nn
import torch
from torch.utils.data import ConcatDataset, DataLoader

from datas import parquetdata, imagedata
from train import training

if __name__ == "__main__":

    roots = [
        r"/root/autodl-tmp/data/anime_wallpapers",
        r"/root/autodl-tmp/data/anime-art-image/data1",
        r"/root/autodl-tmp/data/Anime-face-dataset-diffusion_model",
        r"/root/autodl-tmp/data/anime-sceneries/anime scenery_files",
        r"/root/autodl-tmp/data/Anime_GAN_Lite"
    ]
    parquet_root = r"/root/autodl-tmp/data/Genshin_Impact_Scaramouche_Qwen_Image_2509_Anime2Real_Coser/train-00000-of-00001.parquet"

    datasets = []

    for r in roots:
        if os.path.exists(r):
            ds = imagedata(r)
            datasets.append(ds)


    train_dataset = ConcatDataset(datasets)
    test_dataset = parquetdata(parquet_root)

    train_loader = DataLoader(
        train_dataset,
        batch_size = 36,
        shuffle = True,
        num_workers = 8,
        pin_memory = True,
        persistent_workers = False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size = 36,
        shuffle = False,
        num_workers = 8,
        pin_memory = True,
        persistent_workers = False
    )
    
    device = "cuda" 
    epochs = 80

    losses = []
    t = training(device)

    print("Starting training...")
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")

        avg_loss = t.train(epoch, train_loader)

        if (epoch+1)%10 == 0:
            print(f"Step [{epoch+1}/{len(train_loader)}], Loss: {avg_loss:.4f}")
            losses.append(avg_loss)

    print("Training completed.")