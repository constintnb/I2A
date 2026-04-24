from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as TF
from torch.optim import AdamW
from tqdm import tqdm
import torch

from vae import VAE
from difu import diffusion
from unet import UNet

class training:
    def __init__(self, device="cuda"):
        self.vae_model = VAE().to(device=device)
        self.scheduler = diffusion(timesteps=1000, device=device)
        self.unet_model = UNet(inc=5, outc=4, time_dim=256).to(device)

        self.optimizer = AdamW(self.unet_model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.scaler = GradScaler()
        self.device = device

    def train(self, epoch, dataloader):
        
        optimizer = self.optimizer
        vae_model = self.vae_model
        scheduler = self.scheduler
        unet_model = self.unet_model
        scaler = self.scaler
        device = self.device

        tot_loss = 0.0
        unet_model.train()

        for batch in tqdm(dataloader, total=len(dataloader)):
            loss = 0.0

            img = batch["img"].to(device)
            canny_img = batch["canny_img"].to(device)

            optimizer.zero_grad()

            with autocast():
                latents = vae_model.encode(img)
                cond_64 = TF.interpolate(canny_img, size=(64, 64), mode='bilinear', align_corners=False)

                noise = torch.randn_like(latents)

                bsz = latents.shape[0]
                timesteps = torch.randint(0, scheduler.timesteps, (bsz,), device=device).long()
                
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)

                unet_input = torch.cat([noisy_latents, cond_64], dim=1)
                noise_pred = unet_model(unet_input, timesteps)
                
                loss = TF.mse_loss(noise_pred, noise)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tot_loss += loss.item() 

        if (epoch + 1) % 5 == 0:
            torch.save(unet_model.state_dict(), f"unet_epoch_{epoch+1}.pth")

        return tot_loss / len(dataloader)
