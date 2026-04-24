import torch
import torch.nn as nn
from diffusers import AutoencoderKL

class VAE(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()

        self.vae = AutoencoderKL.from_pretrained("./vae_weights").to(device)

        for para in self.vae.parameters():
            para.requires_grad = False
        
        self.vae.eval()
        self.scaling_factor = 0.18215

    def encode(self, x):
        latent_dist = self.vae.encode(x).latent_dist
        z = latent_dist.sample()
        z = z * self.scaling_factor
        return z
    
    def decode(self, z):
        z = z / self.scaling_factor
        x_recon = self.vae.decode(z).sample
        return x_recon