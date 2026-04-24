from vae import VAE
from unet import UNet
from difu import diffusion
from PIL import Image
from utils import extract_canny
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torch

def generate(real_img_path, unet_path, device="cuda"):
    vae = VAE().to(device)
    unet = UNet().to(device)
    unet.load_state_dict(torch.load(unet_path))
    unet.eval()

    scheduler = diffusion(timesteps=1000, device=device)

    real_img = Image.open(real_img_path).convert('RGB')
    real_img = TF.resize(real_img, 512)
    real_img = TF.center_crop(real_img, 512)

    canny_img = extract_canny(real_img)
    canny_img.save("canny.jpg")
    canny_tensor = TF.to_tensor(canny_img).unsqueeze(0).to(device)
    
    cond_64 = F.interpolate(canny_tensor, size=(64, 64), mode='bilinear', align_corners=False)

    real_tensor = TF.to_tensor(real_img)
    real_tensor = TF.normalize(real_tensor, mean=[0.5]*3, std=[0.5]*3).unsqueeze(0).to(device)
    
    with torch.no_grad():
        base_latent = vae.encode(real_tensor)

    strength = 0.5
    start_t = int(1000 * strength) - 1

    noise = torch.randn_like(base_latent)
    t_tensor = torch.tensor([start_t], device=device).long()
    latent = scheduler.add_noise(base_latent, noise, t_tensor)

    inference_step = 50
    timestep = torch.linspace(start_t, 0, inference_step, dtype=torch.long).to(device)

    with torch.no_grad():
        for i in range(inference_step):
            t1 = timestep[i]
            t0 = timestep[i+1] if i < inference_step - 1 else torch.tensor(-1, device=device)
            
            unet_input = torch.cat([latent, cond_64], dim=1)
            noise_pred = unet(unet_input, t1.unsqueeze(0))

            latent = scheduler.step(noise_pred, t1, t0, latent)

    output_tensor = vae.decode(latent)
    output_tensor = (output_tensor/ 2 + 0.5).clamp(0, 1)
    output_img = TF.to_pil_image(output_tensor.squeeze(0))

    output_img.save("1_output.jpg")
    print("Completed! Saved as 1_output.jpg")

if __name__ == "__main__":
    generate("1.jpg", "unet_epoch_80.pth")