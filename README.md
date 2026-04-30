---

# I2A: Real-Image-to-Anime Generator

I2A is an image style transfer project based on a latent diffusion model. It compresses image features using a VAE and combines this with Canny edge guidance to generate anime-style images in latent space.

## 🚀 Project Features
- **Latent Space Processing**: Utilizes a pre-trained VAE model for image encoding and decoding to reduce computational costs.
- **Canny Edge Guidance**: Extracts Canny operator edges from the original image as auxiliary conditions to maintain structural consistency in the generated images.
- **DDIM Sampling**: Implements forward noise addition and DDIM backward sampling processes based on the Cosine Schedule.
- **Automatic Mixed-Precision Training**: Supports `autocast` and `GradScaler` to improve training efficiency.

## 📂 Directory Structure
- `main.py`: Training entry point; configures dataset paths and data loaders.
- `train.py`: Core training loop; includes loss calculation and model weight saving.
- `inference.py`: Inference script that loads weights and generates anime-style images.
- `unet.py`: Custom UNet architecture supporting temporal embedding and multi-scale convolutions.
- `vae.py`: Wrapper for `diffusers`’s `AutoencoderKL`, used for latent space transformation.
- `difu.py`: Diffusion scheduler, including the cosine-added-noise strategy and DDIM iteration logic.
- `datas.py`: Data preprocessing logic, supporting Parquet and local folder formats.
- `utils.py`: Utility functions, such as Canny edge detection.
- `vae_weights/`: Contains configuration files and weights for pre-trained VAE models.

## 🛠️ Model Architecture
1. **VAE**: Uses weights in the style of `stabilityai/sd-vae-ft-mse` to compress 512x512 images into 64x64 latent tensors.
2. **UNet**: Accepts a `5`-channel input (4 channels of latent noise + 1 channel of Canny edge conditions) and extracts features through 4 layers of downsampling and upsampling.
3. **Condition**: The Canny edge map is resized to 64x64 and concatenated with the latent tensor along the channel dimension.

## 📈 Training Instructions
You can start training by running `main.py`. By default, the model automatically saves weights every 5 epochs (e.g., `unet_epoch_80.pth`).
As shown in `loss_curve.png`, the model exhibits a good convergence trend within 80 epochs.

```bash
python main.py
```

## 🎨 Inference and Generation
Use `inference.py` to perform image transformation. You can control the degree to which the original image is preserved by adjusting the `strength` parameter (default 0.5).

```python
# Example usage
from inference import generate
generate(real_img_path=“input.jpg”, unet_path="unet_epoch_80.pth")
```

The generated image will be saved as `1_output.jpg`, and the corresponding `canny.jpg` edge reference image will also be output.

## 📦 Dependencies
- PyTorch
- Diffusers
- OpenCV-Python
- PIL (Pillow)
- Pandas (for reading Parquet data)
- Tqdm


Translated with DeepL.com (free version)
