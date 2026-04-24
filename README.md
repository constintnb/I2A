---

# I2A-80: 真实图像到动漫风格生成器 (Image-to-Anime)

I2A-80 是一个基于潜在扩散模型（Latent Diffusion Model）的图像风格迁移项目。它通过 VAE 压缩图像特征，并结合 Canny 边缘引导，在潜在空间内生成动漫风格的图像。

## 🚀 项目功能
- **潜在空间处理**：利用预训练的 VAE 模型进行图像的编码与解码，降低计算成本。
- **Canny 边缘引导**：提取原图的 Canny 算子边缘作为辅助条件，保持生成图像的结构一致性。
- **DDIM 采样**：实现了基于余弦调度（Cosine Schedule）的前向加噪和 DDIM 反向采样过程。
- **自动混合精度训练**：支持 `autocast` 和 `GradScaler` 提高训练效率。

## 📂 目录结构
- `main.py`: 训练启动入口，配置数据集路径和数据加载器。
- `train.py`: 核心训练循环，包含损失计算与模型权重保存。
- `inference.py`: 推理脚本，加载权重并生成动漫效果图。
- `unet.py`: 自定义 UNet 结构，支持时间步嵌入和多尺度卷积。
- `vae.py`: 封装了 `diffusers` 的 `AutoencoderKL`，用于潜在空间转换。
- `difu.py`: 扩散调度器，包含余弦加噪策略和 DDIM 步进逻辑。
- `datas.py`: 数据预处理逻辑，支持 Parquet 和本地文件夹格式。
- `utils.py`: 工具函数，如 Canny 边缘提取。
- `vae_weights/`: 存放预训练 VAE 模型的配置文件和权重。

## 🛠️ 模型架构
1. **VAE**: 使用 `stabilityai/sd-vae-ft-mse` 风格的权重，将 512x512 的图像压缩为 64x64 的潜在张量。
2. **UNet**: 接收 `5` 通道输入（4 通道潜在噪声 + 1 通道 Canny 边缘条件），通过 4 层下采样和上采样提取特征。
3. **Condition**: Canny 边缘图被缩放至 64x64，并在通道维度与潜在张量拼接。

## 📈 训练说明
你可以通过运行 `main.py` 开始训练。模型默认每 5 个 epoch 会自动保存一次权重（如 `unet_epoch_80.pth`）。
根据 `loss_curve.png` 显示，模型在 80 个 epoch 内表现出良好的收敛趋势。

```bash
python main.py
```

## 🎨 推理生成
使用 `inference.py` 进行图像转换。你可以通过调整 `strength` 参数（默认 0.5）来控制原图的保留程度。

```python
# 示例用法
from inference import generate
generate(real_img_path="input.jpg", unet_path="unet_epoch_80.pth")
```

生成的图像将保存为 `1_output.jpg`，同时会输出对应的 `canny.jpg` 边缘参考图。

## 📦 依赖要求
- PyTorch
- Diffusers
- OpenCV-Python
- PIL (Pillow)
- Pandas (用于 Parquet 数据读取)
- Tqdm
