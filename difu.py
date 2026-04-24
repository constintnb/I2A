import torch
import math 

class diffusion:
    def __init__(self, timesteps, device="cuda"):
        self.timesteps = timesteps
        self.device = device

        self.betas = self.cos_schedule(timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def cos_schedule(self, timesteps, s=0.008):
        '''
        Improve DDPM 余弦调度
        '''
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alpha_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alpha_cumprod = alpha_cumprod / alpha_cumprod[0]

        betas = 1 - (alpha_cumprod[1:] / alpha_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def extract(self, arr, timesteps, broadcast_shape):
        '''
        辅助函数：从预计算的一维数组中，把当前 batch 中各个 t 对应的数值取出来，
        并 Reshape 成 [N, 1, 1, 1] 以便和形如 [N, C, H, W] 的图像张量进行广播计算
        '''
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res.unsqueeze(-1)
        return res
    
    def add_noise(self, origin, noise, timesteps):
        '''
        前向过程
        '''
        alpha1 = torch.sqrt(self.alphas_cumprod)
        alpha1 = self.extract(alpha1, timesteps, origin.shape)

        alpha2 = torch.sqrt(1.0 - self.alphas_cumprod)
        alpha2 = self.extract(alpha2, timesteps, origin.shape)

        nosiy_samples = alpha1 * origin + alpha2 * noise
        return nosiy_samples
    
    def step(self, model_output, timestep1, timestep0, sample):
        '''
        反向过程 DDIM
        '''
        alpha1 = self.alphas_cumprod[timestep1]
        alpha0 = self.alphas_cumprod[timestep0] if timestep0>=0 else torch.tensor(1.0, device=self.device)

        pred_sample = (sample - torch.sqrt(1 - alpha1) * model_output) / torch.sqrt(alpha1)
        pred_sample = torch.clamp(pred_sample, -5.0, 5.0)

        direction = torch.sqrt(1 - alpha0) * model_output
        prev_sample = torch.sqrt(alpha0) * pred_sample + direction
        return prev_sample

