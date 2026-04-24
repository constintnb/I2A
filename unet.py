import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class time_embedding(nn.Module):
    def __init__(self, dim):
        super(time_embedding, self).__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Conv(nn.Module):
    def __init__(self, inc, outc, time_dim):
        super(Conv, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=3, padding=1),
            nn.GroupNorm(8, outc),
            nn.SiLU(inplace=True)
        )

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, outc)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(outc, outc, kernel_size=3, padding=1),
            nn.GroupNorm(8, outc),
            nn.SiLU(inplace=True)
        )
        

    def forward(self, x, t_emb):

        h = self.conv1(x)
        t_emb = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + t_emb
        h = self.conv2(h)

        return h

class downs(nn.Module):
    def __init__(self, inc, outc, time_dim):
        super(downs, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = Conv(inc, outc, time_dim)
    
    def forward(self, x, t_emb):
        x = self.pool(x)
        return self.conv(x, t_emb)

class ups(nn.Module):
    def __init__(self, inc, outc, time_dim):
        super(ups, self).__init__()
        self.up = nn.ConvTranspose2d(inc, inc//2, kernel_size=2, stride=2)
        self.conv = Conv(inc, outc, time_dim)

    def forward(self, x1, x2, t_emb):
        x1 = self.up(x1)

        dh = x2.size()[2] - x1.size()[2] # H
        dw = x2.size()[3] - x1.size()[3] # W

        if dw > 0 or dh > 0:
            x1 = F.pad(x1, [dw // 2, dw - dw // 2,
                            dh // 2, dh - dh // 2])

        x =  torch.cat([x2, x1], dim=1) #拼接
        return self.conv(x, t_emb)

class outconv(nn.Module):
    def __init__(self, inc, outc):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(inc, outc, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, inc=5, outc=4, time_dim=256):
        super(UNet, self).__init__()

        self.time_dim = time_dim

        self.time_mlp = nn.Sequential(
            time_embedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.inp = Conv(inc, 64, time_dim)
        self.down1 = downs(64, 128, time_dim)
        self.down2 = downs(128, 256, time_dim)
        self.down3 = downs(256, 512, time_dim)
        self.down4 = downs(512, 1024, time_dim)

        self.up1 = ups(1024, 512, time_dim)
        self.up2 = ups(512, 256, time_dim)
        self.up3 = ups(256, 128, time_dim)
        self.up4 = ups(128, 64, time_dim)

        self.out = outconv(64, outc)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x1 = self.inp(x, t_emb)

        x2 = self.down1(x1, t_emb)
        x3 = self.down2(x2, t_emb)
        x4 = self.down3(x3, t_emb)
        x5 = self.down4(x4, t_emb)

        x = self.up1(x5, x4, t_emb)
        x = self.up2(x, x3, t_emb)
        x = self.up3(x, x2, t_emb)
        x = self.up4(x, x1, t_emb)
        logits = self.out(x)

        return logits
        
