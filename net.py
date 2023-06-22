import torch
from torch import nn
import torch.nn.functional as F

class DownBlock(nn.Module):
    def __init__(self, cin, cout, activation, **kwargs):
        super().__init__()
        self.activation = activation
        self.down_conv = nn.Conv2d(cin, cout, 3, stride=2, **kwargs)
        self.conv1 = nn.Conv2d(cout, cout, 3, padding='same')
        self.conv2 = nn.Conv2d(cout, cout, 3, padding='same')
        self.batchnorm = nn.BatchNorm2d(cout)


    def forward(self, x):
        x = self.down_conv(x)
        x0 = x
        x = self.activation(x)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = x + x0 
        x = self.activation(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, cin, cout, activation, **kwargs):
        super().__init__()
        self.activation = activation

        if "kernel_size" not in kwargs:
            kwargs["kernel_size"] = 3
        self.up_conv = nn.ConvTranspose2d(cin, cout, stride=2, **kwargs)

        self.conv1 = nn.Conv2d(cout, cout, 3, padding='same')
        self.conv2 = nn.Conv2d(cout, cout, 3, padding='same')
        self.batchnorm = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = self.up_conv(x)
        x0 = x
        x = self.activation(x)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        x = x + x0
        return x

class UNet(nn.Module):
    def __init__(self, activation, channels_start):
        super().__init__()

        self.activation = activation
        cs = [
            channels_start,
            16,
            64,
            256
        ]

        self.down = nn.ModuleList([
            DownBlock(cs[0], cs[1], activation=activation, padding=1),
            DownBlock(cs[1], cs[2], activation=activation, padding=1),
            DownBlock(cs[2], cs[3], activation=activation, padding=1),
        ])

        self.bottom = nn.Conv2d(cs[3], cs[3], 3)

        # wout = (win - 1) * stride - 2 * padding + dilation * (k - 1) + output_padding + 1
        self.up = nn.ModuleList([
            # 3 * 2 - 2 + 2 + 0 + 1 = 6
            UpBlock(cs[3], cs[2], activation=activation, output_padding=1, kernel_size=5),
            UpBlock(cs[2] * 2, cs[1], activation=activation, output_padding=1, padding=1),
            UpBlock(cs[1] * 2, 16, activation=activation, padding = 1),
        ])

        self.last_layer = nn.Sequential(
            nn.Conv2d(16, 1, 1),
            nn.Tanh(),
        )
    
    def forward(self, x):
        intermediates = []

        for layer in self.down:
            intermediates.append(x)
            x = layer(x)
        x = self.bottom(x)

        for i, layer in enumerate(self.up):
            if i != 0:
                x = torch.cat([x, intermediates.pop()], 1)
            x = layer(x)
        
        x = 3 * self.last_layer(x)
        return x[:,:,:28,:28]

class EpsNet(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.unet = UNet(activation, channels_start=4)
        self.activation = activation

        self.fc1 = nn.Linear(1,16)
        self.fc2 = nn.Linear(16, 64 * 3)

    def time_embed(self, im, t):
        t = t.view(-1,1)
        t = self.fc1(t)
        t = self.activation(t)
        t = self.fc2(t)
        t = self.activation(t)
        t_emb = t.view([-1,3, 8, 8])
        t_emb = F.upsample(t_emb, (28, 28))

        return torch.cat([im, t_emb], 1)
    
    def forward(self, im, t):
        x = self.time_embed(im, t)

        x = F.pad(x, (2,2,2,2))

        return self.unet(x)

NUM_CATEGORIES = 11 

# includes context embedding for classifier-free guidance
class EpsNetContext(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.unet = UNet(activation, channels_start=7)
        self.activation = activation

        self.t_fc1 = nn.Linear(1,16)
        self.t_fc2 = nn.Linear(16, 64 * 3)

        self.ctx_fc1 = nn.Linear(NUM_CATEGORIES,32) # one-hot embedding of digits and NULL category
        self.ctx_fc2 = nn.Linear(32, 64 * 3)

    def time_embed(self, im, t):
        t = t.view(-1,1)
        t = self.t_fc1(t)
        t = self.activation(t)
        t = self.t_fc2(t)
        t = self.activation(t)
        t_emb = t.view([-1,3, 8, 8])
        t_emb = F.upsample(t_emb, (28, 28))

        return torch.cat([im, t_emb], 1)
    
    def ctx_embed(self, im, ctx):
        ctx = self.ctx_fc1(ctx)
        ctx = self.activation(ctx)
        ctx = self.ctx_fc2(ctx)
        ctx = self.activation(ctx)
        ctx_emb = ctx.view([-1,3, 8, 8])
        ctx_emb = F.upsample(ctx_emb, (28, 28))

        return torch.cat([im, ctx_emb], 1)
    
    
    def forward(self, im, ctx, t):
        x = self.time_embed(im, t)
        x = self.ctx_embed(x, ctx)

        x = F.pad(x, (2,2,2,2))

        return self.unet(x)

