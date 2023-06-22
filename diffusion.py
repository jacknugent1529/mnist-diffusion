import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import lightning.pytorch as pl
import torch.nn.functional as F
from net import EpsNet


def cos_schedule(T):
    # taken from here https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#parameterization-of-beta_t
    def f(t, cumulative=False):
        s = 1e-3
        f = lambda t: np.cos((t / T + s) / (1 + s) * np.pi / 2)**2
        alpha = lambda t: f(t) / f(0.)
        if cumulative:
            return alpha(t)
        else:
            # alpha
            return np.maximum(alpha(t) / alpha(t - 1), 1e-3)
    return f

def get_activation(s):
    match (s):
        case "silu":
            return F.silu
        case "relu":
            return F.relu
        case _:
            raise NotImplementedError()
        
def get_schedule(s):
    match (s):
        case "cosine":
            return cos_schedule
        case _:
            raise NotImplementedError()

class DDPM(pl.LightningModule):
    def __init__(self, activation, var_schedule, T, classifier=None):
        super().__init__()
        self.save_hyperparameters(ignore='classifier')

        self.net = EpsNet(get_activation(activation))
        self.T = T
        self.var_schedule = get_schedule(var_schedule)(T)
        self.classifier = classifier
        
    def training_step(self, batch, batch_idx):
        X, _ = batch
        t = torch.randint(1,self.T, X.shape[0:1])

        eps = torch.randn_like(X, device=self.device)
        alpha_bar = self.var_schedule(t, cumulative=True)[:,None,None,None].to(self.device)
        t = t.to(self.device)
        z = alpha_bar.sqrt() * X + (1 - alpha_bar).sqrt() * eps

        pred_eps = self.net(z, (t / self.T))
        loss = F.mse_loss(pred_eps, eps)

        self.log_dict({"loss": loss})

        return loss
    
    def validation_step(self, batch, batch_idx):
        X, _ = batch

        t = torch.randint(1,self.T, X.shape[0:1])
        eps = torch.randn_like(X, device=self.device)
        alpha_bar = self.var_schedule(t, cumulative=True)[:,None,None,None].to(self.device)
        t = t.to(self.device)
        z = alpha_bar.sqrt() * X + (1 - alpha_bar).sqrt() * eps

        pred_eps = self.net(z, (t / self.T))
        loss = F.mse_loss(pred_eps, eps)

        return loss
    
    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.net.parameters())
        return optim

    def sample(self, n, debug=False):
        x = torch.randn(n,1,28,28).to(self.device)

        for t in reversed(range(1,self.T)):
            ts = torch.ones(n) * (t / self.T)
            eps = self.net(x, ts.to(self.device))

            if t > 1:
                z = torch.randn_like(x)
            else:
                z = torch.zeros_like(x)
            
            alpha = self.var_schedule(ts * self.T, cumulative=False)[:,None,None,None]
            alpha_bar = self.var_schedule(ts * self.T, cumulative = True)[:,None,None,None]
            beta = (1 - alpha)

            mu = (1 / alpha.sqrt()) * (x - (1 - alpha) / (1 - alpha_bar).sqrt() * eps)
            x = mu + beta.sqrt() * z

            if debug and t % 100 == 0:
                print(f"{t:.3f}, {x.min():.3f}, {x.max():.3f}, {x.mean():.3f}")

            assert torch.isnan(x).sum() == 0
        return x

        
    def sample_classifier_guidance(self, n, val, w=2, debug=False):
        x = torch.randn(n,1,28,28).to(self.device)

        for t in reversed(range(1,self.T)):
            ts = torch.ones(n) * (t / self.T)
            eps = self.net(x, ts.to(self.device))

            if t > 1:
                z = torch.randn_like(x)
            else:
                z = torch.zeros_like(x)
            
            alpha = self.var_schedule(ts * self.T, cumulative=False)[:,None,None,None]
            alpha_bar = self.var_schedule(ts * self.T, cumulative = True)[:,None,None,None]
            beta = (1 - alpha)

            # classifier score
            score = self.classifier.score(x, val)
            eps_p = eps + (1 - alpha_bar).sqrt() * w * score

            mu = (1 / alpha.sqrt()) * (x - (1 - alpha) / (1 - alpha_bar).sqrt() * eps_p)
            x = mu + beta.sqrt() * z
            
            x = torch.clamp(x, -5, 5)

            assert torch.isnan(x).sum() == 0
        return x
    
np.set_printoptions(precision=3)
