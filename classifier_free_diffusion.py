from diffusion import cos_schedule, get_activation, get_schedule
import torch
from torch import nn
import torch.nn.functional as F
import lightning.pytorch as pl
from net import EpsNetContext

# number of elements in context embedding; 10 for digits and 1 for NULL category
NUM_CATEGORIES = 11 

def digits_dropout(digits, p):
    is_null = torch.rand_like(digits, dtype=torch.float32) < p
    null_digits = torch.ones_like(digits) * NUM_CATEGORIES - 1
    return torch.where(is_null, null_digits, digits)

# similar to DDPM in diffusion.py
class ClassifierFreeDDPM(pl.LightningModule):
    def __init__(self, activation, var_schedule, T):
        super().__init__()
        self.save_hyperparameters(ignore='classifier')

        self.net = EpsNetContext(get_activation(activation))
        self.T = T
        self.var_schedule = get_schedule(var_schedule)(T)
    
    def training_step(self, batch, batch_idx):
        X, digits = batch
        t = torch.randint(1,self.T, X.shape[0:1])
        digits = digits_dropout(digits, 0.1)
        
        ctx = F.one_hot(digits, num_classes=NUM_CATEGORIES).to(torch.float32).to(self.device)

        eps = torch.randn_like(X, device=self.device)
        alpha_bar = self.var_schedule(t, cumulative=True)[:,None,None,None].to(self.device)
        t = t.to(self.device)
        z = alpha_bar.sqrt() * X + (1 - alpha_bar).sqrt() * eps

        pred_eps = self.net(z, ctx, (t / self.T))
        loss = F.mse_loss(pred_eps, eps)

        self.log_dict({"loss": loss})

        return loss
    
    def validation_step(self, batch, batch_idx):
        X, digits = batch

        t = torch.randint(1,self.T, X.shape[0:1])
        digits = digits_dropout(digits, 0.1)
        ctx = F.one_hot(digits,num_classes=NUM_CATEGORIES).to(torch.float32).to(self.device)


        eps = torch.randn_like(X, device=self.device)
        alpha_bar = self.var_schedule(t, cumulative=True)[:,None,None,None].to(self.device)
        t = t.to(self.device)
        z = alpha_bar.sqrt() * X + (1 - alpha_bar).sqrt() * eps

        pred_eps = self.net(z, ctx,  (t / self.T))
        loss = F.mse_loss(pred_eps, eps)

        return loss
    
    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.net.parameters())
        return optim
    
    def sample(self, n, digit, s = 2, debug=False):
        x = torch.randn(n,1,28,28).to(self.device)

        digits = torch.ones(n, dtype=int) * digit
        ctx = F.one_hot(digits, num_classes = NUM_CATEGORIES).to(torch.float32).to(self.device)
        
        null_digits = torch.ones_like(digits) * (NUM_CATEGORIES - 1)
        null_ctx = F.one_hot(null_digits, num_classes = NUM_CATEGORIES).to(torch.float32).to(self.device)


        for t in reversed(range(1,self.T)):
            ts = torch.ones(n) * (t / self.T)
            eps_class = self.net(x, ctx, ts.to(self.device))
            eps_null = self.net(x, null_ctx, ts.to(self.device))
            eps = s * eps_class - (s - 1) * eps_null
            print((eps-eps_null).abs().sum())

            if t > 1:
                z = torch.randn_like(x)
            else:
                z = torch.zeros_like(x)
            
            alpha = self.var_schedule(ts * self.T, cumulative=False)[:,None,None,None]
            alpha_bar = self.var_schedule(ts * self.T, cumulative = True)[:,None,None,None]
            beta = (1 - alpha)

            mu = (1 / alpha.sqrt()) * (x - (1 - alpha) / (1 - alpha_bar).sqrt() * eps)
            x = mu + beta.sqrt() * z

            x = torch.clamp(x, -5, 5)

            if debug and t % 100 == 0:
                print(f"{t:.3f}, {x.min():.3f}, {x.max():.3f}, {x.mean():.3f}")

            assert torch.isnan(x).sum() == 0
        return x