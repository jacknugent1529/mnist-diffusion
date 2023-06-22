from typing import Any, Optional
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from torch import nn
import torch.nn.functional as F
from dataset import get_dataset
from torch.utils.data import DataLoader, random_split
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

def entropy_f(ps):
    return -torch.log(ps).mean()

class MnistClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 128, 3, stride=2),
            nn.Flatten(),
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Linear(128,11),
        )

        m = torch.distributions.Exponential(torch.Tensor([3.]))
        self.sample = lambda n: torch.clamp(1 - m.sample_n(n).flatten(), 0)

    def forward(self, X):
        return self.net(X)
    
    def predict(self, X):
        logits = self.forward(X)
        return F.softmax(logits, dim=1)
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        eps = torch.randn_like(X)
        # add random amount of noise
        sigma = torch.rand(X.shape[0])[:,None,None,None]

        # add noise to data
        X = (1 - sigma).sqrt() * X + sigma.sqrt() * eps
        if sigma[0] > 0.05:
            img = (X[0] + 1) / 2
            self.logger.experiment.add_image("grainy_image", img, batch_idx)


        p = self(X)
        loss = F.cross_entropy(p, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        eps = torch.randn_like(X)
        sigma = torch.rand(X.shape[0])[:,None,None,None]

        # add noise to data
        X = (1 - sigma).sqrt() * X + sigma.sqrt() * eps

        p = self(X)
        loss = F.cross_entropy(p, y)
        pred_y = torch.argmax(p, dim=1)
        acc = (pred_y == y).sum() / len(y)
        
        self.log("val/loss", loss)
        self.log("val/accuracy", acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss
    
    def score_mode(self, on = True):
        requires_grad = not on
        for param in self.net.parameters():
            param.requires_grad = requires_grad
    
    def score(self, X, val):
        X = torch.autograd.Variable(X.detach(), requires_grad=True).to(self.device)
        X.retain_grad()
        self.net.zero_grad()
        logits = self.net(X)

        for i in range(X.shape[0]):
            loss = -logits[i,val]
            loss.backward(retain_graph=True)

        return X.grad
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters())

    def on_train_epoch_end(self):
        # verify that model is uncertain about noisy data
        X = torch.randn(10,1,28,28)
        ps = self.predict(X)
        entropy = entropy_f(ps)
        self.log("entropy", entropy)

def main():
    B = 10
    ds = get_dataset()
    train, val = random_split(ds, [0.8,0.2])
    train_loader = DataLoader(train, B, shuffle=True, num_workers=3)
    val_loader = DataLoader(val, B, num_workers=3)

    model = MnistClassifier()

    trainer = pl.Trainer(max_epochs=10, default_root_dir="lightning_logs_classifier", limit_train_batches=50, limit_val_batches=5)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    main()