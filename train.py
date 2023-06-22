import lightning.pytorch as pl
from diffusion import DDPM, cos_schedule
from classifier_free_diffusion import ClassifierFreeDDPM
from net import EpsNet
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from dataset import get_dataset
import click

def main(activation, schedule, epochs, B, model):
    ddpm = model(activation, schedule, 1000)

    ds = get_dataset()
    train, val = random_split(ds, [0.8,0.2])
    train_loader = DataLoader(train, B, shuffle=True, num_workers=3)
    val_loader = DataLoader(val, B, num_workers=3)

    trainer = pl.Trainer(max_epochs=epochs, precision=16)
    trainer.fit(model = ddpm, train_dataloaders=train_loader, val_dataloaders=val_loader)

@click.command()
@click.option("--activation", type=str, default='silu')
@click.option("--schedule", type=str, default='cosine')
@click.option("--epochs", type=int, required = True)
@click.option("-B", "--batch-size", type=int, required = True)
@click.option("--classifier-free", type=bool, is_flag = True)
def run(activation, schedule, epochs, batch_size, classifier_free):
    if classifier_free:
        print("Training Classifier-Free Model")
        model = ClassifierFreeDDPM
    else:
        print("Training Unconditional Model")
        model = DDPM

    main(activation, schedule, epochs, batch_size, model)

if __name__ == "__main__":
    run()