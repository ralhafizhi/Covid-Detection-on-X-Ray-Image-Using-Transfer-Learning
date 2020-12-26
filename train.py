import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from jcopdl.callback import Callback, set_config

from src.model import CNN
from src.train_utils import loop_fn

import config as cfg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train():
    # Dataset & Dataloader
    train_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(cfg.crop_size, scale=(0.8, 1.0)),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(230),
    transforms.CenterCrop(cfg.crop_size),
    transforms.ToTensor()
    ])


train_set = datasets.ImageFolder(cfg.TRAIN_DIR, transform=train_transform)
trainloader = DataLoader(
    train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=2)

test_set = datasets.ImageFolder(cfg.TEST_DIR, transform=test_transform)
testloader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=True)

# Config
config = set_config({

        "batch_size": cfg.batch_size,
        "crop_size": cfg.crop_size,
        "in_channel": cfg.in_channel,
        "conv1": cfg.conv1,
        "conv2": cfg.conv2,
        "conv3": cfg.conv3,
        "conv4": cfg.conv4,
        "out_channel": cfg.out_channel,
        "kernel": cfg.kernel,
        "pad": cfg.pad,
        "in_size": cfg.in_size,
        "n1": cfg.n1,
        "n2": cfg.n2,
        "dropout": cfg.dropout,
        "out_size": cfg.out_size,
        "batch_norm": cfg.batch_norm,
        "author": cfg.author

})

# Training Preparation
model = CNN().to(device)
criterion = nn.NLLLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
callback = Callback(model, config, outdir=cfg.OUTDIR)

 # Training
 while True:
      train_cost, train_score = loop_fn(
           "train", train_set, trainloader, model, criterion, optimizer, device)
       with torch.no_grad():
            test_cost, test_score = loop_fn(
                "test", test_set, testloader, model, criterion, optimizer, device)

        # Callbacks
        callback.log(train_cost, test_cost, train_score, test_score)
        callback.save_checkpoint()
        if callback.early_stopping(model, monitor="test_score"):
            break


if __name__ == "__main__":
    train()
