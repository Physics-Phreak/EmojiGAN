import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import cv2

from imageLoader import ImageLoader
from model import Generator, Discriminator

device = torch.device("cuda")

dataset = ImageLoader()
loader = DataLoader(dataset=dataset,
                    batch_size=8,
                    shuffle=True,
                    num_workers=4)

gen = Generator().to(device)
disc = Discriminator().to(device)

gen_opt = optim.Adam(gen.parameters(), lr=1e-4)
disc_opt = optim.Adam(disc.parameters(), lr=1e-3)

criterion = nn.MSELoss()

for img in loader:

    img.resize_(8, 1, 500, 500)
    img = img.to(device)

    real_label = torch.ones_like(img[0], 1).to(device)
    fake_label = torch.zeros_like(img[0], 1).to(device)
    noise = torch.randn(img.shape).to(device)

    pred = gen(noise)

    #Training Generator
    gen_opt.zero_grad()

    g_loss = criterion(disc(pred), real_label)
    g_loss.backward()
    gen_opt.step()

    #Training Discriminator
    disc_opt.zero_grad()

    d_loss_fake = criterion(disc(pred), fake_label)
    d_loss_real = criterion(disc(img), real_label)

    d_loss = (d_loss_fake + d_loss_real)/2
    d_loss.backward()

    disc_opt.step()

    #printing/logging metrics
    

    break