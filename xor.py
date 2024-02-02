import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch import FloatTensor
from torch import optim
from torch.utils.data import DataLoader, random_split
from backbone.model import simpleMLP
from dataloader.dataloader import XORDataset
import numpy as np
import matplotlib.pyplot as plt

X = torch.Tensor([[0,0],[0,1], [1,0], [1,1]])
Y = torch.Tensor([0,1,1,0]).view(-1,1)

dataset = XORDataset(size=1000)
ratio = 0.8
train_size = int(len(dataset) * 0.8)
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = int(128)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    drop_last=True,
)
val_loader = DataLoader(val_dataset, batch_size=5, shuffle=True)



backbone = simpleMLP(in_channels=2,
                     hidden_channels=[1],
                     # norm_layer=nn.BatchNorm1d,
                     dropout=0, inplace=False, use_sigmoid=True)

criterion = nn.MSELoss()
lr = 1e-2
opt = optim.Adam(
    backbone.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5, amsgrad=False)


steps = X.size(0)

for i in range(20000):
    loss_list = []
    for sample, target in train_loader:
        y_hat = backbone(sample)

        target = target.reshape(y_hat.shape)
        loss = criterion(y_hat, target)
        loss_list.append(loss.item())

        loss.backward()
        opt.step()
        opt.zero_grad()
    print(f'\r Epoch:{i} loss = {np.mean(loss_list)} ,lr = {opt.param_groups[0]["lr"]}     ', end=' ')









