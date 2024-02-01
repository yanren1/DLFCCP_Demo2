import torch
import torch.nn as nn
from typing import Callable, List, Optional
import random
import torchvision
from torchvision.models import resnet18, ResNet18_Weights


class simpleMLP(nn.Module):
    def __init__(self,
                in_channels: int,
                hidden_channels: List[int],
                norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
                activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
                inplace: Optional[bool] = None,
                bias: bool = True,
                dropout: float = 0.0,
                ):
        super(simpleMLP, self).__init__()
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout, **params))
        #sigmoid
        # layers.append(torch.nn.Linear(hidden_channels[-1], hidden_channels[-1], bias=bias))
        # layers.append(torch.nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)


    def forward(self, x):
        x = self.mlp(x)
        return x



def MyResnet18(pretrained=True):
    model = resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    return model



if __name__ == '__main__':
    model = MyResnet18()
    sample = torch.randn([512, 1, 28, 28])

    out = model(sample)
    print(out.shape)
