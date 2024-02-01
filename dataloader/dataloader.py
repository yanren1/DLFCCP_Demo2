import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

class SampleDataset(Dataset):
    def __init__(self, root_dir, file_name):
        super(SampleDataset, self).__init__()

        self.root_dir = root_dir
        self.file_name = file_name
        self.samples = self.__read_xlsx()

    def __getitem__(self, index):
        samples = self.samples[index]
        # sample, target = samples[:-3],samples[-3:]

        return samples[:-1], samples[-1]

    def __len__(self):
        return len(self.samples)

    def __read_xlsx(self):
        f_pth = os.path.join(self.root_dir, self.file_name)
        # f_pth = os.path.join(root_dir, 'data.xlsx')
        df = pd.read_csv(f_pth,usecols=['make', 'body-style', 'wheel-base', 'engine-size', 'horsepower', 'peak-rpm', 'highway-mpg','price'])

        samples = torch.from_numpy(df.to_numpy()).float()
        # [make, body - style, wheel - base, engine - size, horsepower, peak - rpm, highway - mpg]
        return samples

# if __name__ == '__main__':
#     from torch.utils.data import DataLoader, random_split
#     train_ratio = 0.85
#     dataset = SampleDataset(root_dir='../data', file_name='output_file.csv')
#     train_size = int(train_ratio * len(dataset))
#     val_size = len(dataset) - train_size
#     train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=int(len(train_dataset)),
#         shuffle=True,
#         num_workers=0,
#         pin_memory=True,
#         drop_last=True,
#     )
#     val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True)
