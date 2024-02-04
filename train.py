import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader.dataloader import SampleDataset
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms, models
from torchvision.transforms import transforms

from tqdm import tqdm
import numpy as np
from backbone.model import MyResnet18,simpleMLP,SimpleCNN
from dataloader.dataloader import XORDataset
from backbone.ghostnetv2_torch import MyGhostnetv2
from PIL import Image, ImageDraw, ImageFont
import time



def save_model(model_save_pth,model, epoch,train_ce,val_ce):
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    filename = 'model_{}_epoch_{}_train_ce_{:0.2e}_val_ce_{:0.2e}.pt'.format(current_time,
                                                                                     epoch,
                                                                                     train_ce,
                                                                                     val_ce,)

    filename = os.path.join(model_save_pth,filename)
    torch.save(model.state_dict(), filename)


def train():

    debug = False
    use_pretrain = False

    # split train and val set
    # transform_train = transforms.Compose([
    #     # transforms.RandomResizedCrop(size = 28),
    #     # transforms.ColorJitter(),
    #     # transforms.RandomHorizontalFlip(),
    #     # transforms.RandomRotation(degrees=30),
    #     transforms.ToTensor(),
    # ])
    # transform_val = transforms.Compose([
    #     transforms.ToTensor(),
    # ])
    # train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
    # val_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_val)

    # train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    # val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)

    train_ratio = 0.8
    dataset = SampleDataset(root_dir='data',file_name='output_file.csv')
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    batch_size = int(60)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True)

    # backbone
    backbone = simpleMLP(in_channels=7,
                        # hidden_channels=[1024,2048,4096,1024,512,18],
                        hidden_channels=[16,32,64,128,18],
                        norm_layer=nn.BatchNorm1d,
                        dropout=0, inplace=False,use_sigmoid=False).cuda()

    # backbone = MyResnet18().cuda()
    # backbone = MyGhostnetv2(num_classes=10, width=0.2, dropout=0.1).cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    # try read pre-train model
    if use_pretrain:
        weights_pth = 'final.pt'
        try:
            backbone.load_state_dict(torch.load(weights_pth))
        except:
            print(f'No {weights_pth}')

    # set lr,#epoch, optimizer and scheduler
    lr = 1e-3
    num_epoch = 10000
    optimizer = optim.Adam(
        backbone.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5, amsgrad=False)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=1e-5)

    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    model_save_pth = os.path.join('model_saved', current_time)
    os.mkdir(model_save_pth)
    writer = SummaryWriter(model_save_pth)

    # start training
    backbone.train()
    for epoch in range(num_epoch):
        loss_list = []
        for sample, target in train_loader:
            backbone.zero_grad()
            # print(sample.shape, target.shape)
            sample, target = sample.cuda(), target.cuda()
            # print(sample, target)
            output = backbone(sample)
            loss = criterion(output, target)

            loss.backward()

            optimizer.step()
            loss_list.append(loss.item())

        scheduler.step()

        if epoch % 1 == 0:
            print(f'\r Epoch:{epoch} ce loss = {np.mean(loss_list)} ,lr = {optimizer.param_groups[0]["lr"]}     ', end = ' ')
            writer.add_scalar('Training CE Loss', np.mean(loss_list), epoch)
            writer.add_scalar('Learning rate', optimizer.param_groups[0]["lr"], epoch)

        # valing and save
        if epoch % 10 == 0:
            print('Valing.....')
            val_loss_list = []
            backbone.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_sample, val_target = val_batch
                    val_sample, val_target = val_sample.cuda(), val_target.cuda()

                    output = backbone(val_sample)
                    val_ce = criterion(output, val_target)
                    val_loss_list.append(val_ce.item())

                    _, predicted = torch.max(output.data, 1)
                    total += val_target.size(0)
                    correct += (predicted == val_target).sum().item()

            Train_ce = np.mean(loss_list)
            val_ce = np.mean(val_loss_list)
            accuracy = correct / total

            # img_grid = torchvision.utils.make_grid(val_sample)
            # writer.add_image('Predicted Images', img_grid, epoch + 1)
            # writer.add_text('Predicted Labels', str(predicted.tolist()), epoch + 1)

            # resized_images = torch.nn.functional.interpolate(val_sample, size=(512, 512), mode='bilinear',
            #                                               align_corners=False)

            # class_labels = {
            #     0: 'T-shirt/top',
            #     1: 'Trouser',
            #     2: 'Pullover',
            #     3: 'Dress',
            #     4: 'Coat',
            #     5: 'Sandal',
            #     6: 'Shirt',
            #     7: 'Sneaker',
            #     8: 'Bag',
            #     9: 'Ankle boot'
            # }
            #
            # predicted_labels = [class_labels[label.item()] for label in predicted]
            # actual_labels = [class_labels[label.item()] for label in val_target]
            # # Create image grid
            # img_grid = torchvision.utils.make_grid(resized_images,)
            # import matplotlib.pyplot as plt
            # # Create a figure and add labels
            # fig, ax = plt.subplots(figsize=(10, 10))
            # ax.imshow(img_grid.permute(1, 2, 0).cpu().numpy())
            # ax.set_title(f'Predicted: {predicted_labels}\nActual: {actual_labels}')
            # writer.add_figure('Predicted Images with Labels', fig, global_step=epoch  + 1)
            #
            #
            writer.add_scalar('Validation ce', val_ce, epoch + 1)
            writer.add_scalar('Validation accuracy', accuracy, epoch + 1)

            print(f'VAL Epoch:{epoch} Train ce = {Train_ce}, '
                  f'val ce = {val_ce} , val accuracy = {accuracy}')
            print()
            save_model(model_save_pth,backbone, epoch, Train_ce, val_ce)
            backbone.train()

    torch.save(backbone.state_dict(), os.path.join(model_save_pth,'final.pt'))
    # dummy_input = torch.randn([1, 1, 28, 28], requires_grad=True).cuda()
    # torch.onnx.export(backbone,  # model being run
    #                   dummy_input,  # model input (or a tuple for multiple inputs)
    #                   os.path.join(model_save_pth,'final.onnx'),  # where to save the model
    #                   export_params=True,  # store the trained parameter weights inside the model file
    #                   opset_version=10,  # the ONNX version to export the model to
    #                   do_constant_folding=True,  # whether to execute constant folding for optimization
    #                   input_names=['modelInput'],  # the model's input names
    #                   output_names=['modelOutput'],  # the model's output names
    #                   dynamic_axes={'modelInput': {0: 'batch_size'},  # variable length axes
    #                                 'modelOutput': {0: 'batch_size'}})
    writer.flush()
    writer.close()

if __name__ == '__main__':
    train()