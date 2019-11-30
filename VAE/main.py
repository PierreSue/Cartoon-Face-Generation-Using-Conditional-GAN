from __future__ import print_function
import argparse
import os
import sys
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from folder import ImageFolder
from network import VAE
from utils import CustomLoss

##### Command example
# python3 main.py --dataroot ../selected_cartoonset100k --outf ./result/VAE20190529/ --cuda

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', default=128, type=int,
                    help='batch size of the model (default: 128)')
parser.add_argument('--epochs', default=1000, type=int,
                    help='training epochs (default: 1000)')
parser.add_argument('--log-step', default=1, type=int,
                    help='printing step size (default: 10')
parser.add_argument('--save-freq', default=100, type=int,
                    help='save checkpoints frequency (default: 100)')
parser.add_argument('--manualSeed', type=int, default=1126, help='manual seed')
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')

# GPU setting
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--gpu_id', type=int, default=0, help='The ID of the specified GPU')

opt = parser.parse_args()
print(opt)

# random seed
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# specify the gpu id if using only 1 gpu
if opt.ngpu == 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

# dataset
dataset = ImageFolder(
    root=opt.dataroot,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

# model
netVAE = VAE()
criterion = CustomLoss(3e-6)
optimizer = optim.Adam(netVAE.parameters(), lr=0.0001, betas=(0.5, 0.999))

if opt.cuda:
    netVAE.cuda()

# train
min_loss = float('inf')
kld_loss_list, mse_loss_list = [], []
for epoch in range(1, opt.epochs + 1):
    mse_loss, kld_loss, total_loss = 0, 0, 0
    for batch_idx, in_fig in enumerate(dataloader):
        x = Variable(in_fig) 
        if opt.cuda:
            x = x.cuda()
        optimizer.zero_grad()
        output, mu, logvar = netVAE(x)
        loss = criterion(output, x, mu, logvar)
        loss.backward()
        optimizer.step()

        mse_loss += criterion.latestloss()['MSE'].item()
        kld_loss += criterion.latestloss()['KLD'].item()
        total_loss += loss.item()
        if batch_idx % opt.log_step == 0:
            print('Epoch: {}/{} [{}/{} ({:.0f}%)] Loss: mse({:.6f}), kld({:.6f})'.format(
                    epoch,
                    opt.epochs,
                    batch_idx * dataloader.batch_size,
                    len(dataloader) * dataloader.batch_size,
                        100.0 * batch_idx / len(dataloader),
                    criterion.latestloss()['MSE'].item(),
                    criterion.latestloss()['KLD'].item()
                    ), end='\r')
            sys.stdout.write('\033[K')
    
    print("Epoch: {}/{} Loss:{:.6f}".format(epoch, opt.epochs, total_loss / len(dataloader)))
    mse_loss_list.append(mse_loss / len(dataloader))
    kld_loss_list.append(kld_loss / len(dataloader))
        
    # save
    current_loss = total_loss / len(dataloader)
    state = {
        'model': 'VAE',
        'epoch': epoch,
        'state_dict': netVAE.state_dict(),
        'optimizer': optimizer.state_dict(),
        'kld_loss': kld_loss_list,
        'mse_loss': mse_loss_list
    }

    if not os.path.exists(opt.outf):
        os.makedirs(opt.outf)

    filename = os.path.join(opt.outf, "VAE_epoch_{}.pth".format(epoch))
    best_filename = os.path.join(opt.outf, "VAE_best.pth".format(epoch))

    if epoch % opt.save_freq == 0:
        torch.save(state, f=filename)
    if min_loss > current_loss:
        torch.save(state, f=best_filename)
        print("Saving Epoch: {}, Updating loss {:.6f} to {:.6f}".format(epoch, min_loss, current_loss))
        min_loss = current_loss

