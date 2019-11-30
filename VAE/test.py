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
from network import VAE

##### Command example
# python3 test.py --outf ./test/VAE20190529 --netVAE ./result/VAE20190529/VAE_best.pth --cuda

parser = argparse.ArgumentParser()
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--netVAE', default='', help="path to netVAE model")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, default=1126, help='manual seed')

# GPU setting
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--gpu_id', type=int, default=0, help='The ID of the specified GPU')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

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

# model
netVAE = VAE('test')
if opt.netVAE != '':
    netVAE.load_state_dict(torch.load(opt.netVAE)['state_dict'])
    print('loading model ....')
netVAE.eval()
print(netVAE)

if opt.cuda:
    netVAE.cuda()

print("Saving random generation......")
decoder = netVAE.decoder

# method 1
noise_1 = torch.randn(1024, 5, 5)
noise_2 = torch.randn(1024, 5, 5)
noise = []
for i in range(32):
    noise.append((i/31)*noise_2 + ((31-i)/31)*noise_1)
noise = Variable(torch.stack(noise,0))
if opt.cuda:
    noise = noise.cuda()
predict = decoder(noise)
print(predict.shape)
predict = predict.mul(0.5).add_(0.5)
filename = os.path.join(opt.outf, 'interpolation.jpg')
vutils.save_image(predict.data, filename, nrow=8)

# method 2
noise = torch.randn(32*25, 1024).view(-1, 1024, 5, 5)        
noise = Variable(noise)
if opt.cuda:
    noise = noise.cuda()
predict = decoder(noise)
print(predict.shape)
predict = predict.mul(0.5).add_(0.5)
filename = os.path.join(opt.outf, 'noise.jpg')
vutils.save_image(predict.data, filename, nrow=8)


# method 3
# TBR
# many faces result QQ
# mu = torch.randn(32*25, 1024).view(-1, 1024, 5, 5)
# logvars = [torch.zeros((32*25, 1024)).view(-1, 1024, 5, 5)]
# for i in range(1, 10):
#     logvars.append(torch.ones((32*25, 1024)).view(-1, 1024, 5, 5).mul_(float(i)*0.1))
# mu = Variable(mu)
# for i, logvar in enumerate(logvars):
#     logvar = Variable(logvar)
#     if opt.cuda:
#         mu = mu.cuda()
#         logvar = logvar.cuda()
#     predict = netVAE.inference(mu, logvar)
#     print(predict.shape)
#     predict = predict.mul(0.5).add_(0.5)
#     filename = os.path.join(opt.outf, 'anime_{}.jpg'.format(i))
#     vutils.save_image(predict.data, filename, nrow=8)


# method 4
# TODO:
# read 1 image
# get mu and logvar
# different logvar see diff results

# mu = torch.randn(32*25, 1024).view(-1, 1024, 5, 5)
# logvars = [torch.zeros((32*25, 1024)).view(-1, 1024, 5, 5)]
# for i in range(1, 10):
#     logvars.append(torch.ones((32*25, 1024)).view(-1, 1024, 5, 5).mul_(float(i)*0.1))
# mu = Variable(mu)
# for i, logvar in enumerate(logvars):
#     logvar = Variable(logvar)
#     if opt.cuda:
#         mu = mu.cuda()
#         logvar = logvar.cuda()
#     predict = netVAE.inference(mu, logvar)
#     print(predict.shape)
#     predict = predict.mul(0.5).add_(0.5)
#     filename = os.path.join(opt.outf, 'anime_{}.jpg'.format(i))
#     vutils.save_image(predict.data, filename, nrow=8)
