from __future__ import print_function
import argparse
import os
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
from utils import weights_init, compute_acc
from network import _netG, _netD
from folder import ImageFolder


##### Command example
# python3 test.py --testroot ../sample_test/sample_fid_testing_labels.txt --outf ./test/ACGAN20190527 --netG ./result/ACGAN20190527/netG_epoch_235.pth --netD ./result/ACGAN20190527/netD_epoch_235.pth --cuda

parser = argparse.ArgumentParser()
parser.add_argument('--testroot', required=True, help='path to test file')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=110, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--netG', default='', help="path to netG model")
parser.add_argument('--netD', default='', help="path to netD model")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, default=1126, help='manual seed')
parser.add_argument('--num_classes', type=int, default=15, help='Number of classes for AC-GAN')

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

# some hyper parameters
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
num_classes = int(opt.num_classes)
nc = 3

# Define the generator and initialize the weights
netG = _netG(ngpu, nz)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

# Define the discriminator and initialize the weights
netD = _netD(ngpu, num_classes)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

# tensor placeholders
test_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)

# if using cuda
if opt.cuda:
    netD.cuda()
    netG.cuda()
    test_noise = test_noise.cuda()

# define variables
test_noise = Variable(test_noise)

# read test-file
test_multi_hots = []
with open(opt.testroot, 'r') as testf:
    lines = testf.readlines()
    for id, line in enumerate(lines):
        if id > 1:
            line = line.split()
            line = [[float(i) for i in line]]
            test_multi_hots.append(line)


for id, multi_hot in enumerate(test_multi_hots):
    test_noise_ = np.random.normal(0, 1, (opt.batchSize, nz))
    test_noise_[np.arange(opt.batchSize), :num_classes] = multi_hot
    # test_noise_[np.arange(opt.batchSize), :num_classes] = multi_hot[np.arange(opt.batchSize)]
    test_noise_ = (torch.from_numpy(test_noise_))
    test_noise.data.copy_(test_noise_.view(opt.batchSize, nz, 1, 1))
    fake = netG(test_noise)
    mean = (0.5,0.5,0.5)
    std = (0.5,0.5,0.5)
    for t, m, s in zip(fake, mean, std):
        t.mul_(s).add_(m)
    vutils.save_image(
        fake.data,'{}/{}.png'.format(opt.outf, id)
    )
    print('{}/{}.png saved.'.format(opt.outf, id), end='\r')
