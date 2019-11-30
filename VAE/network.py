import torch
import torch.nn as nn
from torch.autograd import Variable


class VAE(nn.Module):
    def __init__(self, mode='train'):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.sigma = nn.Conv2d(256, 1024, kernel_size=4, bias=False)
        self.mu = nn.Conv2d(256, 1024, kernel_size=4, bias=False)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=4, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=False)
        )

        self.output = nn.Tanh()
        self.mode = mode

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(torch.cuda.FloatTensor(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.mu(encoded)
        logvar = self.sigma(encoded)
        z = self.reparameterize(mu, logvar)
        output = self.output(self.decoder(z))
        return output, mu, logvar

    def inference(self, mu, logvar):
        z = self.reparameterize(mu, logvar)
        output = self.output(self.decoder(z))
        return output
