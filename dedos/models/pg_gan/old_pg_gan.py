"""
This work is based on the Theano/Lasagne implementation of
Progressive Growing of GANs paper from tkarras:
https://github.com/tkarras/progressive_growing_of_gans
PyTorch Model definition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from utils import *


class PixelNormLayer(nn.Module):
    def __init__(self):
        super(PixelNormLayer, self).__init__()

    def forward(self, x):
        return x /torch.sqrt(torch.mean(x**2, dim=1, keepdim = True) + 1e-8)


class WScaleLayer(nn.Module):
    def __init__(self, size):
        super(WScaleLayer, self).__init__()
        self.scale = nn.Parameter(torch.randn([1]))
        self.b = nn.Parameter(torch.randn(size))
        self.size = size

    def forward(self, x):
        x_size = x.size()
        x = x * self.scale + self.b.view(1, -1, 1, 1).expand(
            x_size[0], self.size, x_size[2], x_size[3])

        return x


class NormConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(NormConvBlock, self).__init__()
        self.norm = PixelNormLayer()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.wscale = WScaleLayer(out_channels)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        x = F.leaky_relu(self.wscale(x), negative_slope=0.2)
        return x


class NormUpscaleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(NormUpscaleConvBlock, self).__init__()
        self.norm = PixelNormLayer()
        self.up = nn.Upsample(scale_factor=2,mode='nearest')
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.wscale = WScaleLayer(out_channels)

    def forward(self, x):
        x = self.norm(x)
        x = self.up(x)
        x = self.conv(x)
        x = F.leaky_relu(self.wscale(x), negative_slope=0.2)
        return x


class PGGenerator(nn.Module):
    def __init__(self):
        super(PGGenerator, self).__init__()

        self.features = nn.Sequential(
            NormConvBlock(512, 512, kernel_size=4, padding=3),
            NormConvBlock(512, 512, kernel_size=3, padding=1),
            NormUpscaleConvBlock(512, 512, kernel_size=3, padding=1),
            NormConvBlock(512, 512, kernel_size=3, padding=1),
            NormUpscaleConvBlock(512, 256, kernel_size=3, padding=1),
            NormConvBlock(256, 256, kernel_size=3, padding=1),
            NormUpscaleConvBlock(256, 128, kernel_size=3, padding=1),
            NormConvBlock(128, 128, kernel_size=3, padding=1),
            NormUpscaleConvBlock(128, 64, kernel_size=3, padding=1),
            NormConvBlock(64, 64, kernel_size=3, padding=1),
            NormUpscaleConvBlock(64, 32, kernel_size=3, padding=1),
            NormConvBlock(32, 32, kernel_size=3, padding=1))

        self.output = nn.Sequential(OrderedDict([
                        ('norm', PixelNormLayer()),
                        ('conv', nn.Conv2d(32,
                                           3,
                                           kernel_size=1,
                                           padding=0,
                                           bias=False)),
                        ('wscale', WScaleLayer(3))
                    ]))

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        return x

if __name__ == "__main__":
    CelebA_Gen = PGGenerator()
    CelebA_Gen.load_state_dict(torch.load('./weights/pggan_generator.pth'))
    CelebA_Gen.cuda()
    
    N = 10
    f, ax  = plt.subplots(1,N, figsize=(15,5))
    Latent = np.random.normal(0,1,(N,512,1,1))
    Gen_Images = []
    for i ,data in zip(np.arange(N), Latent):
        data = np.expand_dims(data, axis=0)
        data = torch.Tensor(data).cuda()
        celebA_gen_imgs = CelebA_Gen(data).data.cpu().numpy()
        celebA_gen_imgs = np.transpose(celebA_gen_imgs,[0,2,3,1] )[0]
        Gen_Images.append(celebA_gen_imgs)
        ax[i].imshow(scale_image(celebA_gen_imgs))
        ax[i].axis('Off')
    plt.show()