# Originally from: https://github.com/nashory/pggan-pytorch
# Adapted by: Edwin Pan (edwinpan@stanford.edu)
# November 10th, 2021

import os
import torch as torch
import numpy as np
import sys
from io import BytesIO
import scipy.misc
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from matplotlib import pyplot as plt
from PIL import Image

from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.append('../../../') # importing in unit tests
from dedos.dataloader import DeDOSDataset
from dedos.models.pg_gan.config import config
 
def train_val_test_dataset(dataset, val_split=0.125, test_split=0.1):
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=test_split, random_state=1)
    train_idx, val_idx= train_test_split(train_idx, test_size=val_split, random_state=1)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    datasets['test'] = Subset(dataset, test_idx)
    return datasets

class dataloader:
    def __init__(self, config):
        self.root = config.train_data_root
        self.batch_table = {4:32, 8:32, 16:32, 32:16, 64:16, 128:16, 256:12, 512:3, 1024:1} # change this according to available gpu memory.
        self.batchsize = int(self.batch_table[pow(2,2)])        # we start from 2^2=4
        self.imsize = int(pow(2,2))
        self.num_workers = 4


    def renew(self, resl):
        print('[*] Renew dataloader configuration, load data from {}.'.format(self.root))
        
        self.batchsize = int(self.batch_table[pow(2,resl)])
        self.imsize = int(pow(2,resl))

        self.preprocess = transforms.Compose([transforms.CenterCrop(256),
                                              transforms.Resize(size=(self.imsize,self.imsize), interpolation=transforms.InterpolationMode.NEAREST),
                                              transforms.ToTensor()])

        self.dataset = DeDOSDataset(self.root, preprocess=self.preprocess)
        self.datasets = train_val_test_dataset(self.dataset)
        self.dataloaders = {x:DataLoader(self.datasets[x],self.batchsize, shuffle=True, num_workers=self.num_workers) for x in ['train','val', 'test']}
        self.dataloader = self.dataloaders['train']

    def __iter__(self):
        return iter(self.dataloader)
    
    def __next__(self):
        return next(self.dataloader)

    def __len__(self):
        return len(self.dataloader.dataset)

    def get_batch(self):
        dataIter = iter(self.dataloader)
        return next(dataIter)[1]#.mul(2).add(-1)         # pixel range [-1, 1], only take sharp images

if __name__ == '__main__':
    batchsize=2
    num_workers=4
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using: {device}")

    # loader = dataloader(config)
    # loader.renew(min(floor(self.resl), self.max_resl))
    # sample = loader.get_batch()
    

    pass