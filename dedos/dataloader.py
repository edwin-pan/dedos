import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from PIL import Image

import matplotlib.pyplot as plt
import os.path as osp
import os
import glob
import sys

sys.path.append('../') # importing in unit tests
from dedos.metrics import Metrics

def train_val_test_dataset(dataset, val_split=0.125, test_split=0.1):
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=test_split, random_state=1)
    train_idx, val_idx= train_test_split(train_idx, test_size=val_split, random_state=1)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    datasets['test'] = Subset(dataset, test_idx)
    return datasets
    
class DeDOSDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, datapath, augment=False, preprocess=None):
        'Initialization'
        # Paths
        self.datapath = datapath
        self.datapath_e = osp.join(self.datapath, 'blur') # Blurry images are encoded
        self.datapath_s = osp.join(self.datapath, 'sharp')
        
        # Flags
        self.augment = augment
        
        # Utils
        self.image_size = (432, 368, 3)
        if preprocess==None:
            self.preprocess = transforms.Compose([transforms.ToTensor()]) # [C, H, W]
        else:
            self.preprocess = preprocess
        self.augmentation = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) # Z-Score Norm Using Imagenet
        
        # Grab images
        self.init_image_paths()
        return
      
    def init_image_paths(self):
        self.e_images = sorted(glob.glob(osp.join(self.datapath_e, '*.jpg')))
        self.s_images = sorted(glob.glob(osp.join(self.datapath_s, '*.jpg')))
        self.num_e_images = len(self.e_images)
        self.num_s_images = len(self.s_images)
        
        assert self.num_e_images == self.num_s_images, f"Missing some files {self.num_e_images}!={self.num_s_images}" 
        return


    def __len__(self):
        '''Denotes the total number of samples'''
        return self.num_e_images

        
    def __getitem__(self, index):
        'Generates an encoded and sharp sample of data'        
        # Get encoded and sharp image
        encoded = self.e_images[index] # blurry now has path to chosen blurry image
        sharp = self.s_images[index] # sharp now has path to chosen sharp image
        
        encoded = Image.open(encoded)
        sharp = Image.open(sharp)
        
        encoded = self.preprocess(encoded)
        sharp = self.preprocess(sharp)
        
        if self.augment:
        #TODO: Maybe?
            pass
        
        return encoded, sharp

if __name__ == '__main__':
    batchsize=2
    num_workers=4

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using: {device}")

    preprocess = transforms.Compose([transforms.Resize(size=(256,256), interpolation=transforms.InterpolationMode.NEAREST), # Using bilinear interpolation
                                     transforms.ToTensor()])

    dataset = DeDOSDataset('../../../data/deblurGAN/', preprocess=preprocess)
    datasets = train_val_test_dataset(dataset)
    dataloaders = {x:DataLoader(datasets[x],batchsize, shuffle=True, num_workers=num_workers) for x in ['train','val','test']}
    dl = dataloaders['train']
    metrics = Metrics(device=device)

    blurry, sharp = next(iter(dl))
    blurry = blurry.to(device)
    sharp = sharp.to(device)
    out = metrics(blurry,sharp)

    print(out)
    print(blurry.shape)
    plt.imshow(blurry[0].cpu().permute(1,2,0))
    plt.savefig('./blur.png')
    plt.imshow(sharp[0].cpu().permute(1,2,0))
    plt.savefig('./sharp.png')

