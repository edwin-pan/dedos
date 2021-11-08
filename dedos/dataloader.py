import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import os.path as osp
import os
import cv2
import glob

from metrics import Metrics

class CustomDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, datapath, augment=False):
        'Initialization'
        # Paths
        self.datapath = datapath
        self.datapath_e = osp.join(self.datapath, 'blur') # Blurry images are encoded
        self.datapath_s = osp.join(self.datapath, 'sharp')
        
        # Flags
        self.augment = augment
        
        # Utils
        self.image_size = (432, 368, 3)
        self.preprocess = transforms.Compose([transforms.ToTensor()]) # [C, H, W]
        self.augmentation = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) # Z-Score Norm Using Imagenet
        
        # Grab images
        self.init_image_paths()
        return
      
    def init_image_paths(self):
        self.e_images = glob.glob(osp.join(self.datapath_e, '*.jpg'))
        self.s_images = glob.glob(osp.join(self.datapath_s, '*.jpg'))
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
        
        encoded = cv2.cvtColor(cv2.imread(encoded), cv2.COLOR_BGR2RGB)
        sharp = cv2.cvtColor(cv2.imread(sharp), cv2.COLOR_BGR2RGB)
        
        encoded = self.preprocess(encoded)
        sharp = self.preprocess(sharp)
        
        if self.augment:
        #TODO: Maybe?
            pass
        
        return encoded, sharp

if __name__ == '__main__':
    dataset = CustomDataset('../../../data/deblurGAN/')
    dl = torch.utils.data.DataLoader(dataset, batch_size=2)
    
    metrics = Metrics(device='cpu')

    sample = next(iter(dl))

    out = metrics(sample[1],sample[0])

    print(out)
    print(sample[0].shape)
    plt.imshow(sample[0][0].permute(1,2,0))
    plt.savefig('./blur.png')
    plt.imshow(sample[1][0].permute(1,2,0))
    plt.savefig('./sharp.png')

