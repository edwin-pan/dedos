import torch
from torchvision import transforms
import yaml
from torch.utils.data import DataLoader
from joblib import cpu_count
import torch.optim as optim
from scipy.signal import convolve2d



from dedos.dataloader import DeDOSDataset, train_val_test_dataset
from dedos.models.zernike.kernel_gen import ZernikeGenerator
from dedos.models.DeblurGANv2.models.networks import get_nets


class Optimizer:
    def __init__(self, config, dataloaders):
        self.config = config
        self._init_generator()
        self.num_zernike_terms = 350
        self.dataloaders = dataloaders
        self.zernike_gen = ZernikeGenerator(self.num_zernike_terms)
        
        
    def optimize(self, dl="train"):
        train_dl = self.dataloaders[dl]
        l2_loss = torch.nn.MSELoss()
        
        for encoded, sharp in train_dl:
            # Initialize noise vectors in generator randomly (done automatically in init)
            # Initialize scaled weights for zernike randomly
            # Initialize an optimizer for just this sample
            
            self.zernike_gen = ZernikeGenerator(self.num_zernike_terms)
            params_to_update = []
            for name, param in self.netG.named_parameters():
                if "noise_val" in name: 
                    param.data = torch.randn_like(param.data) # randomize for every sample
                    params_to_update.append(param)
            
            for name, param in self.zernike_gen.named_parameters():
                if "zernike_weights" in name:
                    params_to_update.append(param)
                    
            optimizer = optim.Adam(params_to_update, lr=0.001)
            for _ in range(100):
                self.netG.zero_grad()
                self.zernike_gen.zero_grad()
                
                generated = self.netG(encoded)
                kernel = self.zernike_gen()
                
                # Convolve generated with the kernel
                convolved = convolve2d(generated, kernel, mode='same', boundary='fill')
                loss = l2_loss(encoded, convolved)
                # Forward pass this image through the generator
                # Compute the current filter from zernike
                # Determine loss
                loss.backward()
                optimizer.step()

            predicted_sharp = self.netG(encoded)
        
        
    def _init_generator(self):
        self.netG, _ = get_nets(self.config['model'])
        self.netG.optimize_noise = True
        
        # TODO: Change weight path to be the OG weight path
        weight_path = '/scratch/users/avento/dedos_vals/2021-11-25/dedos_vals/dedos_weights/best_fpn.h5'
        self.netG.load_state_dict(torch.load(weight_path)['model'],strict=False);
        for param in self.netG.parameters():
            param.requires_grad = True
        
        self.netG.cuda()


        

def main(config_path='./dedos/models/DeblurGANv2/config/config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # train, val, test dataloader
    batchsize = 1 # use batch size 1 for optimize
    preprocess = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(256)])
    dataset = DeDOSDataset('/scratch/groups/kpohl/dedos/deblurGAN', preprocess=preprocess)
    datasets = train_val_test_dataset(dataset)
    dataloaders = {x: DataLoader(datasets[x], batchsize, shuffle=True, num_workers=cpu_count()) for x in
                   ['train', 'val', 'test']}
    
    optimizer = Optimizer(config, dataloaders)
    optimizer.optimize()
    
    
    
    
    
if __name__ == "__main__":
    main()
    
    