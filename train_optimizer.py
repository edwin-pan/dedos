import torch
from torchvision import transforms
import yaml
from torch.utils.data import DataLoader
from joblib import cpu_count
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from skimage.metrics import structural_similarity as SSIM
import piqa
import matplotlib.pyplot as plt

from dedos.dataloader import DeDOSDataset, train_val_test_dataset
from dedos.models.zernike.kernel_gen import ZernikeGenerator
from dedos.models.DeblurGANv2.models.networks import get_nets
from dedos.models.DeblurGANv2.util.metrics import PSNR


class Optimizer:
    def __init__(self, config, dataloaders):
        self.config = config
        self._init_generator()
        self.num_zernike_terms = 350
        self.dataloaders = dataloaders
        self.zernike_gen = ZernikeGenerator(self.num_zernike_terms)
        
    def tensor2im(self, image_tensor, imtype=np.uint8):
            image_numpy = image_tensor[0].cpu().float().numpy()
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
            return image_numpy.astype(imtype)

    def optimize(self, split="test"):
        dl = self.dataloaders[split]
        l2_loss = torch.nn.MSELoss().cuda()
        lpips = piqa.LPIPS().cuda()
        
        for idx, (encoded, sharp) in enumerate(dl):
            encoded, sharp = encoded.cuda(), sharp.cuda()
            # Initialize noise vectors in generator randomly (done automatically in init)
            # Initialize scaled weights for zernike randomly
            # Initialize an optimizer for just this sample
            psnr1 = PSNR(self.tensor2im(encoded), self.tensor2im(sharp))
            ssim1 = SSIM(self.tensor2im(encoded), self.tensor2im(sharp), multichannel=True)

            self.zernike_gen = ZernikeGenerator(self.num_zernike_terms)
            params_to_update = []
            for name, param in self.netG.named_parameters():
                if "noise_val" in name: 
                    param.data = torch.randn_like(param.data) # randomize for every sample
                    params_to_update.append(param)
            
            for name, param in self.zernike_gen.named_parameters():
                if "zernike_weights" in name:
                    params_to_update.append(param)
                    
            optimizer = optim.Adam(params_to_update, lr=0.1)
            progress = tqdm(range(1000))
            for i in progress:
                self.netG.zero_grad()
                self.zernike_gen.zero_grad()
                
                generated = self.netG(encoded)
                kernel = self.zernike_gen().unsqueeze(0).cuda()
                kernel = kernel/kernel.sum()
                
                if i == 0:
                    temp = generated.clone()
                    psnr2 = PSNR(self.tensor2im(generated.data), self.tensor2im(sharp))
                    ssim2 = SSIM(self.tensor2im(generated.data), self.tensor2im(sharp), multichannel=True)

                # Convolve generated with the kernel
                convolved = torch.zeros_like(generated)
                gen_convolved = (generated + 1) /2 
                for channel in [0, 1, 2]:
                    fourier = torch.fft.fft2(gen_convolved[:, channel, :, :].squeeze(0)) * torch.fft.fft2(kernel)
                    convolved[:, channel, :, :] = torch.real(torch.fft.ifft2(fourier)).unsqueeze(0)
                
                # Todo: figure out if this is legal
                convolved = torch.clip(convolved, 0, 1)

                # loss = l2_loss(encoded, convolved)
                loss = lpips((encoded+1)/2, convolved)

                progress.set_postfix(loss='%.4f' % loss.item())

                # for name, param in self.netG.named_parameters():
                #     if "noise_val4_" in name: 
                #         print(f"{name}: {param.reshape(-1)}")
                # for name, param in self.zernike_gen.named_parameters():
                #     if "zernike_weights" in name: 
                #         print(f"{name}: {param[-1].detach().numpy()}")

                # Forward pass this image through the generator
                # Compute the current filter from zernike
                # Determine loss
                loss.backward()
                optimizer.step()

            # Plots
            predicted_sharp = self.netG(encoded)
            psnr3 = PSNR(self.tensor2im(predicted_sharp.data), self.tensor2im(sharp))
            ssim3 = SSIM(self.tensor2im(predicted_sharp.data), self.tensor2im(sharp), multichannel=True)
            print("PSNRS:", psnr1, psnr2, psnr3)
            print("SSIMS:", ssim1, ssim2, ssim3) # [original encoded vs sharp, generated vs sharp, optimized vs sharp]

            plt.figure()
            plt.subplot(141)
            plt.imshow(self.tensor2im(encoded))
            plt.axis('off')
            plt.subplot(142)
            plt.imshow(self.tensor2im(temp.data))
            plt.axis('off')
            plt.subplot(143)
            plt.imshow(self.tensor2im(predicted_sharp.data))
            plt.axis('off')
            plt.subplot(144)
            plt.imshow(self.tensor2im(sharp))
            plt.axis('off')
            plt.savefig(f'outputs/test_output_{idx}.png')
        
    def _init_generator(self):
        self.netG, _ = get_nets(self.config['model'])
        self.netG.optimize_noise = True
        
        # TODO: Change weight path to be the OG weight path
        weight_path = './last_fpn.h5'
        self.netG.load_state_dict(torch.load(weight_path)['model'],strict=False)
        for param in self.netG.parameters():
            param.requires_grad = True
        
        self.netG.cuda()



def main(config_path='./dedos/models/DeblurGANv2/config/config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # train, val, test dataloader
    batchsize = 1 # use batch size 1 for optimize
    preprocess = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(256)]) 

    # dataset = DeDOSDataset('/scratch/groups/kpohl/dedos/deblurGAN', preprocess=preprocess)
    dataset = DeDOSDataset('/home/edwin/research/data/deblurGAN', preprocess=preprocess)

    datasets = train_val_test_dataset(dataset)
    dataloaders = {x: DataLoader(datasets[x], batchsize, shuffle=True, num_workers=cpu_count()) for x in
                   ['train', 'val', 'test']}
    
    optimizer = Optimizer(config, dataloaders)
    optimizer.optimize()
    
    
    
    
    
if __name__ == "__main__":
    main()
    
    