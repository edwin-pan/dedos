from genericpath import exists
from numpy.core.numeric import convolve
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
import argparse
import PIL
import os

from dedos.dataloader import DeDOSDataset, train_val_test_dataset
from dedos.models.zernike.kernel_gen import ZernikeGenerator, ZernikeLosses
from dedos.models.DeblurGANv2.models.networks import get_nets
from dedos.models.DeblurGANv2.util.metrics import PSNR
from dedos.utils import create_eval_dir


def parse_args():
    parser = argparse.ArgumentParser(description='Baseline MetaChex: Fine-Tuned ChexNet')
    parser.add_argument('-p', '--plot', action='store_true', help='Periodically generate plots')
    parser.add_argument('-i', '--save_images', action='store_true', help='Save individual images')
    parser.add_argument('-r', '--root', type=str, default='/home/edwin/research/code/dedos', help='Path to dedos directory')
    parser.add_argument('-a', '--name', type=str, default='default', help='Name of experiment')
    parser.add_argument('-s', '--num_steps', type=int, default=1000, help='Number of iterations to optimize over')
    parser.add_argument('-n', '--num_samples', type=int, default=15, help='Number of test samples to run on')

    return parser.parse_args()


class OptmizerLosses(torch.nn.Module):
    def __init__(self, sim_lambda=1, tv_lambda=1, device='cpu'):
        super().__init__()
        self.l2_loss = torch.nn.MSELoss().to(device)
        self.similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-20)
        self.sim_lambda = sim_lambda
        self.tv_lambda = tv_lambda

    def tv_loss(self, img):
        """
        Compute total variation loss. Adapted from CS231n
        Inputs:
        - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
        - tv_weight: Scalar giving the weight w_t to use for the TV loss.
        Returns:
        - loss: PyTorch Variable holding a scalar giving the total variation loss
        for img weighted by tv_weight.
        """
        w_variance = torch.sum(torch.pow(img[:,:,:,:-1] - img[:,:,:,1:], 2))
        h_variance = torch.sum(torch.pow(img[:,:,:-1,:] - img[:,:,1:,:], 2))
        loss =  h_variance + w_variance
        return loss

    def forward(self, x, y, gen):
            cosine_term = (1 - self.similarity(x, y)).mean()
            tv_term = self.tv_loss(gen)
            return self.l2_loss(x, y) + self.sim_lambda * cosine_term + self.tv_lambda * tv_term
            

class Optimizer:
    def __init__(self, config, dataloaders):
        self.config = config
        self._init_generator()
        self.num_zernike_terms = 350
        self.dataloaders = dataloaders
        self.zernike_gen = ZernikeGenerator(self.num_zernike_terms)
        self.loss_fn = OptmizerLosses(tv_lambda=0.01, device='cuda')

    def _init_generator(self):
        self.netG, _ = get_nets(self.config['model'])
        self.netG.optimize_noise = True
        
        # TODO: Change weight path to be the OG weight path
        weight_path = './last_fpn.h5'
        self.netG.load_state_dict(torch.load(weight_path)['model'],strict=False)
        for param in self.netG.parameters():
            param.requires_grad = True
        
        self.netG.cuda()

        
    def tensor2im(self, image_tensor, imtype=np.uint8):
            image_numpy = image_tensor[0].cpu().float().numpy()
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
            return image_numpy.astype(imtype)


    def optimize(self, encoded, sharp=None, num_steps=1000, verbose=False):
        """Post-process optimization on 1 image.
        
            Parameters
            ----------
            encoded: real 2d `numpy.ndarray`
                Input encoded image
            sharp: real 2d `numpy.ndarray`
                Corresponding sharp image. Used for computing metrics.
            num_steps : int, optional
                Number of iterations to optimize for.
            verbose: bool, optional
                Print out stages and evaluations metric results.
            Returns
            -------
        
        """
        # Initialize noise vectors in generator randomly (done automatically in init)
        # Initialize scaled weights for zernike randomly
        self.zernike_gen = ZernikeGenerator(self.num_zernike_terms)
        self.zernike_gen.load_state_dict(torch.load(self.zernike_gen.init_file_path))
        # Initialize an optimizer for just this sample
        params_to_update = []
        for name, param in self.netG.named_parameters():
            if "noise_val" in name: 
                param.data = torch.randn_like(param.data) # randomize for every sample
                params_to_update.append(param)
        
        for name, param in self.zernike_gen.named_parameters():
            if "zernike_weights" in name:
                params_to_update.append(param)
        optimizer = optim.Adam(params_to_update, lr=0.1)

        # Record metrics (baseline encoded vs sharp)
        psnr1 = PSNR(self.tensor2im(encoded), self.tensor2im(sharp))
        ssim1 = SSIM(self.tensor2im(encoded), self.tensor2im(sharp), multichannel=True)

        # Optimize
        progress = tqdm(range(num_steps))
        for i in progress:
            self.netG.zero_grad()
            self.zernike_gen.zero_grad()
            
            # Forward pass this image through the generator
            generated = self.netG(encoded)

            # Compute the current filter from zernike
            kernel = self.zernike_gen().unsqueeze(0).cuda()
            kernel = kernel/kernel.sum()

            # Record keeping for plots
            if i == 0:
                pre_optim = generated.clone()
                psnr2 = PSNR(self.tensor2im(generated.data), self.tensor2im(sharp))
                ssim2 = SSIM(self.tensor2im(generated.data), self.tensor2im(sharp), multichannel=True)

            # Convolve generated with the kernel
            convolved = self.zernike_gen.apply_zernike(generated, device='cuda')

            # Determine loss, backpropogate
            loss = self.loss_fn((encoded+1)/2, convolved, (generated+1)/2)
            loss.backward()
            optimizer.step()

            # More record keeping
            progress.set_postfix(loss='%.4f' % loss.item())

        predicted_sharp = self.netG(encoded)
        psnr3 = PSNR(self.tensor2im(predicted_sharp.data), self.tensor2im(sharp))
        ssim3 = SSIM(self.tensor2im(predicted_sharp.data), self.tensor2im(sharp), multichannel=True)
        if verbose:
            print("PSNRS:", psnr1, psnr2, psnr3)
            print("SSIMS:", ssim1, ssim2, ssim3) # [original encoded vs sharp, generated vs sharp, optimized vs sharp]

        if sharp is not None:
            images = [self.tensor2im(encoded), 
                      self.tensor2im(convolved.data), 
                      self.tensor2im(pre_optim.data), 
                      self.tensor2im(predicted_sharp.data), 
                      self.tensor2im(sharp)]
            metrics = [[psnr1, psnr2, psnr3], [ssim1, ssim2, ssim3]]
            return images, metrics
        else:
            return self.tensor2im(predicted_sharp.data)


def add2avg(average, size, value):
    return (size * average + value) / (size + 1)


def main(args, config_path='./dedos/models/DeblurGANv2/config/config.yaml'):

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # train, val, test dataloader
    batchsize = 1 # use batch size 1 for optimize
    preprocess = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(256)]) 

    dataset = DeDOSDataset(os.path.join(args.root, '../../data/deblurGAN'), preprocess=preprocess)

    datasets = train_val_test_dataset(dataset)
    dataloaders = {x: DataLoader(datasets[x], batchsize, shuffle=True, num_workers=cpu_count()) for x in
                   ['train', 'val', 'test']}
    
    optimizer = Optimizer(config, dataloaders)

    split='test'
    dl = dataloaders[split]

    psnr_record = []
    ssim_record = []
    eval_dir = create_eval_dir(os.path.join(args.root, 'outputs'), name=args.name)

    for idx, (encoded, sharp) in enumerate(dl):
        encoded, sharp = encoded.cuda(), sharp.cuda()

        # Run image
        out_images, out_metrics = optimizer.optimize(encoded, sharp=sharp, num_steps=args.num_steps)

        # Unpack outputs
        encoded, convolved, pre_optim, predicted_sharp, sharp = out_images
        psnrs, ssims = out_metrics

        psnr_record.append(psnrs)
        ssim_record.append(ssims)

        if args.save_images:
            optim_path = os.path.join(eval_dir, f'{idx:06d}')
            os.makedirs(optim_path, exist_ok=True)
            plt.imsave(os.path.join(optim_path, 'encoded.png'), encoded)
            plt.imsave(os.path.join(optim_path, 'convolved.png'), convolved)
            plt.imsave(os.path.join(optim_path, 'pre_optim.png'), pre_optim)
            plt.imsave(os.path.join(optim_path, 'predicted_sharp.png'), predicted_sharp)
            plt.imsave(os.path.join(optim_path, 'sharp.png'), sharp)
            
        if args.plot:
            # Plots
            plt.figure()
            plt.subplot(151)
            plt.title("Encoded")
            plt.imshow(encoded)
            plt.axis('off')
            plt.subplot(152)
            plt.title("Conv")
            plt.imshow(convolved)
            plt.axis('off')
            plt.subplot(153)
            plt.title("DBGv2+N")
            plt.imshow(pre_optim)
            plt.axis('off')
            plt.subplot(154)
            plt.title("P-Sharp")
            plt.imshow(predicted_sharp)
            plt.axis('off')
            plt.subplot(155)
            plt.title("Sharp")
            plt.imshow(sharp)
            plt.axis('off')
            plt.savefig(os.path.join(eval_dir, f'test_output_{idx}.png'))

        if idx+1 >= args.num_samples:
            break
    
    psnr_record_arr = np.array(psnr_record)
    ssim_record_arr = np.array(ssim_record)

    # Write evaluation results to npy
    np.save(os.path.join(eval_dir, 'psnr_record.npy'), psnr_record_arr)
    np.save(os.path.join(eval_dir, 'ssim_record.npy'), ssim_record_arr)

    # Write evaluation avg results to txt
    with open(os.path.join(eval_dir, 'avg_metrics.txt'), 'w') as f:
        f.write("[original encoded vs sharp, generated (pre_optim) vs sharp, optimized vs sharp]\n")
        f.write(f"PSNRS ({args.num_samples}-avg): {psnr_record_arr.mean(axis=0)}\n")
        f.write(f"SSIMS ({args.num_samples}-avg): {ssim_record_arr.mean(axis=0)}\n")
    f.close()

    # Print results
    print(f"PSNRS ({args.num_samples}-avg): {psnr_record_arr.mean(axis=0)}")
    print(f"SSIMS ({args.num_samples}-avg): {ssim_record_arr.mean(axis=0)}")
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
    
    