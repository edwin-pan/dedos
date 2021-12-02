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

class OutputImage(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.image_out = torch.nn.Parameter(torch.randn(1, 3, 256, 256),requires_grad=True)

    def forward(self):
        return self.image_out

class Alg2Loss(torch.nn.Module):
    def __init__(self, tau=0.5, delta=100, tv_lambda=0.001, device='cpu'):
        super().__init__()
        self.l2_loss = torch.nn.MSELoss().to(device)
        self.similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-20)
        self.tau = tau
        self.delta = delta
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

    def forward(self, x_encoded, y_convolved, gen, i, i_convolved):
        tv_term = self.tv_loss(i)
        loss =  self.l2_loss(x_encoded, i_convolved) + \
                self.tau*self.l2_loss(i, gen) + \
                self.delta*self.l2_loss(y_convolved, x_encoded) + \
                self.tv_lambda * tv_term
        return loss

class OptmizerLosses(torch.nn.Module):
    def __init__(self, cos_lambda=1, tv_lambda=1, device='cpu'):
        super().__init__()
        self.l2_loss = torch.nn.MSELoss().to(device)
        self.similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-20)
        self.cos_lambda = cos_lambda
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
        return self.l2_loss(x, y) + self.cos_lambda * cosine_term + self.tv_lambda * tv_term
            

class Optimizer:
    def __init__(self, config, use_alg2=False, cos_lambda=1, tv_lambda=0.01):
        self.config = config
        self._init_generator()
        self.num_zernike_terms = 350
        self.zernike_gen = ZernikeGenerator(self.num_zernike_terms)
        self.use_alg2 = use_alg2
        if self.use_alg2:
            self.image_out = OutputImage().cuda()
            self.loss_fn = Alg2Loss(tv_lambda=tv_lambda, device='cuda')
        else:
            self.loss_fn = OptmizerLosses(cos_lambda=cos_lambda, tv_lambda=tv_lambda, device='cuda')


    def _init_generator(self):
        self.netG, _ = get_nets(self.config['model'])
        self.netG.optimize_noise = True
        
        # TODO: Change weight path to be the OG weight path
        weight_path = './last_fpn_new_noise.h5'
        self.netG.load_state_dict(torch.load(weight_path)['model'],strict=False)
        for param in self.netG.parameters():
            param.requires_grad = True
        self.netG.cuda()
        
    def tensor2im(self, image_tensor, imtype=np.uint8, rescale=True):
            image_numpy = image_tensor[0].cpu().float().numpy()
            image_numpy = np.transpose(image_numpy, (1, 2, 0))
            if rescale: 
                image_numpy = (image_numpy + 1) / 2.0 
            image_numpy *= 255.0
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

        if self.use_alg2:
            for name, param in self.image_out.named_parameters():
                if "image_out" in name: 
                    param.data = torch.randn_like(param.data) # randomize for every sample
                    params_to_update.append(param)


        optimizer = optim.Adam(params_to_update, lr=0.01)

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

            # Convolve output image with kernel
            if self.use_alg2:
                curr_image_out = self.image_out.forward()
                alg2_term = self.zernike_gen.apply_zernike(curr_image_out, device='cuda')
                loss = self.loss_fn((encoded+1)/2, convolved, (generated+1)/2, (curr_image_out+1)/2, alg2_term)
            else:
                # Determine loss, backpropogate
                loss = self.loss_fn((encoded+1)/2, convolved, (generated+1)/2)
            loss.backward()
            optimizer.step()

            if self.use_alg2:
                generated = curr_image_out

            # plt.imsave(f'./tmp/{i:03d}.png', self.tensor2im(generated.data))
            # More record keeping
            psnr_rec = PSNR(self.tensor2im(generated.data), self.tensor2im(sharp))
            ssim_rec = SSIM(self.tensor2im(generated.data), self.tensor2im(sharp), multichannel=True)
            progress.set_postfix(loss='%.4f' % loss.item(), psnr='%.4f' % psnr_rec, ssim='%.4f' % ssim_rec)

        if self.use_alg2:
            predicted_sharp = self.image_out.forward()
        else:
            predicted_sharp = self.netG(encoded)
        psnr3 = PSNR(self.tensor2im(predicted_sharp.data), self.tensor2im(sharp))
        ssim3 = SSIM(self.tensor2im(predicted_sharp.data), self.tensor2im(sharp), multichannel=True)
        if verbose:
            print("PSNRS:", psnr1, psnr2, psnr3)
            print("SSIMS:", ssim1, ssim2, ssim3) # [original encoded vs sharp, generated vs sharp, optimized vs sharp]

        if sharp is not None:
            images = [self.tensor2im(encoded), 
                      self.tensor2im(convolved.data, rescale=False), 
                      self.tensor2im(pre_optim.data), 
                      self.tensor2im(predicted_sharp.data), 
                      self.tensor2im(sharp)]
            metrics = [[psnr1, psnr2, psnr3], [ssim1, ssim2, ssim3]]
            return images, metrics
        else:
            return self.tensor2im(predicted_sharp.data)


def main(args, config_path='./dedos/models/DeblurGANv2/config/config.yaml'):

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # train, val, test dataloader
    batchsize = 1 # use batch size 1 for optimize
    preprocess = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(256)]) 

    dataset = DeDOSDataset(os.path.join(args.root, '../../data/deblurGAN'), preprocess=preprocess)

    datasets = train_val_test_dataset(dataset)
    dataloaders = {x: DataLoader(datasets[x], batchsize, shuffle=False, num_workers=cpu_count()) for x in
                   ['train', 'val', 'test']}
    
    optimizer = Optimizer(config, cos_lambda=5, tv_lambda=0.001)
    # optimizer = Optimizer(config, cos_lambda=0, tv_lambda=0)

    split='test'
    dl = dataloaders[split]

    psnr_record = []
    ssim_record = []
    eval_dir = create_eval_dir(os.path.join(args.root, 'outputs'), name=args.name)

    for idx, (encoded, sharp) in enumerate(dl):
        encoded, sharp = encoded.cuda(), sharp.cuda()

        # Run image
        out_images, out_metrics = optimizer.optimize(encoded, sharp=sharp, num_steps=args.num_steps, verbose=True)

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
    
    