import poppy
import os
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import tqdm
from joblib import cpu_count
import torch.optim as optim

from torch.utils.data import DataLoader
from dedos.dataloader import DeDOSDataset, train_val_test_dataset
from dedos.models.DeblurGANv2.util.metrics import PSNR

def gaussian_fn(M, std):
    n = torch.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    return w

def gkern(kernlen=256, std=128):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = gaussian_fn(kernlen, std=std) 
    gkern2d = torch.outer(gkern1d, gkern1d)
    return gkern2d

class ZernikeLosses(torch.nn.Module):
    def __init__(self, loss_lambda=1, device='cpu'):
        super().__init__()
        self.l2_loss = torch.nn.MSELoss().to(device)
        self.similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-20)
        self.loss_lambda = loss_lambda

    def forward(self, x, y):
            cosine_term = (1 - self.similarity(x, y)).mean()
            return self.l2_loss(x, y) + self.loss_lambda * cosine_term

class ZernikeGenerator(torch.nn.Module):
    def __init__(self, num_terms, save_path='./', patch_size=256, scale_factor=1e-6, verbose=False):
        super().__init__()
        self.num_terms = num_terms
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.save_path = save_path
        self.verbose = verbose
        self.save_fname = os.path.join(save_path, 'zernike_volume_%d.npy'%self.patch_size)
        # Generating the zernike basis functions is expensive - if possible, only do it once.
        if not os.path.exists(self.save_fname):
            print(f"saving zernike volumn to {self.save_fname}")
            self.zernike_volume = self.compute_zernike_volume(resolution=self.patch_size, 
                                                              n_terms=self.num_terms, 
                                                              scale_factor=self.scale_factor).astype(np.float32)
            np.save(self.save_fname, self.zernike_volume)
        else:
            self.zernike_volume = np.load(self.save_fname)

        self.zernike_volume = torch.from_numpy(self.zernike_volume)
        # Check if initialization already exists
        self.init_file_path = os.path.join(self.save_path, 'bk_initialization.pt')
        if not os.path.isfile(self.init_file_path):
            # self.zernike_weights = torch.nn.Parameter(torch.rand(num_terms,1,1),requires_grad=True)
            self.zernike_weights = torch.nn.Parameter(torch.ones(num_terms,1,1),requires_grad=True)
            self.get_initialization()
        else:
            # Dummy init, will be overwritten with loaded weights
            self.zernike_weights = torch.nn.Parameter(torch.ones(num_terms,1,1),requires_grad=True)
            

    def compute_zernike_volume(self, resolution, n_terms, scale_factor):
        zernike_volume = poppy.zernike.zernike_basis(nterms=n_terms, npix=resolution, outside=0.0)
        return zernike_volume * scale_factor


    def psf2otf(self, psf, shape):
        """ Adapted from pypher"""
        if torch.all(psf == 0):
            return torch.zeros_like(psf)

        inshape = psf.shape
        # # Pad the PSF to outsize
        # psf = self.zero_pad(psf, shape, position='corner') # unused, assuming filter always same shape as image

        # Circularly shift OTF so that the 'center' of the PSF is
        # [0,0] element of the array
        for axis, axis_size in enumerate(inshape):
            psf = torch.roll(psf, -int(axis_size / 2), dims=axis)

        # Compute the OTF
        otf = torch.fft.fft2(psf)

        return otf

    def apply_zernike(self, image, device='cpu'):
        convolved = torch.zeros_like(image)
        gen_convolved = (image + 1) /2 # Shift back to 0,1 range

        kernel = self.forward().unsqueeze(0).to(device)
        kernel /= kernel.sum() # Kernel should sum to 1

        otf = self.psf2otf(kernel, (256, 256)).squeeze()

        for channel in [0, 1, 2]:
            fourier = torch.fft.fft2(gen_convolved[:, channel, :, :].squeeze(0)) * otf
            convolved[:, channel, :, :] = torch.abs(torch.fft.ifft2(fourier))

        convolved = torch.clip(convolved, 0, 1)
        return convolved

    def tensor2im(self, image_tensor, imtype=np.uint8, rescale=True):
            image_numpy = image_tensor[0].cpu().float().numpy()
            image_numpy = np.transpose(image_numpy, (1, 2, 0))
            if rescale: 
                image_numpy = (image_numpy + 1) / 2.0 
            image_numpy *= 255.0
            return image_numpy.astype(imtype)

    def get_initialization(self, psnr_threshold=32, max_iters=10000):
        if self.verbose: print(f"[INFO] Training blur initialization and saving to {self.init_file_path}")
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        batchsize = 8
        preprocess = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(256)]) 
        dataset = DeDOSDataset('/home/edwin/research/data/deblurGAN', preprocess=preprocess)

        datasets = train_val_test_dataset(dataset)
        dataloaders = {x: DataLoader(datasets[x], batchsize, shuffle=True, num_workers=cpu_count()) for x in
                    ['train', 'val', 'test']}
        dl = dataloaders['train']
        zloss = ZernikeLosses(loss_lambda=1, device=device)

        optimizer = optim.Adam(self.parameters(), lr=0.01)

        progress=tqdm.tqdm(enumerate(dl))
        for idx, (_, sharp) in progress:
            sharp = sharp.to(device) # [b_size, 3, 256, 256]

            self.zero_grad()
            pred_sharp = self.apply_zernike(sharp, device=device)
            loss = zloss((sharp+1)/2, pred_sharp)
            loss.backward()
            optimizer.step()

            # report
            psnr = PSNR(self.tensor2im(pred_sharp.data, rescale=False), self.tensor2im(sharp))
            message = {"loss":loss.item(), "psnr":psnr}
            progress.set_postfix(message)

            if not idx%100:
                plt.figure()
                plt.subplot(121)
                plt.title("Sharp Image")
                plt.imshow(self.tensor2im(sharp))
                plt.axis('off')
                plt.subplot(122)
                plt.title("Identity Blur")
                plt.imshow(self.tensor2im(pred_sharp.data))
                plt.axis('off')
                plt.savefig(f'outputs/init_{idx}.png')
                
        torch.save(self.state_dict(), self.init_file_path)

    def forward(self):
        # x will be of shape (350,)
        out = self.zernike_volume * self.zernike_weights
        return torch.sum(out, 0)
    
if __name__ == "__main__":
    zern_gen = ZernikeGenerator(350)
    import pdb; pdb.set_trace()

