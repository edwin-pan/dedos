from piqa import PSNR, SSIM, LPIPS
import torch.nn as nn
import torch

class Metrics(nn.Module):
    def __init__(self, device=None):
        super(Metrics, self).__init__()
        self.device=device
        
        self.ssim = SSIM().to(device) # HIGHER BETTER (0,1)
        self.psnr = PSNR().to(device) # HIGHER BETTER (0,inf)
        self.lpips = LPIPS(network='alex').to(device) # LOWER BETTER (0,inf)
        return

    @torch.no_grad()
    def forward(self, x, y):
        '''Computes 3 metrics [ssim, psnr, lpips]'''
        return self.ssim(x,y), self.psnr(x,y), self.lpips(x,y)

if __name__ == '__main__':
    print("TBD")