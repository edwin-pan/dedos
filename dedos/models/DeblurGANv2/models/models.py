import numpy as np
import torch.nn as nn
from skimage.metrics import structural_similarity as SSIM
from dedos.models.DeblurGANv2.util.metrics import PSNR
import sys
import os
from dedos.metrics import Metrics
import torch


class DeblurModel(nn.Module):
    def __init__(self):
        super(DeblurModel, self).__init__()
        self.metrics = Metrics(device='cuda')


    def get_input(self, data):
        img = data['a']
        inputs = img
        targets = data['b']
        inputs, targets = inputs.cuda(), targets.cuda()
        return inputs, targets

    def tensor2im(self, image_tensor, imtype=np.uint8):
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        return image_numpy.astype(imtype)

    def get_images_and_metrics(self, inp, output, target) -> (float, float, np.ndarray):
        inp = self.tensor2im(inp)
        fake = self.tensor2im(output.data)
        real = self.tensor2im(target.data)
        psnr = PSNR(fake, real)
        ssim = SSIM(fake, real, multichannel=True)
        lpips = self.metrics(torch.clip(output.data.cuda(),0,1), target.data.cuda())[2].item()
        vis_img = np.hstack((inp, fake, real))
        return psnr, ssim, lpips, vis_img


def get_model(model_config):
    return DeblurModel()
