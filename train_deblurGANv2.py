import logging
import os
from functools import partial
from datetime import date

import cv2
import torch
import torch.optim as optim
import tqdm
import yaml
from joblib import cpu_count
from torch.utils.data import DataLoader
from torchvision import transforms

from dedos.models.DeblurGANv2.adversarial_trainer import GANFactory
from dedos.models.DeblurGANv2.metric_counter import MetricCounter
from dedos.models.DeblurGANv2.models.losses import get_loss
from dedos.models.DeblurGANv2.models.models import get_model
from dedos.models.DeblurGANv2.models.networks import get_nets
from dedos.models.DeblurGANv2.schedulers import LinearDecay, WarmRestart
from fire import Fire

from dedos.dataloader import DeDOSDataset, train_val_test_dataset

cv2.setNumThreads(0)


class Trainer:
    def __init__(self, config, train: DataLoader, val: DataLoader):
        self.config = config
        self.train_dataset = train
        self.val_dataset = val
        self.adv_lambda = config['model']['adv_lambda']
        self.metric_counter = MetricCounter(config['experiment_desc'])
        self.warmup_epochs = config['warmup_num']
        self.identifier = date.today().isoformat()
        os.makedirs(f"/home/clairezhangbin/cs236/{self.identifier}/dedos_vals/dedos_metrics/", exist_ok=True)
        os.makedirs(f"/home/clairezhangbin/cs236/{self.identifier}/dedos_vals/dedos_weights/", exist_ok=True)

    def train(self):
        self._init_params()
        for epoch in range(0, self.config['num_epochs']):
            # if (epoch == self.warmup_epochs) and not (self.warmup_epochs == 0):
                # self.netG.module.unfreeze()
            params_to_train = []
            for name, param in self.netG.named_parameters():
                if "final" in name or "smooth" in name or "noise" in name:
                    params_to_train.append(param)
            self.optimizer_G = self._get_optim(params_to_train)
            self.scheduler_G = self._get_scheduler(self.optimizer_G)


            self._run_epoch(epoch)
            self._validate(epoch)
            self.scheduler_G.step()
            self.scheduler_D.step()

            if self.metric_counter.update_best_model():
                torch.save({
                    'model': self.netG.state_dict()
                }, '/home/clairezhangbin/cs236/{}/dedos_vals/dedos_weights/best_{}.h5'.format(self.identifier,self.config['experiment_desc']))
            torch.save({
                'model': self.netG.state_dict()
            }, '/home/clairezhangbin/cs236/{}/dedos_vals/dedos_weights/last_{}.h5'.format(self.identifier,self.config['experiment_desc']))
            print(self.metric_counter.loss_message())
            logging.debug("Experiment Name: %s, Epoch: %d, Loss: %s" % (
                self.config['experiment_desc'], epoch, self.metric_counter.loss_message()))

    def _run_epoch(self, epoch):
        self.metric_counter.clear()
        for param_group in self.optimizer_G.param_groups:
            lr = param_group['lr']

        epoch_size = self.config.get('train_batches_per_epoch') or len(self.train_dataset)
        tq = tqdm.tqdm(self.train_dataset, total=epoch_size)
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        i = 0
        for data in tq:
            inputs, targets = next(iter(self.train_dataset))
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = self.netG(inputs)
            loss_D = self._update_d(outputs, targets)
            self.netG.zero_grad()
            loss_content = self.criterionG(outputs, targets)
            loss_adv = self.adv_trainer.loss_g(outputs, targets)
            loss_G = loss_content + self.adv_lambda * loss_adv
            loss_G.backward()
            self.optimizer_G.step()
            self.metric_counter.add_losses(loss_G.item(), loss_content.item(), loss_D)
            curr_psnr, curr_ssim, curr_lpips, img_for_vis = self.model.get_images_and_metrics(inputs, outputs, targets)
            self.metric_counter.add_metrics(curr_psnr, curr_ssim, curr_lpips)
            tq.set_postfix(loss=self.metric_counter.loss_message())
            if i % 50 == 0:
                self.metric_counter.add_image(img_for_vis, tag='train')
            i += 1
            if i > epoch_size:
                break
        tq.close()
        self.metric_counter.write_to_dict(epoch, self.identifier, val=False)
        # self.metric_counter.write_to_tensorboard(epoch)

    def _validate(self, epoch):
        self.metric_counter.clear()
        epoch_size = self.config.get('val_batches_per_epoch') or len(self.val_dataset)
        tq = tqdm.tqdm(self.val_dataset, total=epoch_size)
        tq.set_description('Validation')
        i = 0
        for data in tq:
            inputs, targets = next(iter(self.val_dataset))
            inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                outputs = self.netG(inputs)
                loss_content = self.criterionG(outputs, targets)
                loss_adv = self.adv_trainer.loss_g(outputs, targets)
            loss_G = loss_content + self.adv_lambda * loss_adv
            self.metric_counter.add_losses(loss_G.item(), loss_content.item())
            curr_psnr, curr_ssim, curr_lpips, img_for_vis = self.model.get_images_and_metrics(inputs, outputs, targets)
            self.metric_counter.add_metrics(curr_psnr, curr_ssim, curr_lpips)
            if i % 20 == 0:
                self.metric_counter.add_image(img_for_vis, tag='val')
            i += 1
            if i > epoch_size:
                break
        tq.close()
        self.metric_counter.write_to_dict(epoch, self.identifier, val=True)
        # self.metric_counter.write_to_tensorboard(epoch, validation=True)

    def _update_d(self, outputs, targets):
        if self.config['model']['d_name'] == 'no_gan':
            return 0
        self.optimizer_D.zero_grad()
        loss_D = self.adv_lambda * self.adv_trainer.loss_d(outputs, targets)
        loss_D.backward(retain_graph=True)
        self.optimizer_D.step()
        return loss_D.item()

    def _get_optim(self, params):
        if self.config['optimizer']['name'] == 'adam':
            optimizer = optim.Adam(params, lr=self.config['optimizer']['lr'])
        elif self.config['optimizer']['name'] == 'sgd':
            optimizer = optim.SGD(params, lr=self.config['optimizer']['lr'])
        elif self.config['optimizer']['name'] == 'adadelta':
            optimizer = optim.Adadelta(params, lr=self.config['optimizer']['lr'])
        else:
            raise ValueError("Optimizer [%s] not recognized." % self.config['optimizer']['name'])
        return optimizer

    def _get_scheduler(self, optimizer):
        if self.config['scheduler']['name'] == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             mode='min',
                                                             patience=self.config['scheduler']['patience'],
                                                             factor=self.config['scheduler']['factor'],
                                                             min_lr=self.config['scheduler']['min_lr'])
        elif self.config['optimizer']['name'] == 'sgdr':
            scheduler = WarmRestart(optimizer)
        elif self.config['scheduler']['name'] == 'linear':
            scheduler = LinearDecay(optimizer,
                                    min_lr=self.config['scheduler']['min_lr'],
                                    num_epochs=self.config['num_epochs'],
                                    start_epoch=self.config['scheduler']['start_epoch'])
        else:
            raise ValueError("Scheduler [%s] not recognized." % self.config['scheduler']['name'])
        return scheduler

    @staticmethod
    def _get_adversarial_trainer(d_name, net_d, criterion_d):
        if d_name == 'no_gan':
            return GANFactory.create_model('NoGAN')
        elif d_name == 'patch_gan' or d_name == 'multi_scale':
            return GANFactory.create_model('SingleGAN', net_d, criterion_d)
        elif d_name == 'double_gan':
            return GANFactory.create_model('DoubleGAN', net_d, criterion_d)
        else:
            raise ValueError("Discriminator Network [%s] not recognized." % d_name)

    def _init_params(self):
        self.criterionG, criterionD = get_loss(self.config['model'])
        self.netG, netD = get_nets(self.config['model'])
        # load pretrained weights
        if self.config['model']['pretrained'] == True:
            weight_path = self.config['model']['weight_path']
            self.netG.load_state_dict(torch.load(weight_path)['model'],strict=False);
            for param in self.netG.parameters():
                param.requires_grad = True
                
            
        self.netG.cuda()
        self.adv_trainer = self._get_adversarial_trainer(self.config['model']['d_name'], netD, criterionD)
        self.model = get_model(self.config['model'])
        self.optimizer_G = self._get_optim(filter(lambda p: p.requires_grad, self.netG.parameters()))
        self.optimizer_D = self._get_optim(self.adv_trainer.get_params())
        self.scheduler_G = self._get_scheduler(self.optimizer_G)
        self.scheduler_D = self._get_scheduler(self.optimizer_D)


def main(config_path='./dedos/models/DeblurGANv2/config/config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)


    # train, val, test dataloader
    batchsize = config.pop('batch_size')
    preprocess = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(256)])
    dataset = DeDOSDataset('/home/clairezhangbin/cs236/dedos/deblurGAN', preprocess=preprocess)
    datasets = train_val_test_dataset(dataset)
    dataloaders = {x: DataLoader(datasets[x], batchsize, shuffle=True, num_workers=cpu_count()) for x in
                   ['train', 'val', 'test']}
    # set up trainer
    trainer = Trainer(config, train=dataloaders['train'], val=dataloaders['val'])
    trainer.train()
    print("===== End of Program =====")


if __name__ == '__main__':
    Fire(main)
