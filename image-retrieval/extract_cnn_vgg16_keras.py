from __future__ import print_function

# -*- coding: utf-8 -*-
# Author: yongyuan.name

import numpy as np
from numpy import linalg as LA

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import lib.custom_transforms as custom_transforms
from PIL import Image
import os
import argparse
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import models


class VGGNet:
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
        parser.add_argument('--lr', default=0.03, type=float, help='learning rate')
        parser.add_argument('--resume', '-r', default='ckpt.t7', type=str, help='resume from checkpoint')
        parser.add_argument('--test-only', action='store_true', help='test only')
        parser.add_argument('--low-dim', default=128, type=int,
                            metavar='D', help='feature dimension')
        parser.add_argument('--nce-k', default=4096, type=int,
                            metavar='K', help='number of negative samples for NCE')
        parser.add_argument('--nce-t', default=0.1, type=float,
                            metavar='T', help='temperature parameter for softmax')
        parser.add_argument('--nce-m', default=0.5, type=float,
                            metavar='M', help='momentum for non-parametric updates')


        args = parser.parse_args()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        best_acc = 0  # best test accuracy
        start_epoch = 0  # start from epoch 0 or last checkpoint epoch

        # Data
        print('==> Preparing data..')

        self.transform = transforms.Compose([
            #transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        net = models.__dict__['ResNet50'](low_dim=args.low_dim)
        # define leminiscate
        if device == 'cuda':
            net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True

        # Model
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('/home/omnisky/image-retrieval/checkpoint/ckpt_Resnet50_us_GAN_m=4096_2.1.1.t7')
        net.load_state_dict(checkpoint['net'])
        #lemniscate = checkpoint['lemniscate']
        # best_acc = checkpoint['acc']
        # start_epoch = checkpoint['epoch']
        #ndata = 50000
        # define loss function

        self.net = net.to(device)
    def default_loader(self,path):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def extract_feat(self, img_path):
        '''
        img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        '''
        #data = torchvision.datasets.ImageFolder(root=img_path, transform=self.transform)
        #loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=1)
        im = self.default_loader(img_path)
        im = self.transform(im)
        im.unsqueeze_(dim=0)
        feat = self.net(im).data[0].cpu().numpy()
        #norm_feat = feat[0]/LA.norm(feat[0])
        return feat
