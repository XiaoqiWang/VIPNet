import numpy as np
import pandas as pd
import torch
import torchvision
import torch.utils.data as data
from PIL import Image
import os
import os.path

class DataLoader(object):
    """Dataset class for IQA databases"""

    def __init__(self, path,resize_ratio, distributed=True, img_indx=None,batch_size=1,num_workers=8, test_dataset = 'kadid',istrain=True):

        self.batch_size = batch_size
        self.istrain = istrain
        self.num_workers = num_workers
        # Train transforms
        if istrain:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.Resize((int(resize_ratio * 384), int(resize_ratio*512))),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.448, 0.483, 0.491],
                                                 std=[0.248, 0.114, 0.106])
            ])

            self.pretraining_data = PreTrainingDataset(
                root=path, index=img_indx, transform=transforms)

            if distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.pretraining_data)
            else:
                self.train_sampler = None
        # Test transforms
        else:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((int(resize_ratio * 384), int(resize_ratio*512))),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.448, 0.483, 0.491],
                                                 std=[0.248, 0.114, 0.106])
            ])

            if test_dataset=='kadis':
                self.test_data = PreTrainingDataset(
                    root=path, index=img_indx, transform=transforms)
            if test_dataset == 'kadid':
                self.test_data = TestKADIDDataset(
                root=path, index=list(range(10125)), transform=transforms)

    def get_train_sampler(self):
        return self.train_sampler

    def get_data(self):
        if self.istrain:
            Dataloader = torch.utils.data.DataLoader(
                self.pretraining_data, batch_size=self.batch_size, shuffle=(self.train_sampler is None),
                num_workers=self.num_workers,pin_memory=True, sampler=self.train_sampler)
        else:
            Dataloader = torch.utils.data.DataLoader(
                self.test_data, batch_size=32,num_workers=self.num_workers, shuffle=False,pin_memory=True)
        return Dataloader


class PreTrainingDataset(data.Dataset):

    def __init__(self, root, index, transform):
        all_img = []
        all_img_dir = []
        all_label = []
        dist_dir = os.listdir(root)
        for dir in dist_dir:
            for img in os.listdir(os.path.join(root, dir)):
                label = (int(img[-9:-7])-1) * 5 + int(img[-6:-4])
                all_img_dir.append(dir)
                all_label.append(int(label - 1))
                all_img.append(img)

        sample = []
        for i, item in enumerate(index):
                sample.append((os.path.join(root, all_img_dir[item], all_img[item]), all_label[item]))
        print('pretraining images number:',len(sample))
        self.samples = sample # 6000,000
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = YCbCr_Loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length

class TestKADIDDataset(data.Dataset):

    def __init__(self, root, index, transform):
        all_img = []
        all_label = []
        dist_imgs = pd.read_csv(os.path.join(root, 'dmos.csv'))
        dist_imgs = dist_imgs['dist_img']
        for img in dist_imgs:
            label = (int(img[-9:-7])-1) * 5 + int(img[-6:-4])
            all_label.append(int(label - 1))
            all_img.append(img)

        sample = []
        for i, item in enumerate(index):
                sample.append((os.path.join(root, 'images',all_img[item]), all_label[item]))

        self.samples = sample # 10125
        self.transform = transform

    def __getitem__(self, index):

        path, target = self.samples[index]
        sample = YCbCr_Loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length

def YCbCr_Loader(path, resize=False):
    with open(path, 'rb') as f:
        img = Image.open(f)
        if resize:
            img = img.resize((512,384))
        return img.convert('YCbCr')

def RGB_Loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


