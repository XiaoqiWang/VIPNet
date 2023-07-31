import os
import os.path
import random
from PIL import Image

import csv
import scipy.io
import numpy as np

import torch
import torchvision
import torch.utils.data as data

class DataLoader(object):
    """Dataset class for IQA databases"""

    def __init__(self, config, dataset, path, patch_num, img_indx=None, istrain=True):
        # config.dataset = dataset
        self.train_bs = config.train_bs
        self.eval_bs = config.eval_bs
        self.istrain = istrain
        self.num_workers = config.num_workers


        transforms_rgb = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])

        transforms_ycbcr = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.448, 0.483, 0.491],
                                             std=[0.248, 0.114, 0.106])
        ])

        if dataset == 'live':
            self.data = LIVEFolder(
            root=path, index=img_indx, transforms=TwoTransform(transforms_rgb, transforms_ycbcr), patch_num=patch_num, img_crop=True)
        elif dataset=='csiq':
            self.data = CSIQFolder(
            root=path, index=img_indx, transforms=TwoTransform(transforms_rgb, transforms_ycbcr), patch_num=patch_num, img_crop=True)
        elif dataset == 'tid2013':
            self.data = TID2013Folder(
            root=path, index=img_indx, transforms=TwoTransform(transforms_rgb, transforms_ycbcr), patch_num=patch_num, img_crop=False)
        elif dataset == 'kadid-10k':
            self.data = KADID10KFolder(
            root=path, index=img_indx, transforms=TwoTransform(transforms_rgb, transforms_ycbcr), patch_num=patch_num, img_crop=False)

        elif dataset == 'livec':
            self.data = LIVEChallengeFolder(
            root=path, index=img_indx, transforms=TwoTransform(transforms_rgb, transforms_ycbcr), patch_num=patch_num, img_crop=True)
        elif dataset == 'koniq-10k':
            self.data = Koniq_10kFolder(
            root=path, index=img_indx, transforms=TwoTransform(transforms_rgb, transforms_ycbcr), patch_num=patch_num, img_crop=True)
        else:
            print('Invalid dataset were provided.')

    def get_data(self):
        if self.istrain:
            dataloader = torch.utils.data.DataLoader(self.data, batch_size=self.train_bs, shuffle=True, num_workers=self.num_workers,pin_memory=False)
        else:
            dataloader = torch.utils.data.DataLoader( self.data, batch_size=self.eval_bs,num_workers=self.num_workers, shuffle=False,pin_memory=False)
        return dataloader

class TwoTransform:
    """Create two crops of the same image"""
    def __init__(self, transforms_rgb, transforms_ycbcr):
        self.transforms_rgb = transforms_rgb
        self.transforms_ycbcr = transforms_ycbcr

    def __call__(self, x, y):
        return [self.transforms_rgb(x), self.transforms_ycbcr(y)]

class LIVEFolder(data.Dataset):

    def __init__(self, root, index, transforms, patch_num, img_crop):
        refpath = os.path.join(root, 'refimgs')
        refname = getFileName(refpath, '.bmp')

        jp2kroot = os.path.join(root, 'jp2k')
        jp2kname = self.getDistortionTypeFileName(jp2kroot, 227)

        jpegroot = os.path.join(root, 'jpeg')
        jpegname = self.getDistortionTypeFileName(jpegroot, 233)

        wnroot = os.path.join(root, 'wn')
        wnname = self.getDistortionTypeFileName(wnroot, 174)

        gblurroot = os.path.join(root, 'gblur')
        gblurname = self.getDistortionTypeFileName(gblurroot, 174)

        fastfadingroot = os.path.join(root, 'fastfading')
        fastfadingname = self.getDistortionTypeFileName(fastfadingroot, 174)

        imgpath = jp2kname + jpegname + wnname + gblurname + fastfadingname

        dmos = scipy.io.loadmat(os.path.join(root, 'dmos_realigned.mat'))
        labels = dmos['dmos_new'].astype(np.float32)

        orgs = dmos['orgs']
        refnames_all = scipy.io.loadmat(os.path.join(root, 'refnames_all.mat'))
        refnames_all = refnames_all['refnames_all']
        sample = []

        for i in range(0, len(index)):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = train_sel * ~orgs.astype(np.bool_)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[1].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    sample.append((imgpath[item], labels[0][item]))

        self.img_crop = img_crop
        self.samples = sample
        self.transforms = transforms

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample_rgb = imgread(path, is_rgb=True)
        sample_ycbcr = imgread(path, is_rgb=False)
        sample_rgb, sample_ycbcr = imgprocess(sample_rgb, sample_ycbcr, patch_size=[288, 384], is_crop=self.img_crop)
        sample = self.transforms(sample_rgb, sample_ycbcr)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length

    def getDistortionTypeFileName(self, path, num):
        filename = []
        index = 1
        for i in range(0, num):
            name = '%s%s%s' % ('img', str(index), '.bmp')
            filename.append(os.path.join(path, name))
            index = index + 1
        return filename

class CSIQFolder(data.Dataset):

    def __init__(self, root, index, transforms, patch_num, img_crop):

        refpath = os.path.join(root, 'src_imgs')
        refname = getFileName(refpath, '.png')
        txtpath = os.path.join(root, 'csiq_label.txt')
        fh = open(txtpath, 'r')
        imgnames = []
        target = []
        refnames_all = []
        for line in fh:
            line = line.split('\n')
            words = line[0].split()
            imgnames.append((words[0]))
            target.append(words[1])
            ref_temp = words[0].split(".")
            refnames_all.append(ref_temp[0] + '.' + ref_temp[-1])

        labels = np.array(target).astype(np.float32)
        refnames_all = np.array(refnames_all)
        sample = []

        for i, item in enumerate(index):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    # Create a folder 'all_imgs' and copy all distorted images into it.
                    sample.append((os.path.join(root, 'all_imgs', imgnames[item]), labels[item]))
        self.img_crop = img_crop
        self.samples = sample
        self.transforms = transforms

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample_rgb = imgread(path, is_rgb=True)
        sample_ycbcr = imgread(path, is_rgb=False)
        sample_rgb, sample_ycbcr = imgprocess(sample_rgb, sample_ycbcr, patch_size=[288, 384], is_crop=self.img_crop)
        sample = self.transforms(sample_rgb,sample_ycbcr)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length

class TID2013Folder(data.Dataset):
    def __init__(self, root, index, transforms, patch_num, img_crop):
        refpath = os.path.join(root, 'reference_images')
        refname = sorted(getTIDFileName(refpath, '.bmp.BMP'))
        txtpath = os.path.join(root, 'mos_with_names.txt')
        fh = open(txtpath, 'r')
        imgnames = []
        target = []
        refnames_all = []
        for line in fh:
            line = line.split('\n')
            words = line[0].split()
            imgnames.append((words[1]))
            target.append(words[0])
            ref_temp = words[1].split("_")
            refnames_all.append(ref_temp[0][1:])

        labels = np.array(target).astype(np.float32)
        refnames_all = np.array(refnames_all)
        sample = []
        for i, item in enumerate(index):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    sample.append((os.path.join(root, 'distorted_images', imgnames[item]), labels[item]))
        self.img_crop = img_crop
        self.samples = sample
        self.transforms = transforms

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample_rgb = imgread(path, is_rgb=True)
        sample_ycbcr = imgread(path, is_rgb=False)

        sample_rgb, sample_ycbcr = imgprocess(sample_rgb, sample_ycbcr, patch_size=[288, 384],is_crop=self.img_crop)
        sample = self.transforms(sample_rgb, sample_ycbcr)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length

class KADID10KFolder(data.Dataset):
    def __init__(self, root, index, transforms, patch_num, img_crop):
        imgname = []
        refnames_all = []
        labels = []
        csv_file = os.path.join(root, 'dmos.csv')
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['dist_img'])
                refnames_all.append(row['ref_img'])
                mos = np.array(float(row['dmos']))
                labels.append(mos)
        im_ref = np.unique(refnames_all)
        refnames_all = np.array(refnames_all)
        sample = []
        for i, item in enumerate(index):
            train_sel = (im_ref[index[i]] == refnames_all)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    sample.append((os.path.join(root, 'images', imgname[item]), labels[item]))
        self.img_crop = img_crop
        self.samples = sample
        self.transforms = transforms

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample_rgb = imgread(path, is_rgb=True)
        sample_ycbcr = imgread(path, is_rgb=False)
        sample_rgb, sample_ycbcr = imgprocess(sample_rgb, sample_ycbcr, patch_size=[288, 384], is_crop=self.img_crop)

        sample = self.transforms(sample_rgb, sample_ycbcr)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length

class LIVEChallengeFolder(data.Dataset):

    def __init__(self, root, index, transforms, patch_num, img_crop):
        imgpath = scipy.io.loadmat(os.path.join(root, 'Data', 'AllImages_release.mat'))
        imgpath = imgpath['AllImages_release']
        imgpath = imgpath[7:]
        mos = scipy.io.loadmat(os.path.join(root, 'Data', 'AllMOS_release.mat'))
        labels = mos['AllMOS_release'].astype(np.float32)
        labels = labels[0][7:]

        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join(root, 'Images', imgpath[item][0][0]), labels[item]))
        self.img_crop = img_crop
        self.samples = sample
        self.transforms = transforms

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample_rgb = imgread(path, is_rgb=True)
        sample_ycbcr = imgread(path, is_rgb=False)

        sample_rgb, sample_ycbcr = imgprocess(sample_rgb, sample_ycbcr, patch_size=[288, 384], is_crop=self.img_crop)
        sample = self.transforms(sample_rgb, sample_ycbcr)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length

class Koniq_10kFolder(data.Dataset):
    def __init__(self, root, index, transforms, patch_num, img_crop):
        imgname = []
        labels = []
        csv_file = os.path.join(root, 'koniq10k_scores_and_distributions.csv')
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['image_name'])
                mos = np.array(float(row['MOS_zscore'])).astype(np.float32)
                labels.append(mos)
        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join(root, '512x384', imgname[item]), labels[item]))

        self.img_crop = img_crop
        self.samples = sample
        self.transforms = transforms

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample_rgb = imgread(path, is_rgb=True)
        sample_ycbcr = imgread(path, is_rgb=False)

        sample_rgb, sample_ycbcr = imgprocess(sample_rgb, sample_ycbcr, patch_size=[288, 384], is_crop=self.img_crop)
        sample = self.transforms(sample_rgb, sample_ycbcr)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length

def getFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            filename.append(i)
    return filename

def getTIDFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if suffix.find(os.path.splitext(i)[1]) != -1:
            filename.append(i[1:3])
    return filename

def imgread(path, is_rgb=True):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB') if is_rgb else img.convert('YCbCr')

def imgprocess(img_rgb, img_ycbcr, patch_size=[288, 384],is_crop=False):

    w, h = img_rgb.size
    w_ = np.random.randint(low=0, high=w - patch_size[1] + 1)
    h_ = np.random.randint(low=0, high=h - patch_size[0] + 1)
    img_rgb = img_rgb.crop((w_, h_, w_ + patch_size[1], h_ + patch_size[0]))

    if is_crop:
        img_ycbcr = img_ycbcr.crop((w_, h_, w_ + patch_size[1], h_ + patch_size[0]))
    else:
        img_ycbcr = img_ycbcr.resize([384, 288])

    if random.random() < 0.5:
        img_rgb = img_rgb.transpose(Image.FLIP_LEFT_RIGHT)
        img_ycbcr = img_ycbcr.transpose(Image.FLIP_LEFT_RIGHT)

    return img_rgb, img_ycbcr


if __name__ == '__main__':
    pass



