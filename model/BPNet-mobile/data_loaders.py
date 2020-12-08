# coding: utf-8


import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np
from config import config

from torchvision.transforms import ToTensor
from torchvision import datasets, transforms
from utils import normalize
from PIL import Image
import random
from img_utils import random_scale,random_mirror,generate_random_crop_pos,random_crop_pad_to_shape


class LaneDataSet(Dataset):
    def __init__(self, dataset, transform=None, img_mean=np.array([0.485, 0.456, 0.406]), img_std=np.array([0.229, 0.224, 0.225])):
        self._gt_img_list = []
        self._gt_label_binary_list = []
        self._gt_label_instance_list = []
        self.transform = transform
        self.img_mean = img_mean
        self.img_std = img_std

        with open(dataset, 'r') as file:
            for _info in file:
                info_tmp = _info.strip().split(' ')

                self._gt_img_list.append(info_tmp[0])
                self._gt_label_binary_list.append(info_tmp[1])
                # self._gt_label_instance_list.append(info_tmp[2])

        assert len(self._gt_img_list) == len(self._gt_label_binary_list)

        self._shuffle()

    def _shuffle(self):
        # randomly shuffle all list identically
        c = list(zip(self._gt_img_list, self._gt_label_binary_list))
        random.shuffle(c)
        self._gt_img_list, self._gt_label_binary_list = zip(*c)

    def _split_instance_gt(self, label_instance_img):
        # number of channels, number of unique pixel values, subtracting no label
        # adapted from here https://github.com/nyoki-mtl/pytorch-discriminative-loss/blob/master/src/dataset.py
        #no_of_instances = 5 # we may change the instance number in this line!!!
        #no_of_instances = np.unique(label_instance_img).shape[0]-1
        ins = np.zeros((config.num_of_instances, label_instance_img.shape[0],label_instance_img.shape[1]))
        for _ch, label in enumerate(np.unique(label_instance_img)[1:]):
            ins[_ch, label_instance_img == label] = 1

        return ins
    def __len__(self):
        return len(self._gt_img_list)

    def __getitem__(self, idx):
        assert len(self._gt_label_binary_list) == len(self._gt_img_list)

        # load all
        img = cv2.imread(self._gt_img_list[idx], cv2.IMREAD_COLOR) #bgr
        img = img[:, :, ::-1] #reverse to rgb

        # label_instance_img = cv2.imread(self._gt_label_instance_list[idx].strip(), cv2.IMREAD_UNCHANGED)

        #label_img = cv2.imread(self._gt_label_binary_list[idx], cv2.IMREAD_UNCHANGED)
        '''label_img = Image.open(self._gt_label_binary_list[idx])
        label_img.getdata()
        label_img = np.array(label_img)'''
        #label_img = cv2.imread(self._gt_label_binary_list[idx], cv2.IMREAD_COLOR)
        label_img = cv2.imread(self._gt_label_binary_list[idx], cv2.IMREAD_GRAYSCALE)
	#print('label_fn: ',self._gt_label_binary_list[idx], 'type:',type(label_img))

        # optional transformations
        if self.transform:
            # img = self.transform(img)
            # label_img = self.transform(label_img)
            img, label_img = self.transform((img, label_img))
            # only for qice's data
            img = img[96:240,:,:]
            label_img = label_img[96:240,:]

        ## more transformations
        img, label_img = random_mirror(img, label_img)
        if config.train_scale_array is not None:
            img, label_img, scale = random_scale(img, label_img, config.train_scale_array)

        crop_size = (config.image_height, config.image_width)
        crop_pos = generate_random_crop_pos(img.shape[:2], crop_size)

        img, _ = random_crop_pad_to_shape(img, crop_pos, crop_size, 0)
        label_img, _ = random_crop_pad_to_shape(label_img, crop_pos, crop_size, 255)

        # reshape for pytorch
        # tensorflow: [height, width, channels]
        # pytorch: [channels, height, width]
        img = img.reshape(img.shape[2], img.shape[0], img.shape[1])

        # the img is still uint 8, values in [0-255]
        img = normalize(img, self.img_mean, self.img_std)

        # we could split the instance label here, each instance in one channel (basically a binary mask for each)
        # return (img, label_binary, label_instance_img)
        return (img, label_img)
