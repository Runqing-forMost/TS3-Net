import numpy as np
import scipy.io as sio
import torch
import os

import torchvision.transforms

from foundations.hparams import DatasetHparams
from PIL import Image
import torchvision.transforms as transforms
from datasets import base
from datasets.utils import *

noise_function = {'sym': noisify_multiclass_symmetric,
                  'asym_cifar10': noisify_cifar10_asymmetric,
                  'asym_cifar100': noisify_cifar100_asymmetric,
                  'asym_mnist': noisify_mnist_asymmetric,
                  'pairflip': noisify_pairflip}


def exist_label(base_dir, noise_mode, noise_rate):
    fname = os.path.join(base_dir, '%s_%s.npy' % (noise_mode, str(noise_rate)))
    if os.path.isfile(fname):
        return True
    return False


def get_train_test_data(root):
    train_dir = os.path.join(root, "train_32x32.mat")
    test_dir = os.path.join(root, "test_32x32.mat")
    train_data = sio.loadmat(train_dir)
    test_data = sio.loadmat(test_dir)
    x_train = torch.from_numpy(train_data['X'])
    y_train = torch.from_numpy(train_data['Y'])
    x_test = torch.from_numpy(test_data['X'])
    y_test = torch.from_numpy(test_data['Y'])

    return x_train, y_train, x_test, y_test


class Dataset(base.ImageDataset):
    """The SVHN dataset."""

    @staticmethod
    def num_train_examples():
        return 73257

    @staticmethod
    def num_test_examples():
        return 26032

    @staticmethod
    def num_classes():
        return 10

    @staticmethod
    def get_train_set(use_augmentation, noise_type, noise_ratio, base_dir):
        train_dir = os.path.join(base_dir, "train_32x32.mat")
        train_data = sio.loadmat(train_dir)
        x_train = train_data['X']
        x_train = np.swapaxes(x_train, 0, 3)
        x_train = np.swapaxes(x_train, 2, 3)
        x_train = np.swapaxes(x_train, 1, 2)
        y_train = train_data['y'].astype(int)
        y_train -= 1
        if exist_label(base_dir, noise_type, noise_ratio):
            print('loading existing labels...')
            train_noisy_labels = np.load(
                os.path.join(base_dir, '{}_{}.npy'.format(noise_type, str(noise_ratio))))

        else:
            noise_fun = noise_function[noise_type]
            train_noisy_labels, actual_noise_rate = noise_fun(y_train, noise_ratio, nb_classes=10)
            np.save(os.path.join(base_dir, '{}_{}.npy'.format(noise_type, str(noise_ratio))), train_noisy_labels)
        train_noisy_labels = np.squeeze(train_noisy_labels.astype(int))

        return Dataset(x_train, train_noisy_labels)

        # return Dataset(train_set.data, train_set.targets)

    @staticmethod
    def get_test_set():
        test_dir = os.path.join(DatasetHparams.dataset_basedir, "test_32x32.mat")
        test_data = sio.loadmat(test_dir)
        x_test = test_data['X']
        x_test = np.swapaxes(x_test, 0, 3)
        x_test = np.swapaxes(x_test, 2, 3)
        x_test = np.swapaxes(x_test, 1, 2)
        test_data['y'] -= 1
        # print(test_data['y'].shape)
        return Dataset(x_test, test_data['y'].astype(int).squeeze())

    def __init__(self, examples, labels, image_transforms=None):
        # tensor_transforms = [torchvision.transforms.ToTensor()]
        super(Dataset, self).__init__(examples, labels, [], image_transforms or [])

    def example_to_image(self, example):
        return Image.fromarray(example)


DataLoader = base.DataLoader
