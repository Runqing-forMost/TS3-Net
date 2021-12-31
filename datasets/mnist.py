# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from PIL import Image
import torchvision
import numpy as np
from datasets import base
from platforms.platform import get_platform
from datasets.utils import *

def exist_label(base_dir, noise_mode, noise_rate):
    fname = os.path.join(base_dir, '%s_%s.npy' % (noise_mode, str(noise_rate)))
    if os.path.isfile(fname):
        return True
    return False


noise_function = {'sym': noisify_multiclass_symmetric,
                  'asym_cifar10': noisify_cifar10_asymmetric,
                  'asym_cifar100': noisify_cifar100_asymmetric,
                  'asym_mnist': noisify_mnist_asymmetric,
                  'pairflip': noisify_pairflip}

class Dataset(base.ImageDataset):
    """The MNIST dataset."""

    @staticmethod
    def num_train_examples(): return 60000

    @staticmethod
    def num_test_examples(): return 10000

    @staticmethod
    def num_classes(): return 10

    @staticmethod
    def get_train_set(use_augmentation, noise_type, noise_ratio, base_dir):
        # No augmentation for MNIST.
        train_set = torchvision.datasets.MNIST(
            train=True, root=os.path.join(get_platform().dataset_root, 'mnist'), download=True)
        train_labels = np.array(train_set.targets)
        train_labels = np.asarray([[train_labels[i]] for i in range(len(train_labels))])
        if exist_label(base_dir, noise_type, noise_ratio):
            print('loading existing labels...')
            train_noisy_labels = np.load(
                os.path.join(base_dir, '{}_{}.npy'.format(noise_type, str(noise_ratio))))

        else:
            noise_fun = noise_function[noise_type]
            train_noisy_labels, actual_noise_rate = noise_fun(train_labels, noise_ratio, nb_classes=10)
            np.save(os.path.join(base_dir, '{}_{}.npy'.format(noise_type, str(noise_ratio))), train_noisy_labels)
        # train_noisy_labels, actual_noise_rate = noisify_multiclass_symmetric(train_labels,  0.2)
        train_noisy_labels = np.squeeze(train_noisy_labels)
        # print(train_noisy_labels.shape, np.array(train_set.targets).shape)
        # return Dataset(train_set.data, np.array(train_set.targets), augment if use_augmentation else [])
        return Dataset(train_set.data, train_noisy_labels)

        # return Dataset(train_set.data, train_set.targets)

    @staticmethod
    def get_test_set():
        test_set = torchvision.datasets.MNIST(
            train=False, root=os.path.join(get_platform().dataset_root, 'mnist'), download=True)
        return Dataset(test_set.data, test_set.targets)

    def __init__(self,  examples, labels):
        tensor_transforms = [torchvision.transforms.Normalize(mean=[0.1307], std=[0.3081])]
        super(Dataset, self).__init__(examples, labels, [], tensor_transforms)

    def example_to_image(self, example):
        return Image.fromarray(example.numpy(), mode='L')




DataLoader = base.DataLoader
