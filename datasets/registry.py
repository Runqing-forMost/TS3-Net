# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from torch.utils.data.dataloader import DataLoader
from datasets import base, cifar10, mnist, imagenet, cifar100, clothing_1m, animal,  SVHN, webvision
from foundations.hparams import DatasetHparams
from platforms.platform import get_platform
registered_datasets = {'cifar10': cifar10, 'mnist': mnist, 'imagenet': imagenet, 'cifar100':cifar100, 'animal':animal, 
'svhn':SVHN, 'clothing':clothing_1m, 'webvision':webvision}


def get(dataset_hparams: DatasetHparams, train: bool = True):
    """Get the train or test set corresponding to the hyperparameters."""

    seed = dataset_hparams.transformation_seed or 0
    train_label = None
    # Get the dataset itself.
    if dataset_hparams.dataset_name in registered_datasets:
        use_augmentation = train and not dataset_hparams.do_not_augment
        noise_type = dataset_hparams.noise_type
        noise_ratio = dataset_hparams.noise_ratio
        base_dir = dataset_hparams.dataset_basedir
        if dataset_hparams.dataset_name == 'animal':
            dataset = animal.Animal(train=train)
        elif dataset_hparams.dataset_name == 'clothing':
            dataset = clothing_1m.clothing_dataset(train=train)
        
        elif dataset_hparams.dataset_name == 'webvision':
            dataset = webvision.webvision_dataset(train=train)


        else:
            train_label = None
            if train:
                dataset, train_label = registered_datasets[dataset_hparams.dataset_name].Dataset.get_train_set(use_augmentation,
                                                                                                  noise_type, noise_ratio, base_dir)
            else:
                dataset = registered_datasets[dataset_hparams.dataset_name].Dataset.get_test_set()


    elif dataset_hparams.dataset_name == 'clothing':
        use_augmentation = train and not dataset_hparams.do_not_augment
        if train:
            dataset = clothing_1m.Clothing1M(train)
        else:
            dataset = clothing_1m.Clothing1M(train)
    elif dataset_hparams.dataset_name == 'webvision':
        dataset = webvision.webvision_dataset
    elif dataset_hparams.dataset_name == 'animal':
        dataset = animal.Animal(train=train)
    else:
        raise ValueError('No such dataset: {}'.format(dataset_hparams.dataset_name))

    # Transform the dataset.
    if train and dataset_hparams.random_labels_fraction is not None:
        dataset.randomize_labels(seed=seed, fraction=dataset_hparams.random_labels_fraction)

    if train and dataset_hparams.subsample_fraction is not None:
        dataset.subsample(seed=seed, fraction=dataset_hparams.subsample_fraction)

    if train and dataset_hparams.blur_factor is not None:
        if not isinstance(dataset, base.ImageDataset):
            raise ValueError('Can blur images.')
        else:
            dataset.blur(seed=seed, blur_factor=dataset_hparams.blur_factor)

    if dataset_hparams.unsupervised_labels is not None:
        if dataset_hparams.unsupervised_labels != 'rotation':
            raise ValueError('Unknown unsupervised labels: {}'.format(dataset_hparams.unsupervised_labels))
        elif not isinstance(dataset, base.ImageDataset):
            raise ValueError('Can only do unsupervised rotation to images.')
        else:
            dataset.unsupervised_rotation(seed=seed)

    # Create the loader.
    # if dataset_hparams.dataset_name == 'animal':
    #     return DataLoader(dataset, batch_size=dataset_hparams.batch_size, num_workers=get_platform().num_workers)

    return (registered_datasets[dataset_hparams.dataset_name].DataLoader(
        dataset, batch_size=dataset_hparams.batch_size, num_workers=get_platform().num_workers), train_label) if train_label is not None else registered_datasets[dataset_hparams.dataset_name].DataLoader(
        dataset, batch_size=dataset_hparams.batch_size, num_workers=get_platform().num_workers)


def iterations_per_epoch(dataset_hparams: DatasetHparams):
    """Get the number of iterations per training epoch."""

    if dataset_hparams.dataset_name in registered_datasets:
        if dataset_hparams.dataset_name == 'animal':
            num_train_examples = registered_datasets[dataset_hparams.dataset_name].Animal.num_train_examples()
        elif dataset_hparams.dataset_name == 'clothing':
            num_train_examples = registered_datasets[dataset_hparams.dataset_name].clothing_dataset.num_train_examples()
        elif dataset_hparams.dataset_name == 'webvision':
            num_train_examples = registered_datasets[dataset_hparams.dataset_name].webvision_dataset.num_train_examples()
        else:
            num_train_examples = registered_datasets[dataset_hparams.dataset_name].Dataset.num_train_examples()
    else:
        raise ValueError('No such dataset: {}'.format(dataset_hparams.dataset_name))

    if dataset_hparams.subsample_fraction is not None:
        num_train_examples *= dataset_hparams.subsample_fraction

    return np.ceil(num_train_examples / dataset_hparams.batch_size).astype(int)


def num_classes(dataset_hparams: DatasetHparams):
    """Get the number of classes."""

    if dataset_hparams.dataset_name in registered_datasets:
        if dataset_hparams.dataset_name == 'animal':
            num_classes = registered_datasets[dataset_hparams.dataset_name].Animal.num_classes
        elif dataset_hparams.dataset_name == 'clothing':
            num_classes = registered_datasets[dataset_hparams.dataset_name].clothing_dataset.num_classes
        elif dataset_hparams.dataset_name == 'webvision':
            num_classes = registered_datasets[dataset_hparams.dataset_name].webvision_dataset.num_classes
        else:
            num_classes = registered_datasets[dataset_hparams.dataset_name].Dataset.num_classes()
    else:
        raise ValueError('No such dataset: {}'.format(dataset_hparams.dataset_name))

    if dataset_hparams.unsupervised_labels is not None:
        if dataset_hparams.unsupervised_labels != 'rotation':
            raise ValueError('Unknown unsupervised labels: {}'.format(dataset_hparams.unsupervised_labels))
        else:
            return 4

    return num_classes
