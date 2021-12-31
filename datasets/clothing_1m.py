from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
from datasets import base
from foundations.hparams import DatasetHparams
from PIL import Image
import torch


class clothing_dataset(Dataset):
    @staticmethod
    def num_train_examples():
        return 64000

    @staticmethod
    def num_test_examples():
        return 10000

    @staticmethod
    def num_classes():
        return 14

    def __init__(self, train=True):

        self.root = DatasetHparams.dataset_basedir
        self.train_labels = {}
        self.test_labels = {}
        self.val_labels = {}
        self.train = train
        self.transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
        ])
        with open('%s/noisy_label_kv.txt' % self.root, 'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = '%s/' % self.root + entry[0][7:]
                self.train_labels[img_path] = int(entry[1])
        with open('%s/clean_label_kv.txt' % self.root, 'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = '%s/' % self.root + entry[0][7:]
                self.test_labels[img_path] = int(entry[1])

        if train:
            train_imgs = []
            with open('%s/noisy_train_key_list.txt' % self.root, 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/' % self.root + l[7:]
                    train_imgs.append(img_path)
            random.shuffle(train_imgs)
            class_num = torch.zeros(self.num_classes())
            self.train_imgs = []
            for impath in train_imgs:
                label = self.train_labels[impath]
                if class_num[label] < (64000 / 14) and len(self.train_imgs) < 64000:
                    self.train_imgs.append(impath)
                    class_num[label] += 1
            random.shuffle(self.train_imgs)

        else:
            self.test_imgs = []
            with open('%s/clean_test_key_list.txt' % self.root, 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/' % self.root + l[7:]
                    self.test_imgs.append(img_path)

    def __getitem__(self, index):
        if self.train:
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform_train(image)
            return img, target, index
        else:
            img_path = self.test_imgs[index]
            target = self.test_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform_test(image)
            return img, target, index

    def __len__(self):
        if self.train is False:
            return len(self.test_imgs)
        else:
            return len(self.train_imgs)


DataLoader = base.DataLoader
