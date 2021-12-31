from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
from datasets import base
from foundations.hparams import DatasetHparams
from PIL import Image
import torch


class webvision_dataset(Dataset):
    @staticmethod
    def num_train_examples():
        return 65944

    @staticmethod
    def num_test_examples():
        return 2500

    @staticmethod
    def num_classes():
        return 50

    def __init__(self, train=True):

        self.root = DatasetHparams.dataset_basedir
        self.train_labels = {}
        self.test_labels = {}
        self.val_labels = {}
        self.train = train
        self.transform_train = transforms.Compose([
            transforms.Resize(320),
            transforms.RandomResizedCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.transform_imagenet = transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        # with open(self.root + '/info/val_filelist.txt') as f:
        #     lines = f.readlines()
        #     self.val_imgs = []
        #     self.val_labels = {}
        #     for line in lines:
        #         img, target = line.split()
        #         target = int(target)
        #         if target < self.num_classes():
        #             self.val_imgs.append(img)
        #             self.val_labels[img] = target

        self.data = []
        import os
        base = '/media/data/jrq_data/open_lth_datasets/ILSVRC2012_img_val/'
        imgs = os.listdir('/media/data/jrq_data/open_lth_datasets/ILSVRC2012_img_val')
        imgs = sorted(imgs)[:50]
        c = 0
    
        for i in imgs:
            temp = base + i
            sub_img = os.listdir(temp)
            for j in sub_img:
                self.data.append((c, os.path.join(temp, j)))
            c += 1

        self.train_imgs = []
        with open('%s/info/train_filelist_google.txt' % self.root, 'r') as f:
            lines = f.read().splitlines()
            self.train_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target < self.num_classes():
                    self.train_imgs.append(img)
                    self.train_labels[img] = target

        random.shuffle(self.train_imgs)
            # random.shuffle(train_imgs)
            # class_num = torch.zeros(self.num_classes())
            # self.train_imgs = []
            # for impath in train_imgs:
            #     label = self.train_labels[impath]
            #     if class_num[label] < (64000 / 14) and len(self.train_imgs) < 64000:
            #         self.train_imgs.append(impath)
            #         class_num[label] += 1
            # random.shuffle(self.train_imgs)


        # print(len(self.train_imgs), len(self.data))

    def __getitem__(self, index):
        if self.train:
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            image = Image.open(self.root + '/google_resized_256/' + img_path).convert('RGB')
            img = self.transform_train(image)
            return img, target, index
        else:
            # img_path = self.val_imgs[index]
            # target = self.val_labels[img_path]
            # image = Image.open('/media/data/jrq_data/open_lth_datasets/web_vision/val_images_256/'+ img_path).convert('RGB')
            # img = self.transform_imagenet(image)
            # return img, target, index
            target, img_path = self.data[index]
            image = Image.open(img_path).convert('RGB')
            # print(image.size)
            img = self.transform_imagenet(image)
            return img, target, index


    def __len__(self):
        if self.train is False:
            return len(self.data)
        else:
            return len(self.train_imgs)


DataLoader = base.DataLoader
