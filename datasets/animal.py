from foundations.hparams import DatasetHparams
from PIL import Image
import torch
from datasets import base
import torchvision.transforms as transforms
import os


# class_name_list = ["cane", "cavallo", "elefante", "farfalla", "gallina", "gatto", "mucca", "pecora", "ragno",
#                    "scoiattolo"]
#
#
# def split_training_test_data(root):
#     train_file = os.path.join(root, "train.txt")
#     test_file = os.path.join(root, "test.txt")
#     if os.path.isfile(train_file) and os.path.isfile(test_file):  # if exists, load
#         train_img_path = []
#         train_img_target = []
#         test_img_path = []
#         test_img_target = []
#         with open(train_file, 'r') as f:
#             lines = f.read().splitlines()
#             for l in lines:
#                 category = class_name_list[int(l[:2])]
#                 train_img_path.append(root + '\\' + category + l[3:])
#                 train_img_target.append(int(l[:2]))
#
#         with open(test_file, 'r') as f:
#             lines = f.read().splitlines()
#             for l in lines:
#                 category = class_name_list[int(l[:2])]
#                 test_img_path.append(root + '\\' + category + l[3:])
#                 test_img_target.append(int(l[:2]))
#
#         return train_img_path, train_img_target, test_img_path, test_img_target
#
#     else:  # split dataset as 10 : 1 in every category
#         train_list = []
#         test_list = []
#         for i in range(len(class_name_list)):
#             dir_temp = os.path.join(root, class_name_list[i]) + '/'
#             # print(dir_temp)
#             files = None
#             for r, dirs, file in os.walk(dir_temp):
#                 # print(file)
#                 files = file
#             for j in range(len(files)):
#                 if j < int(0.9 * len(files)):
#                     train_list.append('0' + str(i) + ',' + files[i])
#                 else:
#                     test_list.append('0' + str(i) + ',' + files[i])
#             with open(os.path.join(root, 'train.txt'), 'w') as f:
#                 for t in train_list:
#                     f.write(t)
#                     f.write('\n')
#             with open(os.path.join(root, 'test.txt'), 'w') as f:
#                 for t in test_list:
#                     f.write(t)
#                     f.write('\n')

def get_training_test_data(root):

    train_img_dir = []
    train_img_target = []
    test_img_dir = []
    test_img_target = []
    train_base_dir = os.path.join(root, 'training')
    print(train_base_dir)
    test_base_dir = os.path.join(root, 'testing')
    files = None
    for r, dirs, file in os.walk(train_base_dir):
        files = file
    for f in files:
        train_img_dir.append(os.path.join(train_base_dir, f))
        train_img_target.append(int(f.split('_')[0]))
    for r, dirs, file in os.walk(test_base_dir):
        files = file
    for f in files:
        test_img_dir.append(os.path.join(test_base_dir, f))
        test_img_target.append(int(f.split('_')[0]))
    return train_img_dir, train_img_target, test_img_dir, test_img_target


class Animal(torch.utils.data.Dataset):
    @staticmethod
    def num_train_examples():
        return 50000

    @staticmethod
    def num_test_examples():
        return 10000

    @staticmethod
    def num_classes():
        return 10

    def __init__(self, train=True):
        super(Animal, self).__init__()
        self.train = train
        self.root = DatasetHparams.dataset_basedir
        self.train_img_path, self.train_img_target, self.test_img_path, self.test_img_target = get_training_test_data(
            root=self.root)
        self.transform = transforms.ToTensor()

    def __getitem__(self, item):
        if self.train:
            img = Image.open(self.train_img_path[item])
            img = self.transform(img)
            return img, self.train_img_target[item], item
        else:
            img = Image.open(self.test_img_path[item])
            img = self.transform(img)
            return img, self.test_img_target[item], item

    def __len__(self):
        return len(self.train_img_path) if self.train else len(self.test_img_path)


DataLoader = base.DataLoader
