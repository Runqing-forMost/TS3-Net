import torch.nn as nn
import torch.nn.functional as F
from datasets.registry import num_classes
from foundations import hparams
from foundations.paths import model
from lottery.desc import LotteryDesc
from models import base
from models.imagenet_resnet import ResNet
from pruning import sparse_global
from torchvision import  models
from models.inceptionv2 import InceptionResNetV2


class Model(base.Model):
    """A inception as for webvision."""

    class Block(nn.Module):
        """A ResNet block."""

        def __init__(self, f_in: int, f_out: int, downsample=False):
            super(Model.Block, self).__init__()

            stride = 2 if downsample else 1
            self.conv1 = nn.Conv2d(f_in, f_out, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(f_out)
            self.conv2 = nn.Conv2d(f_out, f_out, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(f_out)

            # No parameters for shortcut connections.
            if downsample or f_in != f_out:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(f_in, f_out, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(f_out)
                )
            else:
                self.shortcut = nn.Sequential()

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            return F.relu(out)

    def __init__(self, plan, initializer, outputs=50):
        super(Model, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        # self.model = InceptionResNetV2(num_classes=50)
        self.model = models.resnet18(num_classes=50)

    def forward(self, x):
        out = self.model(x)
        return out, 0

    @property
    def output_layer_names(self):
        # return ['last_linear.weight', 'last_linear.bias']
        return ['fc.weight', 'fc.bias']


    @staticmethod
    def is_valid_model_name(model_name):
        return True

    @staticmethod
    def get_model_from_name(model_name, initializer, outputs=50):
        """The naming scheme for a ResNet is 'cifar_resnet_N[_W]'.

        The ResNet is structured as an initial convolutional layer followed by three "segments"
        and a linear output layer. Each segment consists of D blocks. Each block is two
        convolutional layers surrounded by a residual connection. Each layer in the first segment
        has W filters, each layer in the second segment has 32W filters, and each layer in the
        third segment has 64W filters.

        The name of a ResNet is 'cifar_resnet_N[_W]', where W is as described above.
        N is the total number of layers in the network: 2 + 6D.
        The default value of W is 16 if it isn't provided.

        For example, ResNet-20 has 20 layers. Exclusing the first convolutional layer and the final
        linear layer, there are 18 convolutional layers in the blocks. That means there are nine
        blocks, meaning there are three blocks per segment. Hence, D = 3.
        The name of the network would be 'cifar_resnet_20' or 'cifar_resnet_20_16'.
        """


        return Model(None, None, outputs)

    @property
    def loss_criterion(self):
        return self.criterion

    @staticmethod
    def default_hparams():
        model_hparams = hparams.ModelHparams(
            model_name='webvision_inceptionv2',
            model_init='kaiming_normal',
            batchnorm_init='uniform',
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='webvision',
            batch_size=128,
        )

        training_hparams = hparams.TrainingHparams(
            optimizer_name='sgd',
            momentum=0.9,
            milestone_steps='60ep,70ep',
            lr=0.05,
            gamma=0.1,
            weight_decay=5e-4,
            training_steps='80ep',
            e1=20,
            e2=60
        )

        pruning_hparams = sparse_global.PruningHparams(
            pruning_strategy='sparse_global',
            pruning_fraction=0.1
        )

        return LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams)