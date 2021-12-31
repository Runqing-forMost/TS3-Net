import torch.nn as nn
import torch.nn.functional as F
from datasets.registry import num_classes
from foundations import hparams
from lottery.desc import LotteryDesc
from models import base
from pruning import sparse_global
from torchvision import  models

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class Model(base.Model):
    """A residual neural network as originally designed for Clothing1M."""

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

    def __init__(self, plan, initializer, outputs=None):
        super(Model, self).__init__()
        # outputs = outputs or 14
        # current_filters = plan[0][0]
        # self.conv = nn.Conv2d(3, current_filters, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn = nn.BatchNorm2d(current_filters)
        # 
        # # The subsequent blocks of the ResNet.
        # blocks = []
        # for segment_index, (filters, num_blocks) in enumerate(plan):
        #     for block_index in range(num_blocks):
        #         downsample = segment_index > 0 and block_index == 0
        #         blocks.append(Model.Block(current_filters, filters, downsample))
        #         current_filters = filters
        # 
        # self.blocks = nn.Sequential(*blocks)

        # Final fc layer. Size = number of filters in last segment.
        self.criterion = nn.CrossEntropyLoss()
        # self.model = models.resnet18(pretrained=True)
        # self.model.fc = nn.Linear(512, 14)
        self.model = models.resnet18(num_classes=14)
        # self.apply(initializer)

    def forward(self, x):
        # out = F.relu(self.bn(self.conv(x)))
        # out = self.blocks(out)
        # out = F.avg_pool2d(out, out.size()[3])
        # out = out.view(out.size(0), -1)
        # f = out.clone().detach()
        # out = self.fc(out)
        out = self.model(x)

        return out, 0

    @property
    def output_layer_names(self):
        return ['fc.weight', 'fc.bias']

    @staticmethod
    def is_valid_model_name(model_name):
        return (model_name.startswith('clothing_resnet_') and
                5 > len(model_name.split('_')) > 2 and
                all([x.isdigit() and int(x) > 0 for x in model_name.split('_')[2:]]) and
                (int(model_name.split('_')[2]) - 2) % 6 == 0 and
                int(model_name.split('_')[2]) > 2)

    @staticmethod
    def get_model_from_name(model_name, initializer, outputs=100):
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

        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        name = model_name.split('_')
        W = 16 if len(name) == 3 else int(name[3])
        D = int(name[2])
        if (D - 2) % 3 != 0:
            raise ValueError('Invalid ResNet depth: {}'.format(D))
        D = (D - 2) // 6
        plan = [(W, D), (2 * W, D), (4 * W, D)]

        return Model(plan, initializer, outputs)

    @property
    def loss_criterion(self):
        return self.criterion

    @staticmethod
    def default_hparams():
        model_hparams = hparams.ModelHparams(
            model_name='Clothing_resnet_20',
            model_init='kaiming_normal',
            batchnorm_init='uniform',
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='clothing1m',
            batch_size=128,
        )

        training_hparams = hparams.TrainingHparams(
            optimizer_name='sgd',
            momentum=0.9,
            milestone_steps='60ep,80ep',
            lr=0.05,
            gamma=0.1,
            weight_decay=5e-4,
            training_steps='100ep',
            e1=20,
            e2=60
        )

        pruning_hparams = sparse_global.PruningHparams(
            pruning_strategy='sparse_global',
            pruning_fraction=0.1
        )

        return LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams)