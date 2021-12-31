import torch.nn as nn
import torch.nn.functional as F

from foundations import hparams
from lottery.desc import LotteryDesc
from models import base
from pruning import sparse_global


def call_bn(bn, x):
    return bn(x)


class Model(base.Model):
    def __init__(self, plan, initializer, outputs=10):
        super(Model, self).__init__()
        self.dropout_rate = 0.25
        self.momentum = 0.1
        self.c1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.c3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.c4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.c5 = nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=1)
        self.c6 = nn.Conv2d(196, 16, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(256, outputs)
        self.bn1 = nn.BatchNorm2d(64, momentum=self.momentum)
        self.bn2 = nn.BatchNorm2d(64, momentum=self.momentum)
        self.bn3 = nn.BatchNorm2d(128, momentum=self.momentum)
        self.bn4 = nn.BatchNorm2d(128, momentum=self.momentum)
        self.bn5 = nn.BatchNorm2d(196, momentum=self.momentum)
        self.bn6 = nn.BatchNorm2d(16, momentum=self.momentum)
        self.criterion = nn.CrossEntropyLoss()
        self.apply(initializer)

    def forward(self, x, ):
        h = x
        h = self.c1(h)
        h = F.relu(call_bn(self.bn1, h))
        h = self.c2(h)
        h = F.relu(call_bn(self.bn2, h))
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = self.c3(h)
        h = F.relu(call_bn(self.bn3, h))
        h = self.c4(h)
        h = F.relu(call_bn(self.bn4, h))
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = self.c5(h)
        h = F.relu(call_bn(self.bn5, h))
        h = self.c6(h)
        h = F.relu(call_bn(self.bn6, h))
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = h.view(h.size(0), -1)
        logit = self.fc(h)
        return logit, h

    @property
    def loss_criterion(self):
        return self.criterion

    @property
    def output_layer_names(self):
        return ['fc.weight', 'fc.bias']

    @staticmethod
    def is_valid_model_name(model_name):
        return (model_name.startswith('cifar_cnn') and
                len(model_name.split('_')) == 2)

    @staticmethod
    def get_model_from_name(model_name, initializer, outputs=10):
        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        outputs = outputs or 10
        plan = None
        # num = int(model_name.split('_')[1])
        # if num == 11:
        #     plan = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]
        # elif num == 13:
        #     plan = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]
        # elif num == 16:
        #     plan = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        # elif num == 19:
        #     plan = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
        # else:
        #     raise ValueError('Unknown VGG model: {}'.format(model_name))

        return Model(plan, initializer, outputs)

    @staticmethod
    def default_hparams():
        model_hparams = hparams.ModelHparams(
            model_name='cifar_cnn',
            model_init='kaiming_normal',
            batchnorm_init='uniform',
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='cifar10',
            batch_size=256
        )

        training_hparams = hparams.TrainingHparams(
            optimizer_name='sgd',
            momentum=0.9,
            milestone_steps='60ep',
            lr=0.05,
            gamma=0.1,
            weight_decay=1e-4,
            training_steps='60ep',
            e1=20,
            e2=80,
            lam=1,
            tau=0.7  
        )

        pruning_hparams = sparse_global.PruningHparams(
            pruning_strategy='sparse_global',
            pruning_fraction=0.2,
            pruning_layers_to_ignore='fc.weight'
        )
        return LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams)


class Net(nn.Module):
    def __init__(self, outputs=10):
        super(Net, self).__init__()
        self.dropout_rate = 0.25
        self.momentum = 0.1
        self.c1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.c3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.c4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.c5 = nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=1)
        self.c6 = nn.Conv2d(196, 16, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(256, outputs)
        self.bn1 = nn.BatchNorm2d(64, momentum=self.momentum)
        self.bn2 = nn.BatchNorm2d(64, momentum=self.momentum)
        self.bn3 = nn.BatchNorm2d(128, momentum=self.momentum)
        self.bn4 = nn.BatchNorm2d(128, momentum=self.momentum)
        self.bn5 = nn.BatchNorm2d(196, momentum=self.momentum)
        self.bn6 = nn.BatchNorm2d(16, momentum=self.momentum)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, ):
        h = x
        h = self.c1(h)
        h = F.relu(call_bn(self.bn1, h))
        h = self.c2(h)
        h = F.relu(call_bn(self.bn2, h))
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = self.c3(h)
        h = F.relu(call_bn(self.bn3, h))
        h = self.c4(h)
        h = F.relu(call_bn(self.bn4, h))
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = self.c5(h)
        h = F.relu(call_bn(self.bn5, h))
        h = self.c6(h)
        h = F.relu(call_bn(self.bn6, h))
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = h.view(h.size(0), -1)
        logit = self.fc(h)
        return logit, h


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
