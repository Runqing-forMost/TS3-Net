import os
import torch.nn
import torch
import torch.nn.functional as F
import numpy as np
from platforms.platform import get_platform
import torchvision
import sys

os.environ["CUDA_VISIBLE_DEVICE"] = "0，1"


class CIFAR100(torchvision.datasets.CIFAR100):

    def download(self):
        if get_platform().is_primary_process:
            with get_platform().open(os.devnull, 'w') as fp:
                sys.stdout = fp
                super(CIFAR100, self).download()
                sys.stdout = sys.__stdout__
        get_platform().barrier()


class Alternate(torch.nn.Module):
    def __init__(
            self,
            r=0.2,
            lambda_entropy=0.1,
            weight_relabeled_samples=0.001,
            e_warm=20,
            e_co_teaching=60,
            e_relabel=120,
            relabel_threshold=0.5,
            ):
        super(Alternate, self).__init__()
        self.r = r
        self.lambda_entropy = lambda_entropy
        self.weight_relabeled_samples = weight_relabeled_samples
        self.epoch_warm_up = e_warm
        self.epoch_co_teaching = e_co_teaching
        self.epoch_relabel = e_relabel
        self.relabel_threshold = relabel_threshold
        train_set = CIFAR100(train=True, root=os.path.join(get_platform().dataset_root, 'cifar100'), download=True)
        train_labels = np.array(train_set.targets)
        self.train_labels = np.asarray([[train_labels[i]] for i in range(len(train_labels))])

    def forward(self, p1, p2, labels, ind, ep, sig):
        # label_copy = labels.copy()
        logsoftmax = torch.nn.LogSoftmax(dim=1).cuda()
        softmax = torch.nn.Softmax(dim=1).cuda()
        if ep < self.epoch_warm_up:  # warm-up stage
            loss_sparse = F.cross_entropy(p1, labels)
            loss_original = F.cross_entropy(p2, labels)
            return loss_sparse + loss_original
        elif ep < self.epoch_co_teaching:  # alternate learning stage
            if ep % 2 == 0:  # fix Dense and update Sparse
                num_remember = int((1 - self.r) * len(ind))
                loss_pick_sparse = F.cross_entropy(p1, labels, reduction='none')
                loss_pick_original = F.cross_entropy(p2, labels, reduction='none')
                ind_sorted_original = np.argsort(loss_pick_original.cpu().data)
                ind_pick_by_original = ind_sorted_original[:num_remember]
                ind_sorted_sparse = np.argsort(loss_pick_sparse.cpu().data)
                ind_pick_by_sparse = ind_sorted_sparse[:num_remember]
                ind_noisy2 = ind_sorted_original[num_remember:]
                # + torch.mean(loss_pick_sparse[ind_pick_by_sparse]) \
                loss_sparse = torch.mean(loss_pick_sparse[ind_pick_by_original]) \
                              - self.lambda_entropy * torch.mean(
                    torch.mul(softmax(p1[ind_noisy2]), logsoftmax(p1[ind_noisy2]))) + torch.mean(
                    loss_pick_sparse[ind_pick_by_sparse])
                return loss_sparse
            else:  # fix Sparse and update Dense
                num_remember = int((1 - self.r) * len(ind))
                loss_pick_sparse = F.cross_entropy(p1, labels, reduction='none')
                loss_pick_original = F.cross_entropy(p2, labels, reduction='none')
                ind_sorted_sparse = np.argsort(loss_pick_sparse.cpu().data)
                ind_pick_by_sparse = ind_sorted_sparse[:num_remember]
                ind_sorted_original = np.argsort(loss_pick_original.cpu().data)
                ind_pick_by_original = ind_sorted_original[:num_remember]
                ind_noisy = ind_sorted_sparse[num_remember:]
                # + torch.mean(loss_pick_original[ind_pick_by_original])
                loss_original = torch.mean(loss_pick_original[ind_pick_by_sparse]) - self.lambda_entropy * torch.mean(
                    torch.mul(softmax(p2[ind_noisy]), logsoftmax(p2[ind_noisy]))) + torch.mean(
                    loss_pick_original[ind_pick_by_original])
                return loss_original
        else:  # relabel stage
            p1 = softmax(p1)
            num_remember = int((1 - self.r) * len(ind))
            loss_pick_sparse = F.cross_entropy(p1, labels, reduction='none')
            loss_pick_original = F.cross_entropy(p2, labels, reduction='none')
            ind_sorted_sparse = np.argsort(loss_pick_sparse.cpu().data)
            ind_pick_by_sparse = ind_sorted_sparse[:num_remember]
            loss_sparse = torch.mean(loss_pick_sparse[ind_pick_by_sparse])
            ind_noisy = ind_sorted_sparse[num_remember:]
            ind_sorted_original = np.argsort(loss_pick_original.cpu().data)

            num_total = 0.
            for idx in ind_noisy:
                pred_label, pse_label = torch.max(p1[int(idx)].unsqueeze(0), dim=1)
                if pred_label > self.relabel_threshold:
                    num_total += 1
                    loss_sparse += self.weight_relabeled_samples * F.cross_entropy(p1[int(idx)].unsqueeze(0),
                                                                                   torch.tensor(
                                                                                       [int(pse_label)]).cuda())
            return loss_sparse


class Alternate_clothing(torch.nn.Module):
    def __init__(
            self,
            r=0.2,
            lambda_entropy=0.1,
            weight_relabeled_samples=0.001,
            e_warm=20,
            e_co_teaching=60,
            e_relabel=120,
            relabel_threshold=0.5,
    ):
        super(Alternate_clothing, self).__init__()
        self.r = r
        self.lambda_entropy = lambda_entropy
        self.weight_relabeled_samples = weight_relabeled_samples
        self.epoch_warm_up = e_warm
        self.epoch_co_teaching = e_co_teaching
        self.epoch_relabel = e_relabel
        self.relabel_threshold = relabel_threshold

    def forward(self, p1, p2, labels, ind, ep):
        logsoftmax = torch.nn.LogSoftmax(dim=1).cuda()
        softmax = torch.nn.Softmax(dim=1).cuda()
        if ep <= self.epoch_warm_up:  # warm-up stage
            loss_sparse = F.cross_entropy(p1, labels)
            loss_original = F.cross_entropy(p2, labels)
            return loss_sparse + loss_original
        elif ep < self.epoch_co_teaching:  # alternate learning stage
            if ep % 2 == 0:  # fix Dense and update Sparse
                num_remember = int((1 - self.r) * len(ind))
                loss_pick_sparse = F.cross_entropy(p1, labels, reduction='none')
                loss_pick_original = F.cross_entropy(p2, labels, reduction='none')
                ind_sorted_original = np.argsort(loss_pick_original.cpu().data)
                ind_pick_by_original = ind_sorted_original[:num_remember]
                ind_sorted_sparse = np.argsort(loss_pick_sparse.cpu().data)
                ind_pick_by_sparse = ind_sorted_sparse[:num_remember]
                ind_noisy2 = ind_sorted_original[num_remember:]
                loss_sparse = torch.mean(loss_pick_sparse[ind_pick_by_original]) \
                              - self.lambda_entropy * torch.mean(
                    torch.mul(softmax(p1[ind_noisy2]), logsoftmax(p1[ind_noisy2])))
                return loss_sparse
            else:  # fix Sparse and update Dense
                num_remember = int((1 - self.r) * len(ind))
                loss_pick_sparse = F.cross_entropy(p1, labels, reduction='none')
                loss_pick_original = F.cross_entropy(p2, labels, reduction='none')
                ind_sorted_sparse = np.argsort(loss_pick_sparse.cpu().data)
                ind_pick_by_sparse = ind_sorted_sparse[:num_remember]
                ind_sorted_original = np.argsort(loss_pick_original.cpu().data)
                ind_pick_by_original = ind_sorted_original[:num_remember]
                ind_noisy = ind_sorted_sparse[num_remember:]
                # + torch.mean(loss_pick_original[ind_pick_by_original])
                loss_original = torch.mean(loss_pick_original[ind_pick_by_sparse]) - self.lambda_entropy * torch.mean(
                    torch.mul(softmax(p2[ind_noisy]), logsoftmax(p2[ind_noisy])))
                return loss_original
        else:
            p1 = softmax(p1)
            num_remember = int((1 - self.r) * len(ind))
            loss_pick_sparse = F.cross_entropy(p1, labels, reduction='none')
            ind_sorted_sparse = np.argsort(loss_pick_sparse.cpu().data)
            ind_pick_by_sparse = ind_sorted_sparse[:num_remember]
            ind_noisy = ind_sorted_sparse[num_remember:]
            loss_sparse = torch.mean(loss_pick_sparse[ind_pick_by_sparse])
            pred_noisy = p1[ind_noisy]

            pred_value, pred_cls = torch.max(pred_noisy, dim=1)

            valid = pred_value > self.relabel_threshold
            valid_relabel = pred_cls[valid].detach()
            num_relabel = len(valid_relabel)
            if num_relabel > 0:
                weight_for_relable = 1.0 / (float(num_remember) / num_relabel) * self.weight_relabeled_samples 
                loss_relable = weight_for_relable * F.cross_entropy(pred_noisy[valid], valid_relabel)
            else:
                weight_for_relable = 0
                loss_relable = 0

            loss_tot = loss_sparse + loss_relable

            return loss_tot





class Alternate_webvision(torch.nn.Module):
    def __init__(
        self,
        r=0.2,
        lambda_entropy=0.1,
        weight_relabeled_samples=0.1,
        e_warm=20,
        e_co_teaching=60,
        e_relabel=120,
        relabel_threshold=0.5,
        ):
        super(Alternate_webvision, self).__init__()
        self.r = r
        self.lambda_entropy = lambda_entropy
        self.weight_relabeled_samples = weight_relabeled_samples
        self.epoch_warm_up = e_warm
        self.epoch_co_teaching = e_co_teaching
        self.epoch_relabel = e_relabel
        self.relabel_threshold = relabel_threshold

    def forward(self, p1, p2, labels, ind, ep):
        logsoftmax = torch.nn.LogSoftmax(dim=1).cuda()
        softmax = torch.nn.Softmax(dim=1).cuda()
        if ep <= self.epoch_warm_up:  # warm-up stage
            loss_sparse = F.cross_entropy(p1, labels)
            loss_original = F.cross_entropy(p2, labels)
            return loss_sparse + loss_original
            
        elif ep < self.epoch_co_teaching:  # alternate learning stage
            if ep % 2 == 0:  # fix Dense and update Sparse
                num_remember = int((1 - self.r) * len(ind))
                loss_pick_sparse = F.cross_entropy(p1, labels, reduction='none')
                loss_pick_original = F.cross_entropy(p2, labels, reduction='none')
                ind_sorted_original = np.argsort(loss_pick_original.cpu().data)
                ind_pick_by_original = ind_sorted_original[:num_remember]
                ind_sorted_sparse = np.argsort(loss_pick_sparse.cpu().data)
                ind_pick_by_sparse = ind_sorted_sparse[:num_remember]
                ind_noisy2 = ind_sorted_original[num_remember:]
                loss_sparse = torch.mean(loss_pick_sparse[ind_pick_by_original]) \
                              - self.lambda_entropy * torch.mean(
                    torch.mul(softmax(p1[ind_noisy2]), logsoftmax(p1[ind_noisy2])))
                return loss_sparse
            else:  # fix Sparse and update Dense
                num_remember = int((1 - self.r) * len(ind))
                loss_pick_sparse = F.cross_entropy(p1, labels, reduction='none')
                loss_pick_original = F.cross_entropy(p2, labels, reduction='none')
                ind_sorted_sparse = np.argsort(loss_pick_sparse.cpu().data)
                ind_pick_by_sparse = ind_sorted_sparse[:num_remember]
                ind_sorted_original = np.argsort(loss_pick_original.cpu().data)
                ind_pick_by_original = ind_sorted_original[:num_remember]
                ind_noisy = ind_sorted_sparse[num_remember:]
                # + torch.mean(loss_pick_original[ind_pick_by_original])
                loss_original = torch.mean(loss_pick_original[ind_pick_by_sparse]) - self.lambda_entropy * torch.mean(
                    torch.mul(softmax(p2[ind_noisy]), logsoftmax(p2[ind_noisy])))
                return loss_original
        else:
            p1 = softmax(p1)
            num_remember = int((1 - self.r) * len(ind))
            loss_pick_sparse = F.cross_entropy(p1, labels, reduction='none')
            ind_sorted_sparse = np.argsort(loss_pick_sparse.cpu().data)
            ind_pick_by_sparse = ind_sorted_sparse[:num_remember]
            ind_noisy = ind_sorted_sparse[num_remember:]
            loss_sparse = torch.mean(loss_pick_sparse[ind_pick_by_sparse])
            pred_noisy = p1[ind_noisy]

            pred_value, pred_cls = torch.max(pred_noisy, dim=1)

            valid = pred_value > self.relabel_threshold
            valid_relabel = pred_cls[valid].detach()
            num_relabel = len(valid_relabel)
            if num_relabel > 0:
                weight_for_relable = 1.0 / (float(num_remember) / num_relabel) * self.weight_relabeled_samples 
                loss_relable = weight_for_relable * F.cross_entropy(pred_noisy[valid], valid_relabel)
            else:
                weight_for_relable = 0
                loss_relable = 0

            loss_tot = loss_sparse + loss_relable

            return loss_tot