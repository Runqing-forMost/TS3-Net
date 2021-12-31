# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing
import warnings
import numpy as np
import os
from datasets.base import DataLoader
from torch import nn
import datasets.registry
from foundations import hparams
from foundations import paths
from foundations.step import Step
from models.base import Model, DataParallel, DistributedDataParallel
import models.registry
from platforms.platform import get_platform
from losses.alternate_learning import Alternate, Alternate_clothing
from training.checkpointing import restore_checkpoint
from training import optimizers
from training import standard_callbacks
from training.metric_logger import MetricLogger
from training.optimizers import get_optimizer, get_lr_schedule
import torch.nn.functional as F

try:
    import apex

    NO_APEX = False
except ImportError:
    NO_APEX = True


def alternate_train(
        model_full: Model,
        model_sub: Model,
        output_location: str,
        dataset_hparams: hparams.DatasetHparams,
        training_hparams: hparams.TrainingHparams,
        start_step: Step = None,
        verbose: bool = True,
        evaluate_every_epoch: bool = True
):
    iterations_per_epoch = datasets.registry.iterations_per_epoch(dataset_hparams)
    train_end_step = Step.from_str(training_hparams.training_steps, iterations_per_epoch)
    if (models.registry.exists(output_location, train_end_step) and
            get_platform().exists(paths.logger(output_location))): return
  

    train_loader, _ = datasets.registry.get(dataset_hparams, train=True)
    test_loader = datasets.registry.get(dataset_hparams, train=False)
    callbacks = standard_callbacks.standard_callbacks(
        training_hparams, train_loader, test_loader, start_step=start_step,
        verbose=verbose, evaluate_every_epoch=evaluate_every_epoch)
    print('training...')
    # Create the output location if it doesn't already exist.
    if not get_platform().exists(output_location) and get_platform().is_primary_process:
        get_platform().makedirs(output_location)

    # Get the optimizer and learning rate schedule.
    model_sub.to(get_platform().torch_device)
    model_full.to(get_platform().torch_device)
    optimizer_sub = optimizers.get_optimizer(training_hparams, model_sub)
    optimizer_full = optimizers.get_optimizer(training_hparams, model_full)
    step_optimizer_sub = optimizer_sub
    step_optimizer_full = optimizer_full
    lr_schedule_sub = optimizers.get_lr_schedule(training_hparams, optimizer_sub, train_loader.iterations_per_epoch)
    lr_schedule_full = optimizers.get_lr_schedule(training_hparams, optimizer_full, train_loader.iterations_per_epoch)
    # Adapt for FP16.
    if training_hparams.apex_fp16:
        if NO_APEX: raise ImportError('Must install nvidia apex to use this model.')
        model_sub, step_optimizer_sub = apex.amp.initialize(model_sub, optimizer_sub, loss_scale='dynamic', verbosity=0)
        model_full, step_optimizer_full = apex.amp.initialize(model_full, optimizer_full, loss_scale='dynamic',
                                                              verbosity=0)

    # Handle parallelism if applicable.
    if get_platform().is_distributed:
        model_sub = DistributedDataParallel(model_sub, device_ids=[get_platform().rank])
        model_full = DistributedDataParallel(model_full, device_ids=[get_platform().rank])
    elif get_platform().is_parallel:
        model_sub = DataParallel(model_sub)
        model_full = DataParallel(model_full)

    # Get the random seed for the data order.
    data_order_seed = training_hparams.data_order_seed

    # Restore the model from a saved checkpoint if the checkpoint exists.
    cp_step_sub, cp_logger_sub = restore_checkpoint(output_location, model_sub, optimizer_sub,
                                                    train_loader.iterations_per_epoch)

    start_step = cp_step_sub or start_step or Step.zero(train_loader.iterations_per_epoch)

    logger_sub = cp_logger_sub or MetricLogger()
    
    with warnings.catch_warnings():  # Filter unnecessary warning.
        warnings.filterwarnings("ignore", category=UserWarning)
        for _ in range(start_step.iteration):
            lr_schedule_sub.step()
            lr_schedule_full.step()

    # Determine when to end training.
    end_step = None
    end_step = end_step or Step.from_str(training_hparams.training_steps, train_loader.iterations_per_epoch)

    if end_step <= start_step: return


    criterion = Alternate(r=dataset_hparams.noise_ratio, e_warm=training_hparams.e1, e_co_teaching=training_hparams.e2, e_relabel=80,
                          relabel_threshold=training_hparams.tau, lambda_entropy=training_hparams.lam) 

    # The training loop.
    sig = 0
    for ep in range(start_step.ep, end_step.ep + 1):

        # Ensure the data order is different for each epoch.
        train_loader.shuffle(None if data_order_seed is None else (data_order_seed + ep))

        for it, (examples, labels, ind) in enumerate(train_loader):

            # Advance the data loader until the start epoch and iteration.
            if ep == start_step.ep and it < start_step.it: continue

            # Run the callbacks.
            step = Step.from_epoch(ep, it, train_loader.iterations_per_epoch)
            for callback in callbacks: callback(output_location, step, model_sub, optimizer_sub, logger_sub)

            # Exit at the end step.
            if ep == end_step.ep and it == end_step.it: return

            # Otherwise, train.
            examples = examples.to(device=get_platform().torch_device)
            labels = labels.to(device=get_platform().torch_device)

            step_optimizer_sub.zero_grad()
            step_optimizer_full.zero_grad()
            model_full.train()
            model_sub.train()

            if training_hparams.e1 < ep < training_hparams.e2:
                if ep % 2 == 0:
                    model_sub.train()
                    model_full.eval()
                else:
                    model_sub.eval()
                    model_full.train()

            logit1, _ = model_sub(examples)
            logit2, _ = model_full(examples)


            criterion.r = dataset_hparams.noise_ratio if dataset_hparams.noise_type.startswith(
                'sy') or dataset_hparams.noise_type.startswith('asym_cifar100') or dataset_hparams.dataset_name == 'animal'\
                else dataset_hparams.noise_ratio / 2
           
            loss = criterion(logit1, logit2, labels, ind, ep, sig)

            if training_hparams.apex_fp16:
                with apex.amp.scale_loss(loss, optimizer_sub) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Step forward. Ignore extraneous warnings that the lr_schedule generates.
          
            if training_hparams.e1 < ep < training_hparams.e2:
                if ep % 2 == 0:
                    step_optimizer_sub.step()
                else:
                    step_optimizer_full.step()
            else:
                step_optimizer_sub.step()
                step_optimizer_full.step()
            with warnings.catch_warnings():  # Filter unnecessary warning.
                warnings.filterwarnings("ignore", category=UserWarning)
                lr_schedule_sub.step()
                lr_schedule_full.step()

    get_platform().barrier()




def alternate_train_clothing1M(
        model_full: Model,
        model_sub: Model,
        output_location: str,
        dataset_hparams: hparams.DatasetHparams,
        training_hparams: hparams.TrainingHparams,
        start_step: Step = None,
        verbose: bool = True,
        evaluate_every_epoch: bool = True
):
    iterations_per_epoch = datasets.registry.iterations_per_epoch(dataset_hparams)
    train_end_step = Step.from_str(training_hparams.training_steps, iterations_per_epoch)
    if (models.registry.exists(output_location, train_end_step) and
            get_platform().exists(paths.logger(output_location))): return
    import time
    start = time.time()
    train_loader = datasets.registry.get(dataset_hparams, train=True)
    test_loader = datasets.registry.get(dataset_hparams, train=False)
    end = time.time()
    print('loading data costs %.3f s' % (end - start))
    callbacks = standard_callbacks.standard_callbacks(
        training_hparams, train_loader, test_loader, start_step=start_step,
        verbose=verbose, evaluate_every_epoch=evaluate_every_epoch)
    print('training...')
    # Create the output location if it doesn't already exist.
    if not get_platform().exists(output_location) and get_platform().is_primary_process:
        get_platform().makedirs(output_location)

    # Get the optimizer and learning rate schedule.
    model_sub.to(get_platform().torch_device)
    model_full.to(get_platform().torch_device)
    optimizer_sub = optimizers.get_optimizer(training_hparams, model_sub)
    optimizer_full = optimizers.get_optimizer(training_hparams, model_full)
    step_optimizer_sub = optimizer_sub
    step_optimizer_full = optimizer_full
    lr_schedule_sub = optimizers.get_lr_schedule(training_hparams, optimizer_sub, train_loader.iterations_per_epoch)
    lr_schedule_full = optimizers.get_lr_schedule(training_hparams, optimizer_full, train_loader.iterations_per_epoch)
    # Adapt for FP16.
    if training_hparams.apex_fp16:
        if NO_APEX: raise ImportError('Must install nvidia apex to use this model.')
        model_sub, step_optimizer_sub = apex.amp.initialize(model_sub, optimizer_sub, loss_scale='dynamic', verbosity=0)
        model_full, step_optimizer_full = apex.amp.initialize(model_full, optimizer_full, loss_scale='dynamic',
                                                              verbosity=0)

    # Handle parallelism if applicable.
    if get_platform().is_distributed:
        model_sub = DistributedDataParallel(model_sub, device_ids=[get_platform().rank])
        model_full = DistributedDataParallel(model_full, device_ids=[get_platform().rank])
    elif get_platform().is_parallel:
        model_sub = DataParallel(model_sub)
        model_full = DataParallel(model_full)

    # Get the random seed for the data order.
    data_order_seed = training_hparams.data_order_seed

    # Restore the model from a saved checkpoint if the checkpoint exists.
    cp_step_sub, cp_logger_sub = restore_checkpoint(output_location, model_sub, optimizer_sub,
                                                    train_loader.iterations_per_epoch)

    start_step =  Step.zero(train_loader.iterations_per_epoch)


   
    logger_sub = cp_logger_sub or MetricLogger()
    # logger_full = cp_logger_full or MetricLogger()
    with warnings.catch_warnings():  # Filter unnecessary warning.
        warnings.filterwarnings("ignore", category=UserWarning)
        for _ in range(start_step.iteration):
            lr_schedule_sub.step()
            lr_schedule_full.step()

    # Determine when to end training.
    end_step = None
    end_step = end_step or Step.from_str(training_hparams.training_steps, train_loader.iterations_per_epoch)
    if end_step <= start_step: return

    criterion = Alternate_clothing(r=dataset_hparams.noise_ratio, e_warm=training_hparams.e1, e_co_teaching=training_hparams.e2,
                          e_relabel=80, relabel_threshold=0.7, lambda_entropy=0.)  # for clothing1M

    # The training loop.
    for ep in range(start_step.ep, end_step.ep + 1):
        


        for it, (examples, labels, ind) in enumerate(train_loader):
          

            # Advance the data loader until the start epoch and iteration.
            if ep == start_step.ep and it < start_step.it: continue

            # Run the callbacks.
            step = Step.from_epoch(ep, it, train_loader.iterations_per_epoch)
            for callback in callbacks: callback(output_location, step, model_sub, optimizer_sub, logger_sub)

            # Exit at the end step.
            if ep == end_step.ep and it == end_step.it: return

            # Otherwise, train.
            examples = examples.to(device=get_platform().torch_device)
            labels = labels.to(device=get_platform().torch_device)

            step_optimizer_sub.zero_grad()
            step_optimizer_full.zero_grad()
            model_full.train()
            model_sub.train()

            if training_hparams.e1 < ep < training_hparams.e2:
                if ep % 2 == 0:
                    model_sub.train()
                    model_full.eval()
                else:
                    model_sub.eval()
                    model_full.train()

            logit1, _ = model_sub(examples)

            logit2, _ = model_full(examples)

            criterion.r = 0.20

            loss = criterion(logit1, logit2, labels, ind, ep)

            if training_hparams.apex_fp16:
                with apex.amp.scale_loss(loss, optimizer_sub) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Step forward. Ignore extraneous warnings that the lr_schedule generates.
            if training_hparams.e1 < ep < training_hparams.e2:
                if ep % 2 == 0:
                    step_optimizer_sub.step()
                else:
                    step_optimizer_full.step()
            else:
                step_optimizer_sub.step()
                step_optimizer_full.step()
            with warnings.catch_warnings():  # Filter unnecessary warning.
                warnings.filterwarnings("ignore", category=UserWarning)
                lr_schedule_sub.step()
                lr_schedule_full.step()

    get_platform().barrier()

