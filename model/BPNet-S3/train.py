from __future__ import division
import os.path as osp
import sys
import argparse
import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"  # specify which GPU(s) to be used

from config import config
from dataloader import get_train_loader
from network import FPNet
from datasets import Cityscapes
from utils.init_func import init_weight, group_weight
from engine.lr_policy import PolyLR
from engine.engine import Engine
from seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d, Dice_loss, GeneralizedDiceLoss4Organs
#from seg_opr.sync_bn import DataParallelModel, Reduce, BatchNorm2d
from lovasz_losses import lovasz_softmax

'''
try:
    from apex.parallel import SyncBatchNorm
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex .")
'''
import random
random.seed(config.seed)
torch.manual_seed(config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.seed)

parser = argparse.ArgumentParser()

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    cudnn.benchmark = True
    if engine.distributed:
        torch.cuda.set_device(engine.local_rank)

    # data loader
    train_loader, train_sampler = get_train_loader(engine, Cityscapes)

    # config network and criterion
    criterion = nn.CrossEntropyLoss(reduction='mean',
                                    ignore_index=255)
    ohem_criterion = ProbOhemCrossEntropy2d(ignore_label=255, thresh=0.7,
                                            min_kept=int(
                                                config.batch_size // len(
                                                    engine.devices) * config.image_height * config.image_width // 16),
                                            use_weight=False)
                                            #use_weight=True)
    dice_criterion = Dice_loss(config.num_classes)
    #dice_criterion = lovasz_softmax
    if engine.distributed:
        BatchNorm2d = nn.SyncBatchNorm
        #BatchNorm2d = SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d
        #BatchNorm2d = BatchNorm2d
    model = FPNet(config.num_classes, is_training=True,
                criterion=criterion,
                ohem_criterion=ohem_criterion,
                dice_criterion=dice_criterion, dice_weight=config.dice_weight,
                pretrained_model=config.pretrained_model,
                norm_layer=BatchNorm2d)
    print(model)
    init_weight(model.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_out', nonlinearity='relu')

    # group weight and config optimizer
    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr * engine.world_size

    params_list = []
    params_list = group_weight(params_list, model,
                               BatchNorm2d, base_lr)

    optimizer = torch.optim.SGD(params_list,
                                lr=base_lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = PolyLR(base_lr, config.lr_power, total_iteration)

    if engine.distributed:
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model,
                                            device_ids=[engine.local_rank],
                                            output_device=engine.local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model = DataParallelModel(model, engine.devices)
        device = torch.device('cuda:0')
        model.to(device)


    engine.register_state(dataloader=train_loader, model=model,
                          optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    optimizer.zero_grad()
    model.train()

    #print('test 107')
    for epoch in range(engine.state.epoch, config.nepochs):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)
        #print('test 113')
        dataloader = iter(train_loader)
        for idx in pbar:
            engine.update_iteration(epoch, idx)

            minibatch = dataloader.next()
            imgs = minibatch['data']
            gts = minibatch['label']

            if engine.distributed:
            # for multiple-gpu
                imgs = imgs.cuda(non_blocking=True)
                gts = gts.cuda(non_blocking=True)
            else:
            # for single-gpu training
                imgs = imgs.to(device)
                gts = gts.to(device)

            loss, loss_ce, loss_dice = model(imgs, gts)

            # reduce the whole loss over multi-gpu
            if engine.distributed:
                print('loss.shape.......',loss.shape, type(loss),loss)
                dist.all_reduce(loss, dist.ReduceOp.SUM)
                print('loss.shape******',loss.shape, type(loss),loss)
                #print('loss.shape*******',loss.shape)
                loss = loss / engine.world_size
            else:
                if len(loss.shape)>1:
                    loss = Reduce.apply(*loss) / len(loss)

            optimizer.zero_grad()
            current_idx = epoch * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            for i in range(0, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr

            loss.backward()
            optimizer.step()
            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss=%.2f' % loss.item() \
                        + ' loss_ce=%.2f' % loss_ce.item() \
                        + ' loss_dice=%.2f' % loss_dice.item() \
                        + '\n'

            pbar.set_description(print_str, refresh=False)

        if (epoch > config.nepochs - 20) or (epoch % config.snapshot_iter == 0):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            # subprocess.run(["python","eval.py"])

