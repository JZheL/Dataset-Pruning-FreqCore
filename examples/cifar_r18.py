import argparse
import datetime
import glob
import json
import random
import shutil

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from freqcore import FreqCore
from torchvision import transforms
from model import *
import torch.distributed as dist
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

RANK = int(os.getenv('RANK', -1))
LOCAL_RANK = -1
# LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))

def safe_print(*args, **kwargs):
    if RANK in (-1, 0):
        print(*args)


def setup_ddp():
    world_size = int(os.getenv('WORLD_SIZE', 1))
    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group('nccl', rank=RANK, world_size=world_size)


def destroy_ddp():
    dist.destroy_process_group()


def save_checkpoint(epoch, model, optimizer, val_acc, best_epoch,
                    best_acc, best_loss, save_path, trainset):

    checkpoint_pattern = os.path.join(save_path, 'checkpoint_epoch_*.pth')
    for ckpt_file in glob.glob(checkpoint_pattern):
        os.remove(ckpt_file)

    checkpoint_path = os.path.join(save_path, f'checkpoint_epoch_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'best_acc': best_acc,
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'scores': trainset.scores,
        'num_pruned': trainset.get_pruned_count()
    }, checkpoint_path)
    print(f" Saved checkpoint at epoch {epoch} to {checkpoint_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
    parser.add_argument('--dataset_name', default='cifar100', type=str, help='name of the dataset (cifar10 or cifar100)')
    parser.add_argument('--lr', default=0.2, type=float, help='learning rate')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--use_freqcore', action='store_true',
                        help='whether use freqcore or not.')
    parser.add_argument('--fp16', action='store_true', help='use mix precision training')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='W')
    parser.add_argument('--optimizer', type=str, default='lars', help='different optimizers')
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--save_path', type=str, default='./result', help='Folder to save checkpoints and log.')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--manualSeed', type=int, help='manual seed')

    # onecycle scheduling arguments
    parser.add_argument('--max-lr', default=5.2, type=float)
    parser.add_argument('--div-factor', default=25, type=float)
    parser.add_argument('--final-div', default=10000, type=float)
    parser.add_argument('--num_epoch', default=200, type=int, help='training epochs')
    parser.add_argument('--pct-start', default=0.3, type=float)
    parser.add_argument('--shuffle', default=True, action='store_true')
    parser.add_argument('--ratio', default=0.5, type=float, help='prune ratio')
    parser.add_argument('--delta', default=0.875, type=float)
    parser.add_argument('--model', default='r18', type=str)
    # parser.add_argument('--pca-topk', default=4, type=int, help='top-k principal components for variance stats')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = 'cuda:0'
    safe_print('==> Building model..')

    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.manualSeed)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = open(os.path.join(args.save_path, 'log_seed_{}.txt'.format(args.manualSeed)), 'a')
    log.write(f"Random Seed: {args.manualSeed},"
              f" Model: {args.model},"
              f" ratio: {args.ratio},"
              f" delta: {args.delta},"
              f" optimizer:{args.optimizer},"
              f" max_lr:{args.max_lr},"
              f" save_path:{args.save_path},"
              f" resume:{args.resume},"
              f" batchsize:{args.batch_size},"
              f" use_freqcore:{args.use_freqcore}\n")
    log.flush()


    if args.model.lower() == 'r18':
        if args.dataset_name == 'cifar10':
            net = ResNet18(num_classes=10)
        else:
            net = ResNet18(num_classes=100)
    
    elif args.model.lower() == 'r50':
        if args.dataset_name == 'cifar10':
            net = ResNet50(num_classes=10)
        else:
            net = ResNet50(num_classes=100)
    
    elif args.model.lower() == 'r101':
        if args.dataset_name == 'cifar10':
            net = ResNet101(num_classes=10)
        else:
            net = ResNet101(num_classes=100)
    
    else:
        raise ValueError(f'Unsupported model: {args.model}')
    net = net.to(device)
    try:
        criterion = nn.CrossEntropyLoss(
            label_smoothing=args.label_smoothing, reduction='none').to(device)
    except:
        safe_print('warning! This version has no label smooth.')
        criterion = nn.CrossEntropyLoss(reduction='none').to(device)
    test_criterion = nn.CrossEntropyLoss().to(device)

    best_acc = 0  
    best_loss = 1e3  
    best_epoch = 0


    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    ########## Please change to your own dataset path. ##########
    trainset = torchvision.datasets.CIFAR10(root='/mnt/data1/mzh/cifar_10', train=True, transform=train_transform, download=True)
    testset = torchvision.datasets.CIFAR10(root='/mnt/data1/mzh/cifar_10', train=False, transform=test_transform, download=True)



    if args.use_freqcore:
        safe_print('Use FreqCore.')
        trainset = FreqCore(trainset, args.num_epoch, args.start_epoch, args.ratio, args.delta)
        if args.resume:
            if os.path.isfile(args.resume):
                checkpoint = torch.load(args.resume, map_location=device)
                args.start_epoch = checkpoint['epoch'] + 1
                trainset = FreqCore(trainset, args.num_epoch, args.start_epoch, args.ratio, args.delta)
                trainset.scores = checkpoint['scores'].cpu()
    else:
        safe_print('Use normal full batch.')



    sampler = None
    train_shuffle = True
    if args.use_freqcore:
        sampler = trainset.sampler
        train_shuffle = False
    safe_print(type(sampler))
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=train_shuffle, num_workers=0, sampler=sampler)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)



    if args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay)
    
    elif args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr,
                                     momentum=args.momentum, weight_decay=args.weight_decay)
    
    elif args.optimizer.lower() == 'lars':
        from lars import Lars
        optimizer = Lars(net.parameters(), lr=args.lr,
                         momentum=args.momentum, weight_decay=args.weight_decay)
    
    elif args.optimizer.lower() == 'lamb':
        from lamb import Lamb
        optimizer = Lamb(net.parameters(), lr=args.lr,
                         momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.max_lr, steps_per_epoch=len(trainloader),
                                                       epochs=args.num_epoch, div_factor=args.div_factor,
                                                       final_div_factor=args.final_div, pct_start=args.pct_start)


    train_acc = []
    valid_acc = []
    tra_loss = []
    valid_loss = []
    scaler = torch.amp.GradScaler('cuda', enabled=args.fp16)
    # scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)


    def train_freqcore(epoch):
        safe_print('\nEpoch: %d, iterations %d' % (epoch, len(trainloader)))
        net.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, blobs in enumerate(trainloader):
            # (indices, (inputs, targets)) = blobs
            inputs, targets = blobs
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            global feature_out_hook
            feature_out_hook = []
            handle = last_conv_layer.register_forward_hook(hook)
            with torch.amp.autocast('cuda', enabled=args.fp16):
                outputs = net(inputs)
                handle.remove() 
                score = similarity(feature_out_hook)
                loss = criterion(outputs, targets)
                loss = trainset.update(loss, score)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        safe_print('epoch:', epoch, '  Training Accuracy:', round(100. * correct / total, 3),
                   '  Train loss:', round(train_loss / len(trainloader), 4))
        train_acc.append(correct / total)
        tra_loss.append(train_loss)
        return round(100. * correct / total, 3)


    def train_normal(epoch):
        safe_print('\nEpoch: %d, iterations %d' % (epoch, len(trainloader)))
        net.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=args.fp16):
                outputs = net(inputs)
                loss = torch.mean(criterion(outputs, targets))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        safe_print('epoch:', epoch, '  Training Accuracy:', round(100. * correct / total, 3),
                   '  Train loss:', round(train_loss / len(trainloader), 4))
        train_acc.append(correct / total)
        tra_loss.append(train_loss)
        return round(100. * correct / total, 3)


    def test(epoch):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        global best_acc
        global best_loss
        global best_epoch
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = test_criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        cur_acc = round(100. * correct / total, 3)
        cur_loss = round(test_loss / len(testloader), 4)
        safe_print('epoch: %d' % epoch, '  Test Acc: %.3f' % cur_acc,
                   '  Test loss: %.4f' % cur_loss,
                   ' Best epoch %d, acc %.3f, loss %.4f' % (best_epoch, best_acc, best_loss))
        if cur_acc > best_acc:
            best_acc = cur_acc
            best_epoch = epoch
        if cur_loss < best_loss:
            best_loss = cur_loss
        valid_acc.append(cur_acc)
        valid_loss.append(cur_loss)
        return cur_acc


    def hook(module, inputdata, output):
        feature_out_hook.append(output.detach())


    def similarity(feature_out_hook_conv):
        batch_tensor = feature_out_hook_conv[0]

        bs, channel, w, h = batch_tensor.shape
        a = math.ceil(w / 4)
        b = math.ceil(h / 4)
        batch_fft = torch.fft.fft2(batch_tensor)
        batch_dct = batch_fft.real
        lf = batch_dct[:, :, :a, :b]
        lf_flat = lf.reshape(bs, -1)
        lf_flat_np = lf_flat.detach().cpu().numpy()

        components = min(4, lf_flat_np.shape[0], lf_flat_np.shape[1])
        pca = PCA(n_components=components) 
        pca.fit(lf_flat_np)
        common_info = pca.components_
        mean = pca.mean_

        centered = lf_flat_np - mean
        projected_coordinates = np.dot(centered, common_info.T)

        weights = pca.explained_variance_ratio_[:components]  # 使用解释方差的比例作为权值
        sim_score = np.sqrt(projected_coordinates[:, :components] ** 2 @ weights)  # 标量
        
        return sim_score

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            args.start_epoch = checkpoint['epoch'] + 1
            val_acc = checkpoint['val_acc']
            best_epoch = checkpoint['best_epoch']
            best_acc = checkpoint['best_acc']
            best_loss = checkpoint['best_loss']
            trainset.num_pruned_samples = checkpoint['num_pruned']
            print(f"Resumed from epoch {checkpoint['epoch']}")


    last_conv_layer = None
    for name, module in net.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv_layer = module
    print(last_conv_layer)

    total_time = 0


    # Train #
    for epoch in range(args.start_epoch, args.num_epoch):

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.max_lr,
                                                           steps_per_epoch=len(trainloader),
                                                           epochs=args.num_epoch, div_factor=args.div_factor,
                                                           final_div_factor=args.final_div, pct_start=args.pct_start,
                                                           last_epoch=epoch * len(trainloader) - 1)
        end = time.time()
        cur_acc = train_freqcore(epoch) if args.use_freqcore else train_normal(epoch)
        total_time += time.time() - end
        val_acc = test(epoch)

        if LOCAL_RANK in (-1, 0):
            save_checkpoint(epoch, net, optimizer, val_acc, best_epoch, best_acc, best_loss, args.save_path, trainset)

            if val_acc > best_acc:
                best_acc = val_acc
                best_path = os.path.join(args.save_path, f'best_checkpoint.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, best_path)
                print(f" New best model saved at epoch {epoch + 1} with acc {val_acc:.4f}")

            log.write(f"Epoch {epoch + 1}/{args.num_epoch}, "
                      f"   Train Acc: {cur_acc:.4f}, "
                      f"   Val Acc: {val_acc:.4f},"
                      f"   Best Acc: {best_acc:.4f}\n")
            log.flush()


    if args.use_freqcore:
        safe_print('Total saved sample forwarding: ', trainset.get_pruned_count())
        log.write(f"Total saved sample forwarding: {trainset.get_pruned_count()} ,")
        log.flush()
    safe_print('Total training time: ', total_time)
    log.write(f" Total training time: {total_time} ")
    log.flush()

    log.close()