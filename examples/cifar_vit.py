import argparse
import datetime
import glob
import json
import random
import shutil
import copy

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from freqcore import FreqCore
from torchvision import transforms
from model import *
import timm
import torch.distributed as dist
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

RANK = int(os.getenv('RANK', -1))
LOCAL_RANK = -1


def safe_print(*args, **kwargs):
    if RANK in (-1, 0):
        print(*args)

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
    parser.add_argument('--weight-decay', type=float, default=0.03, metavar='W')
    parser.add_argument('--optimizer', type=str, default='adamw', help='different optimizers')
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--save_path', type=str, default='./result', help='Folder to save checkpoints and log.')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--manualSeed', type=int, help='manual seed')

    # scheduling arguments
    parser.add_argument('--max-lr', default=1e-4, type=float)
    parser.add_argument('--min-lr', default=1e-6, type=float)
    parser.add_argument('--warmup-epochs', default=5, type=int)
    parser.add_argument('--div-factor', default=25, type=float)
    parser.add_argument('--final-div', default=10000, type=float)
    parser.add_argument('--num_epoch', default=100, type=int, help='training epochs')
    parser.add_argument('--pct-start', default=0.3, type=float)
    parser.add_argument('--shuffle', default=True, action='store_true')
    parser.add_argument('--ratio', default=0.5, type=float, help='prune ratio')
    parser.add_argument('--delta', default=0.875, type=float)
    parser.add_argument('--model', default='vit_ti_16', type=str)
    parser.add_argument('--rand-augment', type=int, default=2,
                        help='RandAugment magnitude (set 0 to disable)')
    parser.add_argument('--color-jitter', type=float, default=0.4,
                        help='ColorJitter strength (set 0 to disable)')
    parser.add_argument('--random-erasing', type=float, default=0.25,
                        help='RandomErasing probability (set 0 to disable)')
    parser.add_argument('--ema-decay', type=float, default=0.999,
                        help='EMA decay (set 0 to disable)')
    parser.add_argument('--pretrained', action='store_true',
                        help='use timm pretrained weights for ViT')
    parser.add_argument('--pretrained-path', type=str, default='',
                        help='path to a pretrained checkpoint (optional)')
    parser.add_argument('--resize-224', action='store_true',
                        help='resize inputs to 224x224 to match pretrained ViT')
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
              f" ema_decay:{args.ema_decay},"
              f" save_path:{args.save_path},"
              f" resume:{args.resume},"
              f" batchsize:{args.batch_size},"
              f" use_freqcore:{args.use_freqcore}\n")
    log.flush()

    model_name = args.model.lower()
    if model_name in ('vit_ti_16', 'vit_tiny_patch16_224'):
        if timm is None:
            raise RuntimeError('timm is required for ViT models. Install with: pip install timm')
        net = timm.create_model(
            'vit_tiny_patch16_224',
            pretrained=args.pretrained,
            num_classes=100,
            img_size=224 if args.resize_224 else 32,
        )
    elif model_name == 'r18':
        net = ResNet18(num_classes=100)
    elif model_name == 'r50':
        net = ResNet50(num_classes=100)
    elif model_name == 'r101':
        net = ResNet101(num_classes=100)
    else:
        raise ValueError(f'Unsupported model: {args.model}')
    net = net.to(device)
    def resize_pos_embed(pos_embed, target_pos_embed):
        if pos_embed.shape == target_pos_embed.shape:
            return pos_embed
        num_extra_tokens = target_pos_embed.shape[1] - int(math.sqrt(target_pos_embed.shape[1] - 1)) ** 2
        extra_tokens = pos_embed[:, :num_extra_tokens]
        pos_tokens = pos_embed[:, num_extra_tokens:]
        tgt_tokens = target_pos_embed[:, num_extra_tokens:]
        gs_old = int(math.sqrt(pos_tokens.shape[1]))
        gs_new = int(math.sqrt(tgt_tokens.shape[1]))
        pos_tokens = pos_tokens.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
        pos_tokens = F.interpolate(pos_tokens, size=(gs_new, gs_new), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
        return torch.cat((extra_tokens, pos_tokens), dim=1)

    if args.pretrained_path:
        if os.path.isfile(args.pretrained_path):
            checkpoint = torch.load(args.pretrained_path, map_location='cpu', weights_only=True)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            if 'head.weight' in state_dict and state_dict['head.weight'].shape != net.head.weight.shape:
                state_dict.pop('head.weight', None)
                state_dict.pop('head.bias', None)
            if 'pos_embed' in state_dict:
                state_dict['pos_embed'] = resize_pos_embed(state_dict['pos_embed'], net.pos_embed)
            missing, unexpected = net.load_state_dict(state_dict, strict=False)
            safe_print(f"Loaded pretrained from {args.pretrained_path}. "
                       f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        else:
            raise FileNotFoundError(f'pretrained-path not found: {args.pretrained_path}')

    net = net.to(device)
    vit_body = net
    ema_model = None
    if args.ema_decay > 0:
        ema_model = copy.deepcopy(vit_body).to(device)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)
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
    train_transforms = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
    ]
    if args.resize_224:
        train_transforms.append(transforms.Resize(224))
    if args.color_jitter > 0:
        train_transforms.append(
            transforms.ColorJitter(
                brightness=args.color_jitter,
                contrast=args.color_jitter,
                saturation=args.color_jitter,
                hue=min(0.2, args.color_jitter)
            )
        )
    if args.rand_augment > 0:
        train_transforms.append(transforms.RandAugment(num_ops=2, magnitude=args.rand_augment))
    train_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    if args.random_erasing > 0:
        train_transforms.append(transforms.RandomErasing(p=args.random_erasing, value='random'))

    train_transform = transforms.Compose(train_transforms)

    test_transforms = []
    if args.resize_224:
        test_transforms.append(transforms.Resize(224))
    test_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose(test_transforms)

    ########## Please change to your own dataset path. ##########
    trainset = torchvision.datasets.CIFAR100(root='/mnt/data1/mzh/cifar_100', train=True, transform=train_transform, download=True)
    testset = torchvision.datasets.CIFAR100(root='/mnt/data1/mzh/cifar_100', train=False, transform=test_transform, download=True)



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
                                     weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.max_lr,
                                      weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'lars':
        from lars import Lars
        optimizer = Lars(net.parameters(), lr=args.lr,
                         momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'lamb':
        from lamb import Lamb
        optimizer = Lamb(net.parameters(), lr=args.lr,
                         momentum=args.momentum, weight_decay=args.weight_decay)

    total_steps = args.num_epoch * len(trainloader)
    warmup_steps = args.warmup_epochs * len(trainloader)
    min_lr_ratio = args.min_lr / args.max_lr if args.max_lr > 0 else 0.0

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


    train_acc = []
    valid_acc = []
    tra_loss = []
    valid_loss = []
    scaler = torch.amp.GradScaler('cuda', enabled=args.fp16)


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
            handle = last_block.register_forward_hook(hook)
            with torch.amp.autocast('cuda', enabled=args.fp16):
                outputs = net(inputs)
                handle.remove() 
                score = similarity(feature_out_hook)
                loss = criterion(outputs, targets)
                loss = trainset.update(loss, score)  
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if ema_model is not None:
                update_ema(ema_model, vit_body, args.ema_decay)
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
            if ema_model is not None:
                update_ema(ema_model, vit_body, args.ema_decay)
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
        model_for_eval = ema_model if ema_model is not None else net
        model_for_eval.eval()
        test_loss = 0
        correct = 0
        total = 0
        global best_acc
        global best_loss
        global best_epoch
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model_for_eval(inputs)
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


    def update_ema(ema_target, online_model, decay):
        with torch.no_grad():
            for ema_param, model_param in zip(ema_target.parameters(), online_model.parameters()):
                ema_param.mul_(decay).add_(model_param, alpha=1.0 - decay)
            for ema_buf, model_buf in zip(ema_target.buffers(), online_model.buffers()):
                ema_buf.copy_(model_buf)


    def similarity(feature_out_hook_tokens):
        # ViT token output: (bs, num_tokens, embed_dim), includes CLS token
        token_tensor = feature_out_hook_tokens[0]
        if token_tensor.dim() != 3:
            raise RuntimeError('Expected token tensor with shape (bs, num_tokens, embed_dim).')

        bs, num_tokens, embed_dim = token_tensor.shape
        if num_tokens <= 1:
            raise RuntimeError('Token count must include patch tokens.')

        patch_tokens = token_tensor[:, 1:, :]
        if hasattr(vit_body, 'patch_embed') and hasattr(vit_body.patch_embed, 'grid_size'):
            grid_h, grid_w = vit_body.patch_embed.grid_size
        else:
            grid_h = int(math.sqrt(num_tokens - 1))
            grid_w = grid_h
            if grid_h * grid_w != (num_tokens - 1):
                raise RuntimeError('Patch token count is not a perfect square.')

        patch_grid = patch_tokens.reshape(bs, grid_h, grid_w, embed_dim).permute(0, 3, 1, 2)

        a = math.ceil(grid_h / 4)
        b = math.ceil(grid_w / 4)
        batch_fft = torch.fft.fft2(patch_grid)
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

        weights = pca.explained_variance_ratio_[:components]
        similarity = np.sqrt(projected_coordinates[:, :components] ** 2 @ weights)

        return similarity

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

    if hasattr(vit_body, 'blocks'):
        last_block = vit_body.blocks[-1]
    else:
        raise RuntimeError('ViT model does not expose blocks for hook.')
    print(last_block)

    total_time = 0


    # Train #
    for epoch in range(args.start_epoch, args.num_epoch):

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