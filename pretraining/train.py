import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

import time
import os
import argparse
import random

from BoTNet import ResNet50
import data_loader

random.seed(12)
img_sel = list(range(0, 6000000))
random.shuffle(img_sel)

def load_data(args):

    train_loader = data_loader.DataLoader(args.pretrain_path, args.resize_ratio, img_indx=img_sel, batch_size=args.batch_size,num_workers=args.num_workers,istrain=True)
    test_loader = data_loader.DataLoader(args.test_path, args.resize_ratio, batch_size=args.batch_size, istrain=False)
    train_loader = train_loader.get_data()
    test_kadid = test_loader.get_data()
    return train_loader, test_kadid

writer = SummaryWriter('./log')
def save_checkpoint(best_acc, model, optimizer, args, epoch):
    print('Best Model Saving...')
    if args.device_num > 1:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    torch.save({
        'model_state_dict': model_state_dict,
        'global_epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }, os.path.join('checkpoints', 'checkpoint_model_best.pth'))


def _train(epoch, train_loader, model, optimizer, criterion, args, best_acc=0.):
    start = time.time()
    model.train()

    losses = 0.
    acc = 0.
    total = 0.
    for idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        iters_per_epoch = len(train_loader)
        output = model(data)
        _, pred = F.softmax(output, dim=-1).max(1)
        acc += pred.eq(target).sum().item()
        total += target.size(0)

        optimizer.zero_grad()
        loss = criterion(output, target)
        losses += loss
        loss.backward()
        if args.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()

        writer.add_scalar('Training Loss',loss.item(), epoch * iters_per_epoch + idx)
        writer.add_scalar('Training Accuracy', acc / total * 100., epoch * iters_per_epoch + idx)

        if idx % args.print_intervals == 0 and idx != 0:
            print('[Epoch: {0:4d}], Loss: {1:.3f}, Acc: {2:.3f}, Correct {3} / Total {4}'.format(epoch,
                                                                                                 losses / (idx + 1),
                                                                                                 acc / total * 100.,
                                                                                                 acc, total))
        if idx % 10000 == 0 and idx != 0:
            save_checkpoint(best_acc, model, optimizer, args, epoch)
    end = time.time()
    print('Time:', end-start)

def _eval(epoch, test_loader, model, args,test_dataset):
    model.eval()

    acc = 0.
    with torch.no_grad():

        for idx, (data, target) in enumerate(test_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            _, pred = F.softmax(output, dim=-1).max(1)
            acc += pred.eq(target).sum().item()

        print('Testing on: ',test_dataset, 'Epoch: {0:4d}, Acc: {1:.3f}'.format(epoch, acc / len(test_loader.dataset) * 100.))
        writer.add_scalar('Testing Accuracy of {}'.format(test_dataset), acc / len(test_loader.dataset) * 100., epoch)

    return acc / len(test_loader.dataset) * 100.


def main(args):
    train_loader, test_kadid_loader = load_data(args)
    model = ResNet50(num_classes=150, resolution=(384*args.resize_ratio,512*args.resize_ratio), heads=16)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    if args.checkpoints is not None:
        checkpoints = torch.load(os.path.join('checkpoints', args.checkpoints))
        model.load_state_dict(checkpoints['model_state_dict'])
        optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
        start_epoch = checkpoints['global_epoch']
    else:
        start_epoch = 1

    if args.cuda:
        model = model.cuda()

    if not args.evaluation:
        criterion = nn.CrossEntropyLoss()
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=2, eta_min=0.0001)

        global_acc = 0.
        for epoch in range(start_epoch, args.epochs + 1):
            _train(epoch, train_loader, model, optimizer, criterion, args, best_acc=global_acc)
            best_acc = _eval(epoch, test_kadid_loader, model, args, 'KADID-10K')
            if global_acc < best_acc:
                global_acc = best_acc
                save_checkpoint(best_acc, model, optimizer, args, epoch)

            lr_scheduler.step()
            lr_ = lr_scheduler.get_last_lr()[0]
            print('Current Learning Rate: {}'.format(lr_))
            writer.add_scalar('Learning Rate', lr_, epoch)

        writer.close()
    else:
        _eval(start_epoch, test_kadid_loader, model, args,'KADID-10K')

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

dataset_path ={
        'pretrain_path': 'E:\wxq\kadis1000k',
        'test_path':['D:\iqadataset\CSIQ\\all_dis_imgs', 'D:\iqadataset\kadid10k\images']
}

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_path', type=str, default=dataset_path['pretrain_path'])
    parser.add_argument('--test_path', type=str, default=dataset_path['test_path'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--resize_ratio', type=float, default=0.75)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--print_intervals', type=int, default=1000)
    parser.add_argument('--evaluation', type=bool, default=False)
    parser.add_argument('--checkpoints', type=str, default=None, help='model checkpoints path')
    parser.add_argument('--device_num', type=int, default=1)
    parser.add_argument('--gradient_clip', type=float, default=2.)

    return parser.parse_args()

if __name__ == '__main__':
    args = load_config()
    main(args)