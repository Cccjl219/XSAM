import argparse
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models import VGG11, ResNet50, ResNet18
from models.desnet import DenseNet121
from utils import XSAM
from utils.xsam import disable_running_stats, enable_running_stats
from utils.dataset import CIFAR10, CIFAR100
from utils.logger import AverageMeter, CSVLogger
from utils.metrics import accuracy

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

parser = argparse.ArgumentParser()
parser.add_argument('--exp_tag', type=str, default='xsam')
parser.add_argument('--dataset', type=str, default="cifar100", help='Dataset')
parser.add_argument('--model', default='resnet18')

parser.add_argument('--epochs', type=int, default=200, help='epochs')
parser.add_argument('--bs', type=int, default=125, help='batch size')
parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
parser.add_argument('--mo', type=float, default=0.9, help='momentum')
parser.add_argument('--wd', type=float, default=1e-3, help='weight decay')

parser.add_argument('--rho', type=float, default=0.15, help='the total length of multi-step gradient ascent')
parser.add_argument('--rho_max', type=float, default=0.30, help='the distance from current theta to get the loss for determining dynamic alpha')
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--is_dynamic', default=False, action='store_true')
parser.add_argument('--steps', type=int, default=1)
parser.add_argument('--eps', type=float, default=1e-12)

parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--num_workers', type=int, default=2, help='num_workers')
parser.add_argument('--print_freq', type=int, default=100)
parser.add_argument('--load_ckpt', default=False, action='store_true')

parser.add_argument('--alpha_max', type=float, default=2.00, help='alpha max for dynamic')
parser.add_argument('--alpha_delta', type=float, default=0.1, help='alpha delta for dynamic')

args = parser.parse_args()
args.rho = args.rho / args.steps  # get single step gradient ascent length

if __name__ == '__main__':
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.dataset == 'cifar10':
        args.num_classes = 10
        args.data_dir = f"data/{args.dataset}"
    elif args.dataset == 'cifar100':
        args.num_classes = 100
        args.data_dir = f"data/{args.dataset}"
    elif args.dataset == 'tinyimagenet':
        args.num_classes = 200
        args.data_dir = f"data/{args.dataset}"
    else:
        print(f"BAD COMMAND dtype: {args.dataset}")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.logdir = 'results/'
    os.makedirs(args.logdir, exist_ok=True)
    args.logdir += (
        f"{args.model}_{args.dataset}_"
        f"lr{args.lr}_wd{args.wd}_k{args.steps}_rho{args.rho:.2f}_"
        f"{f'dynamic{args.rho_max:.2f}_{args.alpha_max}' if args.is_dynamic else f'alpha{args.alpha:.2f}'}_"
        f"seed{args.seed}_..."
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", handlers=[logging.FileHandler(args.logdir + "output.log"), logging.StreamHandler()])
    logging.info(args)

    def main():
        if args.dataset == 'cifar10':
            dataset = CIFAR10()
        elif args.dataset == 'cifar100':
            dataset = CIFAR100()
        elif args.dataset == 'tinyimagenet':
            dataset = TINYIMAGENET()

        train_dataloader = torch.utils.data.DataLoader(dataset.train_set, batch_size=args.bs, shuffle=True, pin_memory=True, pin_memory_device=args.device.type, num_workers=args.num_workers, persistent_workers=args.num_workers != 0)
        test_dataloader = torch.utils.data.DataLoader(dataset.test_set, batch_size=args.bs, shuffle=False, pin_memory=True, pin_memory_device=args.device.type, num_workers=args.num_workers, persistent_workers=args.num_workers != 0)

        if args.model == 'vgg11':
            model = VGG11(num_classes=args.num_classes)
        elif args.model == 'resnet18':
            model = ResNet18(num_classes=args.num_classes)
        elif args.model == 'desenet121':
            model = DenseNet121(num_classes=args.num_classes)
        elif args.model == 'resnet50':
            model = ResNet50(num_classes=args.num_classes)
        else:
            print("unknown model")
            quit()

        model = model.to(args.device)
        criterion = nn.CrossEntropyLoss()

        optimizer = XSAM(model.parameters(), optim.SGD, steps=args.steps, rho=args.rho, rho_max=args.rho_max, alpha=args.alpha, alpha_max=args.alpha_max, alpha_delta=args.alpha_delta, eps=args.eps, lr=args.lr, momentum=args.mo, weight_decay=args.wd)
        base_optimizer = optimizer.base_optimizer

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(base_optimizer, T_max=args.epochs)
        csv_logger = CSVLogger(args, ['Epoch', 'Lr', 'Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy'], args.logdir + 'training_indicators.csv')

        if args.load_ckpt:
            state = torch.load(f"{args.logdir}_best.pth.tar")
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            best_acc = state['best_acc']
            start_epoch = state['epoch'] + 1
        else:
            start_epoch = 1
            best_acc = 0

        last_5_epoch_acc = []
        for epoch in range(start_epoch, args.epochs + 1):
            logging.info('Epoch: [%d | %d]' % (epoch, args.epochs))

            train_loss, train_acc = run_one_epoch('train', train_dataloader, model, criterion, optimizer)
            logging.info('Train_Loss = {0}, Train_acc = {1}'.format(train_loss, train_acc))

            val_loss, val_acc = run_one_epoch('val', test_dataloader, model, criterion, optimizer)
            logging.info('Val_Loss = {0}, Val_acc = {1}'.format(val_loss, val_acc))

            lr = scheduler.optimizer.param_groups[0]['lr']

            csv_logger.save_values(epoch, lr, train_loss, train_acc, val_loss, val_acc)

            scheduler.step()

            if val_acc > best_acc:
                state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'best_acc': best_acc
                }
                torch.save(state, f"{args.logdir}_best.pth.tar")
                best_acc = val_acc

            if epoch == args.epochs:
                state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'best_acc': best_acc
                }
                torch.save(state, f"{args.logdir}_epoch{args.epochs}.pth.tar")

            last_5_epoch_acc = (last_5_epoch_acc + [val_acc])[-5:]
            logging.info(f'best acc:{best_acc}, mean acc of last 5 epochs:{np.mean(last_5_epoch_acc)}')

    def run_one_epoch(phase, loader, model, criterion, optimizer):
        loss, acc = AverageMeter(), AverageMeter()
        start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(loader, 1):
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            if phase == 'train':
                model.train()
                with torch.set_grad_enabled(True):
                    enable_running_stats(model)
                    optimizer.backup_param()
                    for step in range(args.steps):
                        outputs = model(inputs)
                        batch_loss = criterion(outputs, targets)
                        optimizer.zero_grad()
                        batch_loss.backward()
                        optimizer.first_step(step=step)

                        if step == 0:
                            batch_loss_0 = batch_loss
                            outputs_0 = outputs
                            disable_running_stats(model)  # disable for other forward-backward passes

                    outputs = model(inputs)
                    batch_loss = criterion(outputs, targets)
                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.second_step(update_alpha=((args.is_dynamic is True) and (batch_idx == 1)), model=model, inputs=inputs, targets=targets, criterion=criterion)

            elif phase == 'val':
                model.eval()
                with torch.no_grad():
                    outputs_0 = model(inputs)
                    batch_loss_0 = criterion(outputs_0, targets)

            else:
                logging.info('Define correct phase')
                quit()

            loss.update(batch_loss_0.item(), inputs.size(0))
            batch_acc = accuracy(outputs_0, targets, topk=(1,))[0]
            acc.update(float(batch_acc), inputs.size(0))

            if batch_idx % args.print_freq == 0:
                info = f"Phase:{phase} -- Batch_idx:{batch_idx}/{len(loader)}" \
                       f"-- {acc.count / (time.time() - start_time):.2f} samples/sec" \
                       f"-- Loss:{loss.avg:.2f} -- Acc:{acc.avg:.2f}"
                logging.info(info)

        return loss.avg, acc.avg

    main()
