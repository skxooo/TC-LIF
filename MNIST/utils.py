import logging
import torch
import numpy as np
import random
import math
import os
import shutil

from prettytable import PrettyTable

from spiking_neuron import base


# def setup_logging(log_file='log.txt'):
#     """Setup logging configuration"""
#     logging.basicConfig(level=logging.DEBUG,
#                         format="%(asctime)s - %(levelname)s - %(message)s",
#                         datefmt="%Y-%m-%d %H:%M:%S",
#                         filename=log_file,
#                         filemode='w')
#
#     console = logging.StreamHandler()
#     console.setLevel(logging.INFO)
#     formatter = logging.Formatter('%(message)s')
#     console.setFormatter(formatter)
#     logging.getLogger('').addHandler(console)

def setup_logging(log_file='log.txt'):
    # 获取 root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # 清除所有已存在的 handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 文件 Handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)

    # 控制台 Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)

    # 添加 Handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def reset_states(model):
    for m in model.modules():
        if hasattr(m, 'reset'):
            if not isinstance(m, base.MemoryModule):
                print(f'Trying to call `reset()` of {m}, which is not base.MemoryModule')
            m.reset()


def temporal_loop_stack(input, op):
    output_current = []
    for t in range(input.size(0)):
        output_current.append(op(input[t]))
    return torch.stack(output_current, 0)


def seed_everything(seed=1234, is_cuda=False):
    """Some configurations for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if is_cuda:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(seed)


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, dirname, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(dirname, filename))
    if is_best:
        shutil.copyfile(os.path.join(dirname, filename), os.path.join(dirname, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params