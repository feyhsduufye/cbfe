import torch.nn as nn
import torchvision.datasets
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import numpy as np
import random
import torch.backends.cudnn
import os
import math
import torchvision.datasets
import torch.backends.cudnn
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import torch.utils.data as data
import numpy as np
import random
import torch.backends.cudnn

step_ratio = 0.1


def get_tiny_imagenet_data_loader(batch_size, train_dir, val_dir, workers=0):
    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=transforms.Compose([
            transforms.ToTensor()
        ]))
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def draw(epoch, train_loss_list, train_acc_list, val_loss_list, val_acc_list_branch1, val_acc_list_branch2,
         val_acc_list_branch3, val_acc_list_branch4):
    if (epoch + 1) % 2 == 0:
        plt.subplot(221)
        plt.plot(range(epoch + 1), val_acc_list_branch1)
        plt.legend('b1')
        plt.subplot(222)
        plt.plot(range(epoch + 1), val_acc_list_branch2)
        plt.legend('b2')
        plt.subplot(223)
        plt.plot(range(epoch + 1), val_acc_list_branch3)
        plt.legend('b3')
        plt.subplot(224)
        plt.plot(range(epoch + 1), val_acc_list_branch4)
        plt.legend('b4')
        plt.show()


def draw_final(epochs, val_acc_list, color='', label=''):
    plt.plot(range(epochs), val_acc_list, color=color, label=label)


def setup_seed(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


import logging
import time
import os


def log_creater(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_name = 'experiments.log'
    final_log_file = os.path.join(output_dir, log_name)

    # creat a log
    log = logging.getLogger('train_log')
    log.setLevel(logging.DEBUG)

    # FileHandler
    file = logging.FileHandler(final_log_file)
    file.setLevel(logging.DEBUG)

    # StreamHandler
    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s][line: %(lineno)d] ==> %(message)s')

    # setFormatter
    file.setFormatter(formatter)
    stream.setFormatter(formatter)

    # addHandler
    log.addHandler(file)
    log.addHandler(stream)

    log.info('creating {}'.format(final_log_file))
    return log


def adjust_learning_rate(warm_up, learning_rate, optimizer, epoch):
    # step_ratio = 0.1, learning_rate = 0.1
    if warm_up and epoch < 1:
        learning_rate = 0.01
    elif 75 <= epoch < 130:
        learning_rate = learning_rate * step_ratio
    elif 130 <= epoch < 180:
        learning_rate = learning_rate * (step_ratio ** 2)
    elif epoch >= 180:
        learning_rate = learning_rate * (step_ratio ** 3)
    else:
        learning_rate = learning_rate

    logging.info('Epoch [{}] learning rate = {}'.format(epoch, learning_rate))

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


import torch
from functools import reduce



