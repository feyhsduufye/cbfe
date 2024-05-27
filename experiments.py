import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch.optim as optim

import torch.cuda

from tqdm import tqdm
import time
from numpy import *
import os
import torch.nn.functional as F
from utils import *

train_root = '/home/zq/data/tinyimagenet/tiny-imagenet-200/train'
val_root = '/home/zq/data/tinyimagenet/tiny-imagenet-200/val'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 200
T = 3
alpha = 0.1
beta = 4e-5
learning_rate = 0.1
save_path = './wide_resnet38_2.pth'
logger = log_creater('./logger/wide_resnet38_2.log')
resume = '../input/wrn-38-2-scale-factor/resnet18.pth'
step_ratio = 0.1


def kd_loss_function(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs / 3.0, dim=1)
    softmax_targets = F.softmax(targets / 3.0, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()


def evaluating(net, test_loader, optimizer, loss_function, epoch, val_loss_list, val_acc_list_branch1,
               val_acc_list_branch2, val_acc_list_branch3, val_acc_list_branch4, correct_branch1, correct_branch2,
               correct_branch3, correct_branch4, correct_ensemble4=0, correct_ensemble3=0, correct_ensemble2=0):
    net.eval()
    test_bar = tqdm(test_loader)
    length_test_loader = len(test_loader)
    total_sample = len(test_loader.dataset)
    total_loss = 0
    correct_branch5 = 0
    correct_ensemble5 = 0

    inference_time = 0
    time_start = time.time()

    with torch.no_grad():
        for step, data in enumerate(test_loader):
            img, label = data
            img, label = img.to(device), label.to(device)

            outputs, _ = net(img)
            middle_output5, middle_output4, middle_output3, middle_output2, middle_output1 = outputs

            loss = loss_function(middle_output4, label)
            total_loss += loss.item()

            predict_branch5 = torch.max(middle_output5, dim=1)[1]
            predict_branch4 = torch.max(middle_output4, dim=1)[1]
            predict_branch3 = torch.max(middle_output3, dim=1)[1]
            predict_branch2 = torch.max(middle_output2, dim=1)[1]
            predict_branch1 = torch.max(middle_output1, dim=1)[1]

            ensemble_out5 = (middle_output1 + middle_output2 + middle_output3 + middle_output4 + middle_output5) / 5
            ensemble_out4 = (middle_output1 + middle_output2 + middle_output3 + middle_output4) / 4
            ensemble_out3 = (middle_output1 + middle_output2 + middle_output3) / 3
            ensemble_out2 = (middle_output1 + middle_output2) / 2

            ensemble5 = torch.max(ensemble_out5, dim=1)[1]
            ensemble4 = torch.max(ensemble_out4, dim=1)[1]
            ensemble3 = torch.max(ensemble_out3, dim=1)[1]
            ensemble2 = torch.max(ensemble_out2, dim=1)[1]

            correct_branch1 += torch.eq(predict_branch1, label).sum().item()
            correct_branch2 += torch.eq(predict_branch2, label).sum().item()
            correct_branch3 += torch.eq(predict_branch3, label).sum().item()
            correct_branch4 += torch.eq(predict_branch4, label).sum().item()
            correct_branch5 += torch.eq(predict_branch5, label).sum().item()

            correct_ensemble5 += torch.eq(ensemble5, label).sum().item()
            correct_ensemble4 += torch.eq(ensemble4, label).sum().item()
            correct_ensemble3 += torch.eq(ensemble3, label).sum().item()
            correct_ensemble2 += torch.eq(ensemble2, label).sum().item()

            time_end = time.time()
            time_for_one_inference = time_end - time_start
            inference_time += time_for_one_inference

        val_loss = total_loss / length_test_loader

        val_acc_branch1 = correct_branch1 / total_sample
        val_acc_branch2 = correct_branch2 / total_sample
        val_acc_branch3 = correct_branch3 / total_sample
        val_acc_branch4 = correct_branch4 / total_sample
        val_acc_branch5 = correct_branch5 / total_sample

        val_acc_ensemble5 = correct_ensemble5 / total_sample
        val_acc_ensemble4 = correct_ensemble4 / total_sample
        val_acc_ensemble3 = correct_ensemble3 / total_sample
        val_acc_ensemble2 = correct_ensemble2 / total_sample

        val_loss_list.append(val_loss)

        val_acc_list_branch1.append(val_acc_branch1)
        val_acc_list_branch2.append(val_acc_branch2)
        val_acc_list_branch3.append(val_acc_branch3)
        val_acc_list_branch4.append(val_acc_branch4)

        logger.info("Epoch: {} val_loss: {:.3f} lr: {:.5f}, val_acc_branch1: {:.4f}, val_acc_branch2: {:.4f}, "
                    "val_acc_branch3: {:.4f}, val_acc_branch4: {:.4f}, val_acc_branch5: {:.4f}, val_acc_ensemble5: {:.4f}, val_acc_ensemble4: {:.4f}, val_acc_ensemble3: {:.4f}, val_acc_ensemble2: {:.4f}".format(
            epoch + 1, val_loss, optimizer.param_groups[0]['lr'], val_acc_branch1, val_acc_branch2, val_acc_branch3,
            val_acc_branch4, val_acc_branch5, val_acc_ensemble5, val_acc_ensemble4, val_acc_ensemble3, val_acc_ensemble2
        ))
        return val_loss_list, val_acc_list_branch1, val_acc_list_branch2, val_acc_list_branch3, val_acc_list_branch4, \
               val_acc_branch1, val_acc_branch2, val_acc_branch3, val_acc_branch4, val_acc_ensemble4, val_acc_ensemble3, val_acc_ensemble2


def self_kd_train(model, train_loader, optimizer, loss_function, epoch, train_loss_list, train_acc_list):
    model.train()
    train_bar = tqdm(train_loader)
    length_train_loader = len(train_bar)
    total_sample = len(train_loader.dataset)
    total_loss = 0.0
    correct_num = 0
    adjust_learning_rate(False, learning_rate, optimizer, epoch)

    for step, data in enumerate(train_loader):
        img, label = data
        img, label = img.to(device), label.to(device)

        optimizer.zero_grad()

        outputs, feature_loss = model(img)
        middle_output5, middle_output4, middle_output3, middle_output2, middle_output1 = outputs
        ensemble_out5 = (middle_output1 + middle_output2 + middle_output3 + middle_output4 + middle_output5) / 5

        final_loss = loss_function(middle_output5, label)
        middle4_loss = loss_function(middle_output4, label)
        middle3_loss = loss_function(middle_output3, label)
        middle2_loss = loss_function(middle_output2, label)
        middle1_loss = loss_function(middle_output1, label)

        output_loss = final_loss + middle1_loss + middle2_loss + middle3_loss

        # 15,25,35,45

        loss1by5 = kd_loss_function(middle_output1, middle_output5) * T * T
        loss2by5 = kd_loss_function(middle_output2, middle_output5) * T * T
        loss3by5 = kd_loss_function(middle_output3, middle_output5) * T * T
        
        kd_loss = loss1by5 + loss2by5 + loss3by5

        loss = (1 - alpha) * output_loss + alpha * kd_loss + beta * feature_loss + middle4_loss

        loss.backward()
        optimizer.step()

        total_loss += final_loss.item()
        predict = torch.max(middle_output4, dim=1)[1]
        correct_num += torch.eq(predict, label).sum().item()

    train_loss = total_loss / length_train_loader
    train_acc = correct_num / total_sample
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)

    logger.info("Epoch: {} \t train_loss: {:.3f} \t lr: {:.5f} train_acc: {:.3f}".format(
        epoch + 1, train_loss, optimizer.param_groups[0]['lr'], train_acc
    ))

    return train_loss_list, train_acc_list


def self_kd_main():
    from wide_resnet_model import wide_resnet38_2

    self_kd = wide_resnet38_2(num_classes=200)
    self_kd.to(device)

    batch_size = 128
    setup_seed(100)
    optimizer = optim.SGD(self_kd.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    train_loader, test_loader = get_tiny_imagenet_data_loader(batch_size, train_root, val_root)

    logger.info('start defining training.')
    loss_function = nn.CrossEntropyLoss()

    logger.info('defining training end.')

    best_acc_branch1 = 0.0
    best_acc_branch2 = 0.0
    best_acc_branch3 = 0.0
    best_acc_branch4 = 0.0

    best_acc_ensemble4 = 0.0
    best_acc_ensemble3 = 0.0
    best_acc_ensemble2 = 0.0

    correct_branch4 = 0
    correct_branch3 = 0
    correct_branch2 = 0
    correct_branch1 = 0
    correct_ensemble = 0

    val_loss_list = []
    val_acc_list_branch1 = []
    val_acc_list_branch2 = []
    val_acc_list_branch3 = []
    val_acc_list_branch4 = []

    train_loss_list = []
    train_acc_list = []

    # if save_path:
    #     if os.path.isfile(save_path):
    #         self_kd = torch.load(save_path)

    total_time = 0.0
    start_epoch = 0

    for epoch in range(epochs - start_epoch):
        time_start = time.time()
        epoch = epoch + start_epoch
        logger.info('----------train-------------')
        train_loss_list, train_acc_list = self_kd_train(self_kd, train_loader, optimizer, loss_function, epoch,
                                                        train_loss_list, train_acc_list)

        logger.info('evaluating.')
        val_loss_list, val_acc_list_branch1, val_acc_list_branch2, val_acc_list_branch3, val_acc_list_branch4, \
        val_acc_branch1, val_acc_branch2, val_acc_branch3, val_acc_branch4, val_acc_ensemble4, val_acc_ensemble3, val_acc_ensemble2 = evaluating(
            self_kd,
            test_loader,
            optimizer,
            loss_function,
            epoch,
            val_loss_list,
            val_acc_list_branch1,
            val_acc_list_branch2,
            val_acc_list_branch3,
            val_acc_list_branch4,
            correct_branch1,
            correct_branch2,
            correct_branch3,
            correct_branch4,
            correct_ensemble)

        if val_acc_branch4 > best_acc_branch4:
            best_acc_branch4 = val_acc_branch4
            torch.save(self_kd.state_dict(), save_path)
        if val_acc_branch3 > best_acc_branch3:
            best_acc_branch3 = val_acc_branch3
        if val_acc_branch2 > best_acc_branch2:
            best_acc_branch2 = val_acc_branch2
        if val_acc_branch1 > best_acc_branch1:
            best_acc_branch1 = val_acc_branch1

        if val_acc_ensemble4 > best_acc_ensemble4:
            best_acc_ensemble4 = val_acc_ensemble4
        if val_acc_ensemble3 > best_acc_ensemble3:
            best_acc_ensemble3 = val_acc_ensemble3
        if val_acc_ensemble2 > best_acc_ensemble2:
            best_acc_ensemble2 = val_acc_ensemble2

        time_end = time.time()
        time_for_one_epoch = time_end - time_start
        total_time += time_for_one_epoch

    avg_acc_branch1 = mean(val_acc_list_branch1)
    avg_acc_branch2 = mean(val_acc_list_branch2)
    avg_acc_branch3 = mean(val_acc_list_branch3)
    avg_acc_branch4 = mean(val_acc_list_branch4)

    logger.info('total time: {}'.format(total_time))

    return self_kd, val_acc_list_branch1, best_acc_branch1, avg_acc_branch1, \
           val_acc_list_branch2, best_acc_branch2, avg_acc_branch2, \
           val_acc_list_branch3, best_acc_branch3, avg_acc_branch3, \
           val_acc_list_branch4, best_acc_branch4, avg_acc_branch4, \
           best_acc_ensemble4, best_acc_ensemble3, best_acc_ensemble2


if __name__ == '__main__':
    self_kd, val_acc_list_branch1, best_acc_branch1, avg_acc_branch1, \
    val_acc_list_branch2, best_acc_branch2, avg_acc_branch2, \
    val_acc_list_branch3, best_acc_branch3, avg_acc_branch3, \
    val_acc_list_branch4, best_acc_branch4, avg_acc_branch4, \
    best_acc_ensemble4, best_acc_ensemble3, best_acc_ensemble2 = self_kd_main()

    logger.info('the best acc and avg acc of branch1 are {} and {}'.format(best_acc_branch1,
                                                                           avg_acc_branch1))
    logger.info('the best acc and avg acc of branch2 are {} and {}'.format(best_acc_branch2,
                                                                           avg_acc_branch2))
    logger.info('the best acc and avg acc of branch3 are {} and {}'.format(best_acc_branch3,
                                                                           avg_acc_branch3))
    logger.info('the best acc and avg acc of branch4 are {} and {}'.format(best_acc_branch4,
                                                                           avg_acc_branch4))
    logger.info('the best acc of ensemble4, ensemble3 and ensemble2 are {}, {} and {}'.format(best_acc_ensemble4,
                                                                                              best_acc_ensemble3,
                                                                                              best_acc_ensemble2))
