import os
import sys
import json
import pickle
import random
from torch.autograd import Variable
import pandas as pd
import csv
import numpy as np
import torch
import math
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_fscore_support
import torch.nn.functional as F

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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



def txt_matrix(matrix):
    with open('./ReNeTC_TMC_respond.txt', 'w') as f:
        for row in matrix:
            f.write('\t'.join(map(str, row)) + '\n')

def plot_roc(probability_value, y_true, classes, uncertain_all):
    class_labels = list(range(0, classes - 1))
    y_true_binary = label_binarize(y_true, classes=class_labels)

    y_score = probability_value.cpu().numpy()

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(classes - 1):
        fpr[i], tpr[i], _ = roc_curve(y_true_binary[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])


    fpr["last"], tpr["last"], _ = roc_curve(y_true == classes - 1, uncertain_all)
    roc_auc["last"] = auc(fpr["last"], tpr["last"])

    plt.figure(figsize=(8, 6))
    for i in range(classes - 1):
        plt.plot(fpr[i], tpr[i], lw=2, label='Class {0} (AUC = {1:0.5f})'.format(i, roc_auc[i]))
    plt.plot(fpr["last"], tpr["last"], lw=2, linestyle='--',
             label='Class {0} (AUC = {1:0.2f})'.format(classes - 1, roc_auc["last"]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC')
    plt.legend(loc="lower right")
    plt.show()




def u_k_plot_roc(uncertain_all, y_true, classes, u_index, k_index):
    binary_labels = np.array([1 if label == classes - 1 else 0 for label in y_true])

    with open('./revise_u_{}_k{}_roc.csv'.format(u_index, k_index), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['y_true_binary', 'y_pred_test_prob'])
        for true_label, prob in zip(binary_labels, uncertain_all):
            writer.writerow([true_label, *prob])

    fpr, tpr, thresholds = roc_curve(binary_labels, uncertain_all)
    roc_auc = auc(fpr, tpr)

    loss_values = 2 * tpr - fpr
    max_loss_idx = np.argmax(loss_values)  # 找到最大值的索引
    optimal_threshold = thresholds[max_loss_idx]  # 对应的最优阈值
    print('optimal_threshold: ', optimal_threshold )

    # 绘制 ROC 曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.5f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for Known vs Unknown Classes')
    plt.legend(loc="lower right")
    plt.show()


def plot_uncertainty(data):

    sns.displot(data, bins=1000, label='Quantity', kde=False)

    plt.xlabel('Uncertainty')
    plt.ylabel('Density')
    plt.legend()
    plt.xlim(0,0.04)
    plt.xticks(rotation = -45, va='top')
    plt.tight_layout()

    plt.show()


def save_uncertainty(data, byte_num, packet_num, k_index, u_index, is_open):
    for i in data:
        if os.path.exists("./Save_Uncertainty") is False:
            os.makedirs("./Save_Uncertainty")

        # f = open('./Save_Uncertainty/CIC_256_4_U_0.3_IP_PAY_uncertainty.csv', 'a', newline='')
        if is_open:
            f = open(f'./Save_Uncertainty/Only_unknown_K_{k_index}_U_{u_index}_{byte_num}_{packet_num}_uncertainty.csv', 'a', newline='')
        else:
            f = open(f'./Save_Uncertainty/known_K_{k_index}_U_{u_index}_{byte_num}_{packet_num}_uncertainty.csv', 'a', newline='')

        writer = csv.writer(f)
        writer.writerow(i)

        f.close()


def multi_label_auc(y_true, y_pred):
    """
    计算多标签AUC的函数
    :param y_true: 真实标签，形状为[N, num_classes] 得是one_hot
    :param y_pred: 预测标签，形状为[N, num_classes] 得是概率值
    :return: 多标签AUC
    """

    total_auc = 0.

    for i in range(y_true.shape[1]):
        try:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        except ValueError:
            auc = 0.5
        total_auc += auc

    multi_auc = total_auc / y_true.shape[1]

    return multi_auc




def train_one_epoch(model, optimizer, train_loader, device, epoch, views):
    model.train()
    loss_meter = AverageMeter()
    tra_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    train_loader = tqdm(train_loader, file=sys.stdout)
    data_num = 0


    for batch_idx, (data, target) in enumerate(train_loader):

        for v_num in range(len(data)):
            data[v_num] = data[v_num].to(device)
        data_num += target.size(0)

        target = target.long().to(device)
        optimizer.zero_grad()

        evidences, envidence_a, _, loss = model(data, target, epoch,device)
        pre_cla = torch.max(envidence_a,1)[1]
        accu_num += torch.eq(pre_cla, target.to(device)).sum() #.detach()
        loss.backward()
        tra_loss += loss


        if epoch % 3 == 0:

            train_loader.desc = "[train epoch {}] loss: {:.5f}, acc: {:.5f}".format(epoch,
                                                                               tra_loss.item() / (batch_idx + 1),
                                                                               accu_num.item() / data_num)
            train_result=[]

            train_result.append([epoch, tra_loss.item() / (batch_idx + 1),  accu_num.item() / data_num])

        optimizer.step()


    return tra_loss.item() / (batch_idx + 1),  accu_num.item() / data_num




@torch.no_grad()
def evaluate(model, data_loader, device, epoch,views):
    model.eval()

    accu_num = torch.zeros(1).to(device)
    eval_loss = torch.zeros(1).to(device)
    data_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for batch_idx, (data, target) in enumerate(data_loader):
        for v_num in range(views):
            data[v_num] = data[v_num].to(device)
        data_num += target.size(0)
        with torch.no_grad():
            target = target.long().to(device)
            evidences, envidence_a, _, loss = model(data, target, epoch,device)
            pre_cla = torch.max(envidence_a, 1)[1]
            accu_num += torch.eq(pre_cla, target.to(device)).sum()


        if epoch % 3 == 0:
            data_loader.desc = "[valid epoch {}], acc: {:.5f}".format(epoch, accu_num.item() / data_num)
            val_result = []
            val_result.append([epoch,  accu_num.item() / data_num])


    return   accu_num.item() / data_num


def test(model, data_loader, device, classes, byte_num, packet_num, unknown_file, threshold, is_open, epoch, views):
    model.eval()

    accu_num = torch.zeros(1).to(device)
    test_loss = torch.zeros(1).to(device)
    data_num = 0


    if is_open:
        model_dir = "./Tmc_dataset_save_model/Tmc_{}_{}_best_model.pth".format(byte_num, packet_num)
    else:
        model_dir = "./Tmc_dataset_save_model_K/Tmc_{}_{}_best_model.pth".format(byte_num, packet_num)
    model.load_state_dict(torch.load(model_dir))


    y_true = torch.LongTensor(0).to(device)
    y_predict = torch.LongTensor(0).to(device)
    uncertain_all = torch.LongTensor(0).to(device)
    probability_value = torch.LongTensor(0).to(device)

    data_loader = tqdm(data_loader, file=sys.stdout)
    if not is_open:
        k_index = model_dir.split('/')[1].split('_')[0]  ##已知数据集
        u_index = ' '
        for batch_idx, (data, target) in enumerate(data_loader):
            for v_num in range(views):
                data[v_num] = data[v_num].to(device)
            data_num += target.size(0)
            with torch.no_grad():
                target = target.long().to(device)
                evidences, envidence_a, uncertain, loss = model(data, target, epoch, device)

                uncertain_batch = uncertain[:, 0]
                uncertain[:, 0] = torch.tensor(uncertain_batch)
                uncertain_all = torch.cat([uncertain_all, uncertain], 0)


                pre_cla = torch.max(envidence_a, 1)[1]
                y_predict = torch.cat([y_predict, pre_cla], 0)
                y_true = torch.cat([y_true, target],0)

                accu_num += torch.eq(pre_cla, target.to(device)).sum()
                test_loss += loss
            data_loader.desc = "[test loss: {:.5f}], acc: {:.5f}]".format(test_loss.item() / (batch_idx + 1),
                                                                          accu_num.item() / data_num)


    else:
        k_index = model_dir.split('/')[1].split('_')[0]  ##已知数据集
        u_index = unknown_file.split('_')[0]
        for batch_idx, (data, target) in enumerate(data_loader):
            for v_num in range(views):
                data[v_num] = data[v_num].to(device)
            data_num += target.size(0)
            with torch.no_grad():
                target = target.long().to(device)
                evidences, envidence_a, uncertain, loss = model(data, target, epoch, device)

                pre_cla = torch.max(envidence_a, 1)[1]
                uncertain_batch = uncertain[:,0]

                for i in range(len(uncertain_batch)):
                    if uncertain_batch[i] > threshold:
                        pre_cla[i] = classes - 1

                uncertain[:,0] = torch.tensor(uncertain_batch)
                uncertain_all = torch.cat([uncertain_all, uncertain], 0)
                y_predict = torch.cat([y_predict, pre_cla], 0)
                y_true = torch.cat([y_true, target], 0)

                probability_value = torch.cat([probability_value, envidence_a], 0)

                accu_num += torch.eq(pre_cla, target.to(device)).sum()

            data_loader.desc = "[test acc: {:.5f}]".format(accu_num.item() / data_num)

    uncertain_all = uncertain_all.cpu().numpy()
    y_predict = y_predict.cpu().numpy()
    y_true = y_true.cpu().numpy()


    matrix = confusion_matrix(y_true, y_predict)


    # #
    # plot_uncertainty(uncertain_all)
    # # #
    # save_uncertainty(uncertain_all.numpy(), byte_num, packet_num, k_index, u_index, is_open)

    # #
    # plot_roc(probability_value, y_true, classes, uncertain_all)

    # # #
    # u_k_plot_roc(uncertain_all, y_true, classes, u_index, k_index)




    precision, recall, fmeasure, support = precision_recall_fscore_support(y_true, y_predict)
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_fmeasure = np.mean(fmeasure)

    print('-------recall: {}'.format(recall))
    print('-------precision: {}'.format(precision))
    print('-------fmeasure: {}'.format(fmeasure))
    print('-------macro recall: {:.5f}'.format(macro_recall))
    print('-------macro precision: {:.5f}'.format(macro_precision))
    print('-------macro fmeasure: {:.5f}'.format(macro_fmeasure))


    print(matrix)


    return test_loss.item() / (batch_idx + 1), accu_num.item() / data_num