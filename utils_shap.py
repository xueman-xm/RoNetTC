"""
data: 2024/11/21-16:17
"""
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
import shap
from makse_model import MaskedModel
shap.utils._masked_model.MaskedModel = MaskedModel






# from shap.utils import MaskedModel
# print('shap.__file__:', shap.__file__)

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


def plot_roc(probability_value, y_true, classes, uncertain_all):
    # 二值化标签，只考虑前 classes-1 个类
    class_labels = list(range(0, classes - 1))
    y_true_binary = label_binarize(y_true, classes=class_labels)

    y_score = probability_value.cpu().numpy()

    # 计算前 classes-1 个类的ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(classes - 1):
        fpr[i], tpr[i], _ = roc_curve(y_true_binary[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 特殊处理最后一个类别
    # 假设 uncertain_all 已经是一个合适的概率分数表示
    # 注意：这里可能需要根据 uncertain_all 的具体含义进行调整
    # 例如，如果需要，可以使用 fpr["last"], tpr["last"], _ = roc_curve(y_true_binary[:, -1], 1 - uncertain_all)
    fpr["last"], tpr["last"], _ = roc_curve(y_true == classes - 1, uncertain_all)
    roc_auc["last"] = auc(fpr["last"], tpr["last"])

    # 绘制ROC曲线
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
    # 生成二元标签：所有已知类别标为0，未知类别标为1
    binary_labels = np.array([1 if label == classes - 1 else 0 for label in y_true])

    with open('./revise_u_{}_k{}_roc.csv'.format(u_index, k_index), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['y_true_binary', 'y_pred_test_prob'])  # 写入列名
        for true_label, prob in zip(binary_labels, uncertain_all):
            writer.writerow([true_label, *prob])  # 分行写入数据


    # 使用 uncertain_all 作为预测概率，计算 ROC 曲线和 AUC
    fpr, tpr, thresholds = roc_curve(binary_labels, uncertain_all)
    roc_auc = auc(fpr, tpr)

    # 计算 ℓ(θ) = 2 * TPR(θ) - FPR(θ)
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
    # sns.kdeplot(data,label='Quantity')
    # sns.kdeplot(data, label='Quantity', fill=True, color='b', cut=0, clip=(0, 1), bw_adjust=0.6)
    # sns.kdeplot(data, label='Density',bw_adjust=0.4, bw_method=0.5, fill=False,
    #             legend=True)  # '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
    # plt.xticks([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
    #            ['0', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4'])
    # ax.set(yticklabels=[ ])
    # ax.set(ylabel=None)

    # data = [-math.log(i) for i in data]
    # sns.displot(data, bins = 1000, label='Quantity', kde=False, kde_kws={'bw_adjust':0.0003})
    sns.displot(data, bins=1000, label='Quantity', kde=False)
    # sns.lineplot(data,  label="Density")
    plt.xlabel('Uncertainty')
    plt.ylabel('Density')
    # plt.figure(figsize=(10,6))
    plt.legend()
    plt.xlim(0,0.04)
    # plt.ylim(0,500)
    plt.xticks(rotation = -45, va='top')
    # plt.yticks([])
    plt.tight_layout()


    # # fig = plt.gcf()
    # # size = fig.get_size_inches()

    plt.show()
    # print('figure size: ',size)


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
    # 将标签转换为numpy数组
    # y_true = y_true.cpu().numpy()
    # y_pred = y_pred.cpu().numpy()


    # 初始化多标签AUC值
    total_auc = 0.

    # 计算每个标签的AUC值，并对所有标签的AUC值求平均
    for i in range(y_true.shape[1]):
        try:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        except ValueError:
            auc = 0.5  # 如果标签中只有一个类别，则返回0.5
        total_auc += auc

    multi_auc = total_auc / y_true.shape[1]

    return multi_auc




def train_one_epoch(model, optimizer, train_loader, device, epoch, views):
    model.train()
    loss_meter = AverageMeter()
    tra_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)
    train_loader = tqdm(train_loader, file=sys.stdout)
    data_num = 0
    # # optimizer.zero_grad()


    for batch_idx, (data, target) in enumerate(train_loader):

        for v_num in range(len(data)):
            # data[v_num] = Variable(data[v_num].to(device))
            data[v_num] = data[v_num].to(device)
        data_num += target.size(0)

        target = target.long().to(device)
        optimizer.zero_grad()
        # data.to(device)
        # data_num += data.shape[0]
        # # print('type(target)',type(target))
        # # exit()
        # # target = Variable(target.long().cuda())
        # target.to(device)

        evidences, envidence_a, _, loss = model(data, target, epoch,device)
        pre_cla = torch.max(envidence_a,1)[1]
        accu_num += torch.eq(pre_cla, target.to(device)).sum() #.detach()
        loss.backward()
        # accu_loss += loss.clone().detach()
        tra_loss += loss


        if epoch % 3 == 0:
            # print("[train epoch {}] loss: {:.4f}, acc: {:.4f}%".format(epoch, tra_loss.item() / (batch_idx + 1),
            #                                                            (accu_num.item() * 100.)/ data_num))
            train_loader.desc = "[train epoch {}] loss: {:.5f}, acc: {:.5f}".format(epoch,
                                                                               tra_loss.item() / (batch_idx + 1),
                                                                               accu_num.item() / data_num)
            train_result=[]

            train_result.append([epoch, tra_loss.item() / (batch_idx + 1),  accu_num.item() / data_num])

        # if not torch.isfinite(loss):
        #     print('WARNING: non-finite loss, ending training ', loss)
        #     sys.exit(1)

        optimizer.step()
        # # optimizer.zero_grad()


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
            # data[v_num] = Variable(data[v_num].to(device))
            data[v_num] = data[v_num].to(device)
        # data.to(device)
        data_num += target.size(0)
        with torch.no_grad():
            # target = Variable(target.long().cuda())
            target = target.long().to(device)
        # target.to(device)
            evidences, envidence_a, _, loss = model(data, target, epoch,device)
            pre_cla = torch.max(envidence_a, 1)[1]
            accu_num += torch.eq(pre_cla, target.to(device)).sum()


        if epoch % 3 == 0:
            # print("[valid epoch {}], acc: {:.5f}%".format(epoch,  (accu_num.item() * 100.)/ data_num))
            data_loader.desc = "[valid epoch {}], acc: {:.5f}".format(epoch, accu_num.item() / data_num)
            val_result = []
            val_result.append([epoch,  accu_num.item() / data_num])


    return   accu_num.item() / data_num


# def test_with_shap(model, data_loader, device, classes, byte_num, packet_num, unknown_file, threshold, is_open, epoch, views):
#     model.eval()
#
#     accu_num = torch.zeros(1).to(device)
#     test_loss = torch.zeros(1).to(device)
#     data_num = 0
#     # model_dir = './Ablation_experiment/CIC_U_0.3_IP_PAY_256_4_best_model.pth'
#     model_dir = "./Tmc_dataset_save_model/1_IP_Tmc_{}_{}_best_model.pth".format(byte_num, packet_num)
#     model.load_state_dict(torch.load(model_dir))
#
#
#     y_true = torch.LongTensor(0).to(device)
#     y_predict = torch.LongTensor(0).to(device)
#     uncertain_all = torch.LongTensor(0).to(device)
#     probability_value = torch.LongTensor(0).to(device)
#
#     data_loader = tqdm(data_loader, file=sys.stdout)
#     if is_open:
#         k_index = model_dir.split('/')[1].split('_')[0]  ##已知数据集
#         u_index = unknown_file.split('_')[0]
#         for batch_idx, (data, target) in enumerate(data_loader):
#             for v_num in range(views):
#                 # data[v_num] = Variable(data[v_num].to(device))
#                 data[v_num] = data[v_num].to(device)
#             # data.to(device)
#             data_num += target.size(0)
#             with torch.no_grad():
#                 # target = Variable(target.long().cuda())
#                 target = target.long().to(device)
#                 # target.to(device)
#                 evidences, envidence_a, uncertain, loss = model(data, target, epoch, device)
#                 # pre_a = F.softmax(envidence_a, dim=1)
#
#                 pre_cla = torch.max(envidence_a, 1)[1]
#                 uncertain_batch = uncertain[:,0]
#                 # uncertain_batch = [math.log(10*i) for i in uncertain_batch]
#
#                 for i in range(len(uncertain_batch)):
#                     if uncertain_batch[i] > threshold:
#                         pre_cla[i] = classes - 1
#
#                 uncertain[:,0] = torch.tensor(uncertain_batch)
#                 uncertain_all = torch.cat([uncertain_all, uncertain], 0)
#                 y_predict = torch.cat([y_predict, pre_cla], 0)
#                 y_true = torch.cat([y_true, target], 0)
#                 # 概率值
#                 # y_true_one_h = F.one_hot(y_true, classes)
#                 probability_value = torch.cat([probability_value, envidence_a], 0)
#
#                 accu_num += torch.eq(pre_cla, target.to(device)).sum()
#                 # test_loss += loss
#
#             data_loader.desc = "[test acc: {:.5f}]".format(accu_num.item() / data_num)


class DictMasker:
    def __init__(self, data):
        """Initialize the DictMasker with data."""
        self.args = [v for v in data.values()]  # 将所有张量作为 args
        print(f"DictMasker initialized with args: {[arg.shape for arg in self.args]}")

    def __call__(self, mask, data):
        print("DictMasker __call__ invoked.")
        print(f"Mask: {mask}, Type: {type(mask)}")
        print(f"Data Keys: {list(data.keys())}")
        masked_data = {}
        for key, value in data.items():
            print(f"Key: {key}, Value Shape: {value.shape}, Device: {value.device}")
            # 将 mask_tensor 移动到 value 的设备上
            mask_tensor = torch.tensor(mask[:value.numel()], dtype=value.dtype, device=value.device).view(value.shape)
            masked_data[key] = value * mask_tensor
        return masked_data


def test_with_shap(model, data_loader, device, classes, byte_num, packet_num, unknown_file, threshold, is_open, epoch, views):
    model.eval()
    accu_num = torch.zeros(1).to(device)
    test_loss = torch.zeros(1).to(device)
    data_num = 0
    model_dir = "./Tmc_dataset_save_model/1_IP_Tmc_{}_{}_best_model.pth".format(byte_num, packet_num)
    model.load_state_dict(torch.load(model_dir))

    y_true = torch.LongTensor(0).to(device)
    y_predict = torch.LongTensor(0).to(device)
    uncertain_all = torch.LongTensor(0).to(device)
    probability_value = torch.LongTensor(0).to(device)
    shap_values_list = []  # Collect SHAP values for each sample


    shap_pre = []
    shap_true = []
    accu_shap = 0
    # Define SHAP explainer using a reference dataset
    def model_predict(inputs):
        y_shap_extracted = torch.tensor(inputs[0, -1], dtype=torch.long).to(device)  # 取最后一列的值
        inputs = inputs[:, :-1]
        # print(f'inputs len(): {len(inputs)}')
        # print(f'inputs[0] len(): {len(inputs[0])}')
        inputs_tensor = torch.tensor(inputs, dtype=torch.long)
        padding_dim = inputs_tensor.shape[-1]
        padded_inputs = {}
        padded_inputs[0] = inputs_tensor
        for i in range(1, packet_num):
            # padded_inputs[i] = torch.zeros((1, padding_dim), dtype=torch.long)  # 用零填充
            padded_inputs[i] = inputs_tensor

        padded_inputs_device = {key: value.to(device) for key, value in padded_inputs.items()}
        # print(f'padded_inputs_device[4].shape: {padded_inputs_device[4].shape} ')
        # print(f'padded_inputs_device[0]: {padded_inputs_device[0]}')

        with torch.no_grad():
            outputs = model(padded_inputs_device, y_shap_extracted, 1, device)
            # print(f"Model Outputs: {outputs}")

        return outputs[1].cpu().numpy()

    data_loader = tqdm(data_loader, file=sys.stdout)

    if not is_open:
        k_index = model_dir.split('/')[1].split('_')[0]  ##已知数据集
        u_index = ' '
        for batch_idx, (data, target) in enumerate(data_loader):
            for v_num in range(views):
                # data[v_num] = Variable(data[v_num].to(device))
                data[v_num] = data[v_num].to(device)
            # data.to(device)
            data_num += target.size(0)
            with torch.no_grad():
                # target = Variable(target.long().cuda())
                target = target.long().to(device)
                # target.to(device)
                evidences, envidence_a, uncertain, loss = model(data, target, epoch, device)
                # pre_a = F.softmax(envidence_a, dim=1)

                uncertain_batch = uncertain[:, 0]
                uncertain[:, 0] = torch.tensor(uncertain_batch)
                uncertain_all = torch.cat([uncertain_all, uncertain], 0)

                pre_cla = torch.max(envidence_a, 1)[1]
                y_predict = torch.cat([y_predict, pre_cla], 0)
                y_true = torch.cat([y_true, target], 0)

                accu_num += torch.eq(pre_cla, target.to(device)).sum()
                test_loss += loss



        # k_index = model_dir.split('/')[1].split('_')[0]  # Known dataset
        # u_index = unknown_file.split('_')[0]
        #
        # for batch_idx, (data, target) in enumerate(data_loader):
        #     for v_num in range(views):
        #         data[v_num] = data[v_num].to(device)
        #     data_num += target.size(0)
        #
        #     with torch.no_grad():
        #         target = target.long().to(device)
        #         evidences, envidence_a, uncertain, loss = model(data, target, epoch, device)
        #         pre_cla = torch.max(envidence_a, 1)[1]
        #         uncertain_batch = uncertain[:, 0]
        #
        #         for i in range(len(uncertain_batch)):
        #             if uncertain_batch[i] > threshold:
        #                 pre_cla[i] = classes - 1
        #
        #         y_predict = torch.cat([y_predict, pre_cla], 0)
        #         y_true = torch.cat([y_true, target], 0)
        #         probability_value = torch.cat([probability_value, envidence_a], 0)
        #         accu_num += torch.eq(pre_cla, target.to(device)).sum()

            # Compute SHAP values for the current batch
            b = data[0]
            sample_idx = 0  # 从 batch 中选择一个样本
            y_shap = target[sample_idx]
            # print(f'y_shap,type: {type(y_shap)}')
            sample_data = {key: value[sample_idx:sample_idx+1, :] for key, value in data.items()}
            flattened_data = [sample_data[key] for key in sample_data]

            batch_shap_values = []

            # # 创建 SHAP Explainer
            # explainer = shap.Explainer(model_predict, flattened_data)
            #
            # # 计算 SHAP 值
            # shap_values = explainer(flattened_data)  # SHAP values for 当前样本
            # batch_shap_values.append(shap_values)  # 存储当前样本的 SHAP 值
            #
            # # Post-process or visualize SHAP values
            # for idx, shap_vals in enumerate(batch_shap_values):
            #     print(f"Sample {idx}: SHAP Values")
            #     for v_num, shap_value in enumerate(shap_vals):
            #         print(f" View {v_num}: {shap_value}")

            y_shap_tensor = y_shap.view(1, 1)  # 扩展为 (1, 1) 的张量
            # 遍历 sample_data 中每个包（假设 sample_data 是一个包含 8 个视角数据的字典）
            for idx, (key, view_data) in enumerate(sample_data.items()):
                # print(f"Processing packet {idx + 1}, view {key}")
                y_shap_tensor = y_shap_tensor.to(view_data.device)
                flattened_data = torch.cat((view_data, y_shap_tensor), dim=1)

                # view_data 是当前包的单个视角数据 (1, feature_dim)，直接用作 SHAP 输入
                flattened_data = flattened_data.cpu().numpy()  # 当前包的单视角数据
                # print(f'flattened_data: {flattened_data.shape}')
                masker = shap.maskers.Independent(flattened_data)

                # 创建 SHAP Explainer
                explainer = shap.Explainer(model_predict, masker)
                predictions = model_predict(flattened_data)
                pre_cla = np.argmax(predictions)
                shap_pre.append(pre_cla)
                shap_true.append(y_shap.cpu().numpy())
                # print(f'pre_cla: {pre_cla}')

                # 计算 SHAP 值
                shap_values = explainer(flattened_data)  # SHAP values for 当前包
                shap_values = shap_values[:, :-1, :]  # 去掉最后一个特征，变成 (1, 20, 13)
                batch_shap_values.append(shap_values)  # 存储当前包的 SHAP 值

                # # 可视化或输出 SHAP 值
                # print(f"Packet {idx + 1} SHAP Values:")
                # for feature_idx, feature_value in enumerate(shap_values[0].values):
                #     print(f"  Feature {feature_idx}: SHAP value = {feature_value}")
            shap_values_list.append(batch_shap_values)

    shap_true = np.array(shap_true).astype(int)  # 转为整数
    shap_pre = np.array(shap_pre).astype(int)
    # print("Shape of shap_pre:", shap_pre.shape)
    # print("Shape of shap_true:", shap_true.shape)
    #
    # print(f'len(shap_pre): {len(shap_pre)}')
    # print(f'len(shap_pre): {len(shap_true)}')
    correct_predictions = np.sum(shap_pre == shap_true)
    # print(f'correct_predictions: {correct_predictions}')
    total_samples = len(shap_true)
    # print(f"Total samples: {total_samples}")
    accuracy = correct_predictions / total_samples
    print(f"Accuracy: {accuracy* 100:.2f}%")

    uncertain_all = uncertain_all.cpu().numpy()
    y_predict = y_predict.cpu().numpy()
    y_true = y_true.cpu().numpy()
    # y_true_one_h = y_true_one_h.cpu().numpy()
    # pre_num= pre_num.cpu() / data_num

    matrix = confusion_matrix(y_true, y_predict)
    precision, recall, fmeasure, support = precision_recall_fscore_support(y_true, y_predict)
    # 计算宏平均的精度、召回率和F1值
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_fmeasure = np.mean(fmeasure)

    print('-------recall: {}'.format(recall))
    print('-------precision: {}'.format(precision))
    print('-------fmeasure: {}'.format(fmeasure))
    print('-------macro recall: {:.5f}'.format(macro_recall))
    print('-------macro precision: {:.5f}'.format(macro_precision))
    print('-------macro fmeasure: {:.5f}'.format(macro_fmeasure))
    # print(matrix)
    feature_names = [f'feature_{i}' for i in range(20)]  # 动态生成特征名称
    output_folder = "./SHAP_fig"
    os.makedirs(output_folder, exist_ok=True)
    for package_idx in range(packet_num):  # 假设有 packet_num 个包（视图）
        # 初始化一个 list 用于存储该包所有样本的 SHAP 值
        all_shap_values = []

        # 收集所有样本在当前包的 SHAP 值
        for sample_shap_values in shap_values_list:
            # 获取该包的 SHAP 值，shape 可能是 (90, 21, 13)，你需要将维度 13 合并或移除
            shap_values_for_package = sample_shap_values[package_idx].values  # 这个是一个 shape 为 (90, 21, 13) 的数组
            shap_values_for_package = shap_values_for_package.mean(axis=-1)  # 对最后一个维度（13）求平均，得到 (90, 21)
            all_shap_values.append(shap_values_for_package)

        # 将所有样本的 SHAP 值合并成一个大的 numpy 数组
        all_shap_values = np.concatenate(all_shap_values, axis=0)
        print(f'all_shap_values.shape: {all_shap_values.shape}')  # 现在应该是 (90, 21)

        # 创建 SHAP Summary 图
        plt.figure()
        shap.summary_plot(all_shap_values, feature_names=feature_names, show=False)  # 不直接显示图
        plt.title(f"SHAP Summary for Package {package_idx + 1}")

        # 保存图到指定文件夹
        output_path = os.path.join(output_folder, f"SHAP_Summary_Package_{package_idx + 1}.png")
        plt.savefig(output_path)
        plt.close()  # 关闭当前图，以防止内存溢出
