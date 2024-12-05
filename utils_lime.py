"""
data: 2024/11/18-14:54
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


from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as pl
from skimage.segmentation import slic
from skimage.color import gray2rgb  # 防止单通道数据报错
from lime.lime_tabular import LimeTabularExplainer
from torch.utils.data import Subset
from sklearn.preprocessing import StandardScaler
from lime.lime_text import LimeTextExplainer
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors

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


# def test(model, data_loader, device, classes, byte_num, packet_num, unknown_file, threshold, is_open, epoch, views):
#     model.eval()
#
#     accu_num = torch.zeros(1).to(device)
#     test_loss = torch.zeros(1).to(device)
#     data_num = 0
#     # model_dir = './Ablation_experiment/CIC_U_0.3_IP_PAY_256_4_best_model.pth'
#     model_dir = "./Tmc_dataset_save_model/Tmc_{}_{}_best_model.pth".format(byte_num, packet_num)
#     model.load_state_dict(torch.load(model_dir))
#
#
#     y_true = torch.LongTensor(0).to(device)
#     y_predict = torch.LongTensor(0).to(device)
#     uncertain_all = torch.LongTensor(0).to(device)
#     probability_value = torch.LongTensor(0).to(device)
#
#     data_loader = tqdm(data_loader, file=sys.stdout)
#     if not is_open:
#         k_index = model_dir.split('/')[1].split('_')[0]  ##已知数据集
#         u_index = ' '
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
#                 uncertain_batch = uncertain[:, 0]
#                 uncertain[:, 0] = torch.tensor(uncertain_batch)
#                 uncertain_all = torch.cat([uncertain_all, uncertain], 0)
#
#
#                 pre_cla = torch.max(envidence_a, 1)[1]
#                 y_predict = torch.cat([y_predict, pre_cla], 0)
#                 y_true = torch.cat([y_true, target],0)
#
#                 accu_num += torch.eq(pre_cla, target.to(device)).sum()
#                 test_loss += loss
#             data_loader.desc = "[test loss: {:.5f}], acc: {:.5f}]".format(test_loss.item() / (batch_idx + 1),
#                                                                           accu_num.item() / data_num)
#
#
#     else:
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




def prepare_inputs(data, packet_num):
    """
    将单视角的 `data` 整理为模型可接受的输入格式。
    """
    # 从字典中提取 `packet_num` 个张量，并将它们堆叠到第二个维度
    view_tensors = [data[str(i)] for i in range(packet_num)]
    inputs = torch.stack(view_tensors, dim=1)  # 堆叠，形状为 (batch_size, packet_num, 20)
    return inputs



# def test_with_lime_01(model, data_loader, device, classes, byte_num, packet_num, unknown_file, threshold, is_open, epoch, views):
#     model.eval()
#
#     accu_num = torch.zeros(1).to(device)
#     test_loss = torch.zeros(1).to(device)
#     data_num = 0
#     model_dir = "./Tmc_dataset_save_model/1_IP_Tmc_{}_{}_best_model.pth".format(byte_num, packet_num)
#     model.load_state_dict(torch.load(model_dir))
#
#     y_true = torch.LongTensor(0).to(device)
#     y_predict = torch.LongTensor(0).to(device)
#     dataset = data_loader.dataset  # 原始的 dataset 对象
#     data_loader = tqdm(data_loader, file=sys.stdout)
#
#     for batch_idx, (data, target) in enumerate(data_loader):
#         data_num += target.size(0)
#         with torch.no_grad():
#             target = target.long().to(device)
#
#             # 整理输入数据，单视角情况
#             # inputs = prepare_inputs(data, packet_num)  # 形状为 (batch_size, packet_num, 20)
#
#             # 调用模型
#             evidences, envidence_a, uncertain, loss = model(data, target, epoch, device)
#
#             pre_cla = torch.max(envidence_a, 1)[1]
#             y_predict = torch.cat([y_predict, pre_cla], 0)
#             y_true = torch.cat([y_true, target], 0)
#
#             accu_num += torch.eq(pre_cla, target.to(device)).sum()
#             test_loss += loss
#             data_loader.desc = "[test loss: {:.5f}], acc: {:.5f}]".format(test_loss.item() / (batch_idx + 1),
#                                                                           accu_num.item() / data_num)
#
#         # ======= LIME Explanation =======
#         if batch_idx == 0:  # 仅解释第一个 batch 的样本
#
#             sample_idx = 0  # 从 batch 中选择一个样本
#
#             # # 访问整数键，提取第一个键对应的单个样本数据
#             sample_data = {key: value[sample_idx:sample_idx+1, :] for key, value in data.items()}
#
#             print(f'sample_data[0].shape: {sample_data[0].shape}')
#             # sample_data = data[sample_idx].cpu().numpy()  # 使用整数键访问数据
#
#             # # LIME 需要输入的数据为二维的 (1, feature_dim)
#             # sample_data = sample_data.reshape(1, -1)  # 保持一维即可，无需再处理成伪图像
#
#             sample_target = target[sample_idx].item()
#
#             # 将 sample_data 转换为适合 LimeTabularExplainer 的格式（展平到一维）
#             sample_data_flat = torch.cat([v.view(1, -1) for v in sample_data.values()], dim=1).cpu().long().numpy().reshape(-1)
#
#             print(f'sample_data_flat: {sample_data_flat.shape}')
#             # # 构造 LIME 输入格式（值a索引形式的字符串）
#             # pkt_bytelen = sample_data_flat.shape[0]
#             # src_text = ' '.join([f"{sample_data_flat[i]}a{i}" for i in range(pkt_bytelen)])
#
#             # 自定义预测函数
#             def predict_fn(images):
#                 print(f"Predicting on {len(images)} samples...")
#
#                 batch_size = images.shape[0]
#                 # packet_dim = sample_data[sample_idx].shape[1]  # 提取每个 packet 的维度（假设固定）
#                 packet_dim = sample_data[0].shape[1]  # 提取每个 packet 的维度（假设固定）
#
#                 images_tensor = torch.tensor(images).long().to(device)
#                 # 创建伪数据字典
#                 data_dict = {key: images_tensor[:, i * packet_dim:(i + 1) * packet_dim].to(device).long()
#                              for i, key in enumerate(sample_data.keys())}
#
#                 # 提供伪标签
#                 dummy_y = torch.zeros(batch_size, dtype=torch.long, device=device)
#
#                 # 调用模型
#                 with torch.no_grad():
#                     outputs = model(data_dict, dummy_y, 1, device)  # 使用伪标签
#                     probs = torch.nn.functional.softmax(outputs[1], dim=1)  # 获取 softmax 概率
#
#                 result = probs.cpu().detach().numpy()
#                 print(f"Row sums: {np.sum(result, axis=1)}")  # 应该接近 1
#                 print(f"Result min: {result.min()}, max: {result.max()}, mean: {result.mean()}")
#
#                 print(f'result.shape:{result.shape}')
#                 for i in range(result.shape[1]):
#                     print(f"Class {i}: min={result[:, i].min()}, max={result[:, i].max()}, mean={result[:, i].mean()}")
#
#                 return result
#
#             class_names = [f"Class {i}" for i in range(classes)]  # 生成 ["Class 0", "Class 1", ...]
#             print(f'class_names len(): {len(class_names)}')
#
#             real_training_data_list = []  # 用于存储处理后的样本
#
#             for i in range(2400):
#                 i_value = dataset[i][0]
#                 flattened_tensor = torch.cat([v.view(1, -1) for v in i_value.values()], dim=1).cpu().long().numpy().reshape(-1)
#                 real_training_data_list.append(flattened_tensor)
#
#             real_training_data_array = np.array(real_training_data_list)
#
#             # 创建 LimeTabularExplainer
#             explainer = LimeTabularExplainer(
#                 training_data=real_training_data_array,  # 伪造的训练数据，用于初始化
#                 mode="classification",
#                 feature_names=[f"feature_{i}" for i in range(sample_data_flat.shape[0])],
#                 class_names=class_names
#             )
#
#             # 生成 LIME 解释
#             explanation = explainer.explain_instance(
#                 data_row=sample_data_flat,  # 输入样本数据（展平后的一维数组）
#                 predict_fn=predict_fn,  # 自定义预测函数
#                 num_features=10,  # 显示的重要特征数
#                 num_samples=2400 # 限制扰动样本数量为500
#                 # labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # 指定所有类别
#
#             )
#             print(f'training_data shape: {real_training_data_array.shape}')
#             print(f'sample_data_flat shape: {sample_data_flat.shape}')
#             print(f'training_data range: {real_training_data_array.min()} - {real_training_data_array.max()}')
#             print(f'sample_data_flat range: {sample_data_flat.min()} - {sample_data_flat.max()}')
#
#             # 绘制解释结果
#             print(f'explanation.top_labels:{explanation.top_labels}')  # 打印 top_labels
#             # label_to_use = explanation.top_labels[0]
#             # print(f'explanation.as_list(): {explanation.as_list()}')
#
#             # # 提取相应的特征权重
#             # weights = explanation.as_list(label=label_to_use)
#             # features, feature_weights = zip(*weights)
#             # plt.barh(features, feature_weights)
#             # plt.title(f"LIME Feature Importance (Class: {classes[sample_target]})")
#             # plt.xlabel("Weight")
#             # plt.ylabel("Feature")
#             # plt.show()
#
#
#             # # 直接访问 local_exp
#             for label, feature_weights in explanation.local_exp.items():
#                 print(f"Explanation for class {label}")
#
#                 # 提取特征和权重
#                 positive_feature_weights = [(f, w) for f, w in feature_weights if w > 0]  # 筛选正向权重
#                 if not positive_feature_weights:
#                     print(f"No positive contributions for class {label}")
#                     continue
#
#                 features, weights = zip(*positive_feature_weights)  # 解包正向特征和权重
#
#                 # 绘制正向特征的重要性
#                 plt.barh(features, weights)
#                 plt.title(f"LIME Positive Feature Importance (Class: {label})")
#                 plt.xlabel("Weight")
#                 plt.ylabel("Feature")
#                 plt.show()



def test_with_lime(model, data_loader, device, classes, byte_num, packet_num, unknown_file, threshold, is_open, epoch, views):
    model.eval()

    accu_num = torch.zeros(1).to(device)
    test_loss = torch.zeros(1).to(device)
    data_num = 0
    datasets_index = 'IP'

    model_dir = "./Tmc_dataset_save_model/1_{}_Tmc_{}_{}_best_model.pth".format(datasets_index, byte_num, packet_num)
    model.load_state_dict(torch.load(model_dir))

    y_true = torch.LongTensor(0).to(device)
    y_predict = torch.LongTensor(0).to(device)
    dataset = data_loader.dataset  # 原始的 dataset 对象
    data_loader = tqdm(data_loader, file=sys.stdout)
    explainer_num = 0
    output_folder = "./SHAP_fig"
    os.makedirs(output_folder, exist_ok=True)

    importances_max, importances_min = [], []
    for batch_idx, (data, target) in enumerate(data_loader):
        data_num += target.size(0)
        with torch.no_grad():
            target = target.long().to(device)

            # 整理输入数据，单视角情况
            # inputs = prepare_inputs(data, packet_num)  # 形状为 (batch_size, packet_num, 20)

            # 调用模型
            evidences, envidence_a, uncertain, loss = model(data, target, epoch, device)

            pre_cla = torch.max(envidence_a, 1)[1]
            y_predict = torch.cat([y_predict, pre_cla], 0)
            y_true = torch.cat([y_true, target], 0)

            accu_num += torch.eq(pre_cla, target.to(device)).sum()
            test_loss += loss
            data_loader.desc = "[test loss: {:.5f}], acc: {:.5f}]".format(test_loss.item() / (batch_idx + 1),
                                                                          accu_num.item() / data_num)

        # ======= LIME Explanation =======

        sample_idx = 0  # 从 batch 中选择一个样本

        # 访问整数键，提取第一个键对应的单个样本数据
        sample_data = {key: value[sample_idx:sample_idx + 1, :] for key, value in data.items()}
        packet_dim = sample_data[0].shape[1]  # 提取每个 packet 的维度
        sample_target = target[sample_idx].item()

        # 将 sample_data 转换为适合 LimeTextExplainer 的文本格式
        sample_data_flat = torch.cat([v.view(1, -1) for v in sample_data.values()], dim=1).cpu().long().numpy().reshape(-1)

        # 构造 LIME 输入格式（值a索引形式的字符串）
        pkt_bytelen = sample_data_flat.shape[0]
        src_text = ' '.join([f"{sample_data_flat[i]}a{i}" for i in range(pkt_bytelen)])

        def predict_fn(texts, batch_size_limit=64):
            # print(f"Predicting on {len(texts)} samples...")

            pkt_bytelen = sample_data_flat.shape[0]  # 确保使用正确的 pkt_bytelen
            packet_dim = sample_data[0].shape[1]  # 提取每个 packet 的维度

            results = []  # 保存预测结果
            for start in range(0, len(texts), batch_size_limit):
                end = min(start + batch_size_limit, len(texts))
                batch_texts = texts[start:end]

                # 将文本数据还原为数值矩阵
                images_tensor = []
                for text in batch_texts:
                    features = [0] * pkt_bytelen

                    # 遍历每个 "值a索引" 对
                    for item in text.split():
                        if 'a' in item:
                            value, index = item.split('a')
                            value = int(value.strip())  # 提取值
                            index = int(index.strip())  # 提取索引
                            if 0 <= index < pkt_bytelen:  # 确保索引合法
                                features[index] = value  # 按索引填充值
                    images_tensor.append(features)
                images_tensor = torch.tensor(images_tensor).long().to(device)

                # 创建伪数据字典
                data_dict = {key: images_tensor[:, i * packet_dim:(i + 1) * packet_dim].to(device).long()
                             for i, key in enumerate(sample_data.keys())}

                # 调用模型并获取预测
                dummy_y = torch.zeros(len(batch_texts), dtype=torch.long, device=device)
                with torch.no_grad():
                    outputs = model(data_dict, dummy_y, 1, device)
                    probs = torch.nn.functional.softmax(outputs[1], dim=1).cpu().detach().numpy()

                results.append(probs)

            # 合并所有批次的结果
            return np.concatenate(results, axis=0)

        class_names = [f"Class {i}" for i in range(classes)]  # 生成 ["Class 0", "Class 1", ...]
        # print(f'class_names len(): {len(class_names)}')

        # 创建 LimeTextExplainer
        explainer = LimeTextExplainer(class_names=class_names)

        # 生成 LIME 解释
        explanation = explainer.explain_instance(
            text_instance=src_text,  # 输入样本数据（文本格式）
            classifier_fn=predict_fn,  # 自定义预测函数
            num_features=pkt_bytelen  # 显示的重要特征数
        )

        # 打印解释结果
        # print(f"LIME explanation for sample {sample_idx}:")
        # print(f'ture_Y : {sample_target}')

        if sample_target in explanation.available_labels():
            res = explanation.as_list(label=sample_target)
            res.sort(key=lambda t: t[1], reverse=True)
            # print(f"Explanation for true label {sample_target}:\n", res)

            # 提取字段位置和重要性值
            positions = [int(field.split('a')[1]) for field, _ in res]  # 提取字段位置
            importances = [importance for _, importance in res]  # 提取重要性值

            # 创建空数组并填充重要性值
            max_position = max(positions) + 1  # 确定最大字段索引
            importance_array = np.zeros((1, max_position))  # 一维数组 (1 行, max_position 列)
            max_value = max(importances)
            importances_max.append(max_value)
            importances_min.append(min(importances))

            for pos, imp in zip(positions, importances):
                importance_array[0, pos] = max(0, imp)  # 将重要性小于 0 的值置为 0
            max_index = np.argmax(importance_array)  # 注意：这是在展平后的数组中的索引

            print(f'索引值：{explainer_num}， {datasets_index}_importances.max(): {max_value}, 位置：{max_index},  importances.min(): {min(importances)}')
            # print(f'importance_array: {importance_array}')

            # # 按照整个流划分， 绘制热力图
            # sns.heatmap(importance_array, cmap="Blues", annot=False, cbar=True,
            #             xticklabels=False, yticklabels=False, cbar_kws={"ticks": []},  # 移除颜色条的刻度
            #             )
            #
            # # 添加标题和标签
            # plt.title(f"Feature Importance Heatmap {sample_target}_{explainer_num}")
            # plt.xlabel("Field Position (Index)")
            # plt.ylabel("Importance")
            # explainer_num += 1
            #
            # plt.xticks([])  # 移除 x 轴刻度
            # # 显示图表
            # plt.tight_layout()
            # output_path = os.path.join(output_folder, f"Lime_{sample_target}_{explainer_num}.pdf")
            # plt.savefig(output_path)
            # plt.show()



            # 按 packet_dim 划分数据
            num_packets = max_position // packet_dim
            packet_arrays = [
                importance_array[:, i * packet_dim:(i + 1) * packet_dim]
                for i in range(num_packets)
            ]

            # custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_blues", ["white", "#073777"])
            if datasets_index == 'Payload':
                # print('Payload true')
                norm = plt.Normalize(vmin=0, vmax=0.6, clip=True)     # 数据实际范围
                colorbar_norm = plt.Normalize(vmin=0, vmax=0.14)  # 公共标准范围
            else:
                norm = plt.Normalize(importance_array.min(), importance_array.max())
                colorbar_norm = norm

            # 创建子图，堆叠排列
            fig, axes = plt.subplots(num_packets, 1, figsize=(5,  num_packets), sharex=True, gridspec_kw={"hspace": 0.5})

            for i, ax in enumerate(axes):
                sns.heatmap(
                    packet_arrays[i],
                    cmap="Blues",
                    norm=colorbar_norm,
                    annot=False,
                    cbar=False,  # 禁用单独的颜色条
                    xticklabels=False,
                    yticklabels=False,  # 禁用 y 轴刻度
                    ax=ax
                )
                ax.set_ylabel(f"Packet {i + 1}", rotation=0, labelpad=20, fontsize=8)
                ax.yaxis.set_label_position("left")  # Y 轴标签在左侧
                ax.tick_params(axis='x', labelsize=8)  # 调整 x 轴刻度字体大小


            # 添加共享颜色条
            cbar_ax = fig.add_axes([0.91, 0.16, 0.015, 0.7])  # 颜色条位置 [left, bottom, width, height]
            # norm = plt.Normalize(importance_array.min(), importance_array.max())


            # sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.Blues)
            sm = plt.cm.ScalarMappable(norm=colorbar_norm, cmap='Blues')

            sm.set_array([])
            fig.colorbar(sm, cax=cbar_ax, orientation='vertical', ticks=[])  # 全局颜色条，无刻度,  orientation='vertical'： 颜色条是竖直方向的（默认）。
            # fig.colorbar(sm, cax=cbar_ax, orientation='vertical')  # 有刻度

            # 添加整体标题
            plt.suptitle(f"Value Importance Heatmap", fontsize=16)
            explainer_num += 1
            # 调整布局
            plt.tight_layout(rect=[0, 0, 0.8, 0.95])  # 为颜色条腾出空间
            output_path = os.path.join(output_folder, f"Lime_{datasets_index}_{sample_target}_{explainer_num}.pdf")
            plt.savefig(output_path)
            plt.show()



    print(f'最后的max：{max(importances_max)}， min: {min(importances_min)}')












    # # uncertain_all = uncertain_all.cpu().numpy()
    # y_predict = y_predict.cpu().numpy()
    # y_true = y_true.cpu().numpy()
    # # y_true_one_h = y_true_one_h.cpu().numpy()
    # # pre_num= pre_num.cpu() / data_num
    #
    # matrix = confusion_matrix(y_true, y_predict)
    # # recall = recall_score(target,pre_cla)
    # # precision = precision_score(target,pre_cla)
    # # f1 = f1_score(target,pre_cla)
    #
    # # # 画uncertainty曲线
    # # plot_uncertainty(uncertain_all)
    # # # # 存uncertainty值
    # # save_uncertainty(uncertain_all.numpy(), byte_num, packet_num, k_index, u_index, is_open)
    #
    # # # 画已知和未知的roc图
    # # plot_roc(probability_value, y_true, classes, uncertain_all)
    #
    # # # # 只画未知图
    # # u_k_plot_roc(uncertain_all, y_true, classes, u_index, k_index)
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # precision, recall, fmeasure, support = precision_recall_fscore_support(y_true, y_predict)
    # # 计算宏平均的精度、召回率和F1值
    # macro_precision = np.mean(precision)
    # macro_recall = np.mean(recall)
    # macro_fmeasure = np.mean(fmeasure)
    #
    # print('-------recall: {}'.format(recall))
    # print('-------precision: {}'.format(precision))
    # print('-------fmeasure: {}'.format(fmeasure))
    # print('-------macro recall: {:.5f}'.format(macro_recall))
    # print('-------macro precision: {:.5f}'.format(macro_precision))
    # print('-------macro fmeasure: {:.5f}'.format(macro_fmeasure))
    #
    #
    # # # 微平均（Micro-average）多数类的影响更大
    # # weighted_precision = np.average(precision, weights=support)
    # # weighted_recall = np.average(recall, weights=support)
    # # weighted_fmeasure = np.average(fmeasure, weights=support)
    # # print('-------weighted recall: {:.5f}'.format(weighted_recall))
    # # print('-------weighted precision: {:.5f}'.format(weighted_precision))
    # # print('-------weighted fmeasure: {:.5f}'.format(weighted_fmeasure))
    #
    #
    # # #####与上面的计算宏平均类似
    # # recall = list()
    # # precision = list()
    # # fmeasure = list()
    # # for i in range(matrix.shape[0]):
    # #     if matrix[i, i] == 0:
    # #         recalli = 0
    # #         precisioni = 0
    # #         fmeasurei = 0
    # #     else:
    # #         recalli = matrix[i, i] / matrix[i, :].sum()
    # #         # precisioni = (matrix[:, i].sum()-matrix[i,i])/(y_true.shape[0] - matrix[i,:].sum())
    # #         precisioni = matrix[i, i] / matrix[:, i].sum()
    # #         fmeasurei = 2 * precisioni * recalli / (precisioni + recalli)
    # #     recall.append(recalli)
    # #     precision.append(precisioni)
    # #     fmeasure.append(fmeasurei)
    # # print('-------recall: {}'.format(recall))
    # # print('-------precision: {}'.format(precision))
    # # print('-------fmeasure: {}'.format(fmeasure))
    # # print('-------recall.mean(): {:.5f}'.format(np.array(recall).mean()))
    # # print('-------precision.mean(): {:.5f}'.format(np.array(precision).mean()))
    # # print('-------fmeasure.mean(): {:.5f}'.format(np.array(fmeasure).mean()))
    #
    #
    # # AUC
    # # auc = multi_label_auc(y_true_one_h, probability_value)
    # # print('-------auc: {:.5f}'.format(auc))
    # # # print('-------auc_ovo: {}'.format(auc_ovo))
    #
    #
    # """
    # 用于计算TPR, FPR
    # """
    # # TPR = list()
    # # FPR = list()
    # # PPV = list()
    # # ACC = list()
    # # for i in range(matrix.shape[0]):
    # #     if matrix[i, i] == 0:
    # #         recalli = 0
    # #         precisioni = 0
    # #         fmeasurei = 0
    # #     else:
    # #         FP = matrix.sum(axis=0) - np.diag(matrix)
    # #         FN = matrix.sum(axis=1) - np.diag(matrix)
    # #         TP = np.diag(matrix)
    # #         TN = matrix.sum() - (FP + FN + TP)
    # #
    # #         FP = FP.astype(float)
    # #         FN = FN.astype(float)
    # #         TP = TP.astype(float)
    # #         TN = TN.astype(float)
    # #
    # #         TPR = TP / (TP + FN)
    # #         FPR = FP / (FP + TN)
    # #         PPV = TP / (TP + FP)
    # #         # ACC = (TP + TN) / (TP + FP + FN + TN)  #  ACC值不对
    # # # print('-------TPR: {}'.format(TPR))
    # # # print('-------FPR: {}'.format(FPR))
    # # # print('-------PPV: {}'.format(PPV))
    # # # print('-------ACC: {}'.format(ACC))
    # # print('-------TPR.mean(): {:.5f}'.format(np.array(TPR).mean()))
    # # print('-------FPR.mean(): {:.5f}'.format(np.array(FPR).mean()))
    # # print('-------PPV.mean(): {:.5f}'.format(np.array(PPV).mean()))
    # # # print('-------ACC.mean(): {:.5f}'.format(np.array(ACC).mean()))
    #
    #
    # print(matrix)
    #
    #
    # return test_loss.item() / (batch_idx + 1), accu_num.item() / data_num