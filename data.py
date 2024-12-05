import numpy as np
import pylab as p
import scipy.io as sio
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Pool
import torch
import os
import copy
import csv
from PIL import Image
import random
from collections import Counter


from torchvision import transforms


class Splite_view_data():
    """
    load multi-view data
    """

    def __init__(self, root, is_open, unknown_path,  classes, byte_num, packet_num):
        """
        :param root: data name and path
        """
        super(Splite_view_data, self).__init__()
        self.root = root
        self.classes = classes
        self.byte_num = byte_num
        self.packet_num = packet_num
        self.unknown_path = unknown_path
        self.is_open = is_open


    def read_data(self, path):
        assert os.path.exists(path), '{} 文件不存在'.format(path)
        split_result = []
        deci_result = []
        ele_lis = []
        n = 0
        with open(path, 'r', encoding='utf-8') as f:
            for item_line in f.readlines():
                split_line = [item.strip() for item in item_line.split('\t')]
                for item_ele in split_line:
                    if item_ele.startswith('0x'):
                        try:
                            ele_lis = [int(item, 16) for item in item_ele.split()]
                        except ValueError:
                            print(f"Warning: Unable to convert '{item_ele}' to a valid hexadecimal number.'{n}'")
                        deci_result.append(ele_lis)
                        # ele_lis=[ ]
                n += 1
                if len(deci_result) < self.packet_num:
                    deci_result.extend([[0] * len(deci_result[0]) for _ in range(self.packet_num - len(deci_result))])
                else:
                    deci_result = deci_result[:self.packet_num]
                if split_line[1] in 'client_payload':
                    deci_result = [sublist[:(self.byte_num - 20 - 32)] for sublist in deci_result]

                split_line[2:] = deci_result
                deci_result = []
                split_result.append(split_line)
        print('---------read data-01')
        return split_result



    def split_mul_data(self):
        split_data = self.read_data(self.root)
        target = []
        ip_header_data = []
        tcp_header_data = []
        clie_payload_data = []
        for ind, item in enumerate(split_data):
            if ind % 3 == 0:
                target.append(int(item[0]))
                ip_header_data.append(item[2:])
            elif ind % 3 == 1:
                tcp_header_data.append(item[2:])
            else:
                clie_payload_data.append(item[2:])

        print('---------read data-02')
        if self.is_open:
            U_split_data = self.read_data(self.unknown_path)
            U_target = []
            U_ip_header_data = []
            U_tcp_header_data = []
            U_clie_payload_data = []
            for ind, item in enumerate(U_split_data):
                if ind % 3 == 0:
                    U_target.append(int(item[0]))
                    U_ip_header_data.append(item[2:])
                elif ind % 3 == 1:
                    U_tcp_header_data.append(item[2:])
                else:
                    U_clie_payload_data.append(item[2:])

            max_cla = max(target)
            U_target = [i + max_cla +1 for i in U_target]
            U_conter_result = dict(Counter(U_target))
            print('U_conter_result: ', U_conter_result)
        counter_result = dict(Counter(target))
        print('counter_result: ', counter_result)


        tra_data = []
        tra_label = []
        tes_data = []
        tes_label = []
        val_label = []
        val_data = []
        random.seed(2)

        keys_list = list(counter_result.keys())
        con = 0
        for key, value in counter_result.items():
            print('key: {}, {}'.format(key, value))
            each_num = value
            train_num = int(each_num * 0.6)
            val_num = int(each_num * 0.2)
            cla_tra_each = random.sample(range(each_num), train_num)

            cla_tes_val = list(set(range(each_num)) - set(cla_tra_each))

            cla_val_each = random.sample(cla_tes_val, val_num)
            cla_tes_each = list(set(cla_tes_val) - set(cla_val_each))

            tra_label.extend(target[i + con] for i in cla_tra_each)
            tra_data.extend([ip_header_data[i + con], tcp_header_data[i + con], clie_payload_data[i + con]] for i in cla_tra_each)

            tes_label.extend(target[i + con] for i in cla_tes_each)
            tes_data.extend([ip_header_data[i + con], tcp_header_data[i + con], clie_payload_data[i + con]] for i in cla_tes_each)

            val_label.extend(target[i + con] for i in cla_val_each)
            val_data.extend([ip_header_data[i + con], tcp_header_data[i + con], clie_payload_data[i + con]] for i in cla_val_each)



            con += int(value)

        if self.is_open:

            tes_label.extend([max_cla + 1] * len(U_ip_header_data))
            tes_data.extend([U_ip_header_data[i], U_tcp_header_data[i], U_clie_payload_data[i]] for i in range(len(U_ip_header_data)))



        print('----------read data-02')
        print('----------label: ', counter_result)
        print('----------tra_data: ', len(tra_data))
        print('----------tes_data: ', len(tes_data))
        print("----------val_data: ", len(val_data))
        return tra_data, tra_label, tes_data, tes_label, val_data, val_label


















class Mul_dataset(Dataset):
    def __init__(self, images_path: list, images_class: list, views :int):
        self.images_path = images_path
        self.images_class = images_class
        self.views = views

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = self.images_path[item]
        label = self.images_class[item]

        img_all = []
        for i in range(self.views):
            for p in range(len(img[i])):
                img_all.append(torch.tensor(img[i][p]))
        img = {i:img_all[i] for i in range(len(img_all))}

        return img, label

