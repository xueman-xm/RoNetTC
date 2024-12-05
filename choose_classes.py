"""
data: 2024/1/12-16:23
"""
from collections import Counter



def read_text_file(file_path):
    try:
        data = []
        target = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for item_line in file.readlines():
                split_line = [item.strip() for item in item_line.split('\t')]
                data.extend([split_line])
                target.append(int(split_line[0]))
            return data, target
    except FileNotFoundError:
        print(f"文件 '{file_path}' 不存在。")
    except Exception as e:
        print(f"发生错误：{e}")

# # 用法示例
file_path = './datasets/mapps_work_01_256.txt'  # 将文件路径替换为实际的文件路径
text_content, label = read_text_file(file_path)
counter_result = dict(Counter(label))
filtered_counter_result = {key: value for key, value in counter_result.items() if value >= 1500}    ###由于是三个view，大于500，所以是1500
keys_list = sorted(filtered_counter_result.keys())
key_500_dict = {item: i for i, item in enumerate(keys_list)}
print('len(key_500_dict): {}, key_500_dict: {}'.format(len(key_500_dict), key_500_dict))
new_label = []
new_data = []

# for item in text_content:
#     con = int(item[0])
#     if con in key_500_dict:
#         new_cla = key_500_dict[con]
#         if new_label.count(new_cla) < 6000:    # 注意这里是小于6000，而不是小于等于6000
#             new_label.append(new_cla)
#             item[0] = new_cla
#             new_data.append(item)
count_dict = {}
for item in text_content:
    con = int(item[0])
    if con in key_500_dict:
        new_cla = key_500_dict[con]
        if count_dict.get(new_cla, 0) < 6000:    # 检查计数
            new_label.append(new_cla)
            item[0] = new_cla
            new_data.append(item)
            count_dict[new_cla] = count_dict.get(new_cla, 0) + 1  # 更新计

print('1111')
counter_result_new = dict(Counter(new_label))
print('len(counter_result_new): {} ,counter_result_new: {}'.format(len(counter_result_new), counter_result_new))
with open('./datasets/Mapps_256.txt', 'w', encoding='utf-8') as new_file:
    for item in new_data:
        new_file.write('\t'.join(map(str, item)) + '\n')





# path = './datasets/Mapps_256.txt'
# deci_result = []
# with open(path, 'r', encoding='utf-8') as f:
#     # for item_line in f.readlines():
#     #     split_line = [item.strip() for item in item_line.split('\t')]
#     #     for item_ele in split_line:
#     #         if item_ele.startswith('0x'):
#     #             try:
#     #                 ele_lis = [int(item, 16) for item in item_ele.split()]
#     #             except ValueError:
#     #                 print(f"Warning: Unable to convert '{item_ele}' to a valid hexadecimal number.'{n}'")
#     #             deci_result.append(ele_lis)
#     #             # ele_lis=[ ]
#     #     print('111')
#     split_result=[]
#     n =0
#     for item_line in f.readlines():
#         split_line = [item.strip() for item in item_line.split('\t')]
#         for item_ele in split_line:
#             if item_ele.startswith('0x'):
#                 try:
#                     ele_lis = [int(item, 16) for item in item_ele.split()]
#                 except ValueError:
#                     print(f"Warning: Unable to convert '{item_ele}' to a valid hexadecimal number.'{n}'")
#                 deci_result.append(ele_lis)
#                 # ele_lis=[ ]
#         n += 1
#         if len(deci_result) < 16:
#             deci_result.extend([[0] * len(deci_result[0]) for _ in range(16 - len(deci_result))])
#         else:
#             deci_result = deci_result[:16]
#         if split_line[1] in 'client_payload':
#             deci_result = [sublist[:(256 - 20 - 32)] for sublist in deci_result]
#
#         split_line[2:] = deci_result
#         deci_result = []
#         split_result.append(split_line)
# print('---------read data-01')
#
# target = []
# ip_header_data = []
# tcp_header_data = []
# clie_payload_data = []
# for ind, item in enumerate(split_result):
#     if ind % 3 == 0:
#         target.append(int(item[0]))
#         ip_header_data.append(item[2:])
#     elif ind % 3 == 1:
#         tcp_header_data.append(item[2:])
#     else:
#         clie_payload_data.append(item[2:])
#
# counter_result = dict(Counter(target))
# print('counter_result: ', counter_result)
