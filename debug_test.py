# # import pandas as pd
# # import os
# # df_raw = pd.read_csv(r'E:\Project\LTSF-Linear\dataset\RC.xlsx')
#
# import torch.nn as nn
# import torch
#
# b, s, c = 3, 4, 2
# a = nn.AvgPool1d(kernel_size=3, stride=1, padding=0)
# x = torch.arange(0, b*s*c, dtype=torch.float).view(b, s, c)
# front = x[:, 0:1, :].repeat(1, (3-1)//2, 1)
# end = x[:, -1:, :].repeat(1, (3-1)//2, 1)
# input = torch.cat([front, x, end], dim=1)
# trend = a(input.permute(0, 2, 1)).permute(0, 2, 1)
# season = x-trend
#
# season, trend = season.permute(0, 2, 1), trend.permute(0, 2, 1)
#
# # individual is true：
# Linear_Seasonal1 = nn.ModuleList()
# Linear_Trend1 = nn.ModuleList()
# for i in range(2):
#     Linear_Seasonal1.append(nn.Linear(4, 6))
#     Linear_Trend1.append(nn.Linear(4, 6))
# for model_list in [Linear_Seasonal1, Linear_Trend1]:
#     for layer in model_list:
#         nn.init.constant_(layer.weight, 1.0)
#         nn.init.constant_(layer.bias, 0)
#
# seasonal_output1 = torch.zeros([season.size(0), season.size(1), 6], dtype=season.dtype).to(season.device)
# trend_output1 = torch.zeros([trend.size(0), trend.size(1), 6], dtype=trend.dtype).to(trend.device)
# for i in range(2):
#     seasonal_output1[:, i, :] = Linear_Seasonal1[i](season[:, i, :])
#     trend_output1[:, i, :] = Linear_Trend1[i](trend[:, i, :])
#
# y1 = seasonal_output1 + trend_output1
#
# # individual is false：
# Linear_Seasonal2 = nn.Linear(4, 6)
# Linear_Trend2 = nn.Linear(4, 6)
# nn.init.constant_(Linear_Seasonal2.weight, 1.0)
# nn.init.constant_(Linear_Seasonal2.bias, 0)
# nn.init.constant_(Linear_Trend2.weight, 1.0)
# nn.init.constant_(Linear_Trend2.bias, 0)
# seasonal_output2 = Linear_Seasonal2(season)
# trend_output2 = Linear_Trend2(trend)
#
# y2 = seasonal_output2 + trend_output2
# print(x)
# print('____________')
# print(y1.permute(0, 2, 1))
# print('____________')
# print(y2.permute(0, 2, 1))
#


# # 实验结果保存到excel
# import pandas as pd
# from datetime import datetime
#
# path = r'E:\Project\LTSF-Linear\test_results\result.xlsx'
#
# df = pd.read_excel(path)
# cols_data = df.columns[1:]
# df_data = df[cols_data]
#
# # new_data = {'Data': 'Z', 'Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Model': 'Base', 'SeqLen': 12, 'PredLen': 336,
# #             'ID': 'someting mesg', 'Param': 'batch size:5',
# #             'MAE': 0.001, 'MSE': 0.005, 'RMSE': 0.004, 'MAPE': 0.005, 'MSPE': 0.004, 'RES': 0.007, 'CORR': 0.008}
#
# # df_data = df_data.append(new_data, ignore_index=True)
# # df_data = df_data.sort_values(by='Data', key=lambda x: x.str.lower())
# # df_data.insert(0, 'Index', range(1, len(df_data)+1))
#
# # df_data = df_data.sort_values(['Data', 'Model', 'SeqLen'], key=lambda x: x.str.lower())
# df_data = df_data.sort_values(['Data', 'Model', 'SeqLen', 'PredLen'])
# # add_index = lambda group: group.reset_index(drop=True).reset_index().rename(columns={'index': '序号'})
# # new_df = pd.concat(df_sorted.apply(add_index)).sort_index().reset_index(drop=True)
# df_data.insert(0, 'Index', range(1, len(df_data)+1))
# df_data.to_excel(path, index=False)


# # 测试.npy文件保存的数据格式
# import numpy as np
# b, s, c = 4, 3, 2
# a = np.arange(b*s*c).reshape(b, s, c)
# print(a.shape)
# print(a)
# b = a.reshape(-1, a.shape[-2], a.shape[-1])
# print('________')
# print(b)
# print(a.shape)
# print(b.shape)


# import matplotlib.pyplot as plt
# import numpy as np
# import time
#
# # 生成多张子图
# fig, axs = plt.subplots(2, 3, figsize=(10, 6))
# for i, ax in enumerate(axs.flatten()):
#     x = np.linspace(0, 2 * np.pi, 100)
#     y = np.sin((i + 1) * x)
#     ax.plot(x, y)
#     ax.set_title(f"Plot {i+1}")
#
# # 调整子图的布局
# plt.subplots_adjust(wspace=0.3, hspace=0.3)     # 设置横向间距和纵向间距
#
# # 将子图合并为一张大图
# fig.canvas.draw()       # 重绘图形
# w, h = fig.canvas.get_width_height()        # 获取当前Figure对象的画布（canvas）的宽度和高度
# # 将一个Figure对象的Canvas对象中的图像转换成RGB图像的二进制字符串表示
# merged_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
#
# # 显示合并后的大图
# fig = plt.figure()
# ax = fig.add_axes([0, 0, 1, 1])
# plt.imshow(merged_image)        # 默认会出现正在当前活动的子图上，所以会出现在右下角
# plt.axis('off')     # 关闭绘图时的坐标轴
# plt.show()

# import pandas as pd
# import os
# df_raw = pd.read_excel(r'E:/RC.xlsx')
# cols_data = df_raw.columns[1:]
# df_data = df_raw[cols_data]
# df_data = df_data.drop(df_data[df_data.values[:, 0] == 0].index).dropna()
# print(df_data[0:2])

# import torch
# x = torch.randn(3, 4)
#
# # 在第一维（索引位置 0）添加一个新的维度
# x_new = x.unsqueeze(0)
# print(x)
# print(x_new)

import numpy as np

# # 创建一个7*1的数组
# a = np.array([1, 2, 3, 4, 5, 6, 7]).reshape(7, 1)
# # 将数组沿着轴0进行扩展
# b = np.expand_dims(a, axis=0)
# # 将数组扩展为3维数组
# c = np.expand_dims(b, axis=2)
# # 将数组沿着第3个轴复制6次
# d = np.tile(c, reps=(1, 1, 7))
# # 输出结果
# print(d)

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# x = np.random.rand(34560, 7)
# scaler.fit(x)
#
# a = np.random.rand(106, 1)
# y = scaler.inverse_transform(np.tile(a, reps=(1, 7)))
# print(a.shape)
# print(y.shape)
# print(y[:, -1])
# print(a)
# print(b)


# import matplotlib.pyplot as plt
# import numpy as np
#
# # 生成数据
# x = np.linspace(0, 10, 100)
# y1 = np.sin(x)
# y2 = np.cos(x)
#
# # 绘制数据
# fig, ax = plt.subplots()
# ax.plot(x, y1, label='Sin')
# ax.plot(x, y2, label='Cos')
# ax.legend()
#
# # 计算MAE和MSE
# mae = np.mean(np.abs(y1 - y2))
# mse = np.mean((y1 - y2) ** 2)
#
# # 获取图例位置
# legend = ax.get_legend()
# bbox_to_anchor = legend.get_bbox_to_anchor().bounds
# # legend_loc = legend._loc
# size = fig.get_size_inches()
# print(bbox_to_anchor)
# print(size)
# print(ax.get_xlim(), ax.get_xlim())
#
# transFigureToAxes = fig.transFigure.inverted()
# boundsInAxes = transFigureToAxes.transform(bbox_to_anchor[0: 2])
# boundsInSubplot = ax.transAxes.inverted().transform(boundsInAxes)
# print(boundsInSubplot)
#
# # 在图例上方显示MAE和MSE
# ax.text(1, 1, f"MAE = {mae:.2f}, MSE = {mse:.2f}",
#         ha='center', va='bottom', transform=ax.transAxes)
#
# plt.show()

# import pandas as pd
# border1 = 0
# border2 = 12 * 30 * 24 * 4
# df_raw = pd.read_csv(r'E:\Project\LTSF-Linear\dataset\ETTm1.csv')
# df_stamp = df_raw[['date']][border1:border2]
# df_stamp['date'] = pd.to_datetime(df_stamp.date)
# df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 0)
# df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
# df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
# df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
# df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
# df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
# print('test')

import math
import torch
# b, h, l, d = 1, 8, 100, 512//8
# sample_k = int(math.log(100)) + 1
# n_top = sample_k
# q, k = torch.randn(b, h, l, d), torch.randn(b, h, l, d)
# k_expand = k.unsqueeze(-3).expand(b, h, l, l, d)
# index_sample = torch.randint(l, (l, sample_k))
# k_sample = k_expand[:, :, torch.arange(l).unsqueeze(1), index_sample, :]
# q_k_sample = torch.matmul(q.unsqueeze(-2), k_sample.transpose(-2, -1)).squeeze()
# print(f"q_k_sample.shape{q_k_sample.shape}")
#
# m = q_k_sample.max(-1)[0] - torch.div(q_k_sample.sum(-1), l)
# m_top = m.topk(n_top, sorted=False)[1]
# print(f"m_top_shape:{m_top.shape}\nm_shape:{m.shape}")
#
# b_test = torch.arange(b)[:, None, None, None]
# h_test = torch.arange(h)[None, :, None, None]
# print(f"b_test:{b_test.shape}\nh_test:{h_test.shape}")
# q_reduce = q[
#            b_test,
#            h_test,
#            m_top, :]
#
# q_test = q[:, :, ]
# print(f"q.shape:{q.shape}")
# print(f"q_reduce.shape:{q_reduce.shape}")
# q_k = torch.matmul(q_reduce, k.transpose(-2, -1))

# a = torch.arange(54).view(3, 3, 3, 2)
# print("------a--------")
# print(a)
# b = torch.arange(3)[:, None, None]
# c = torch.arange(3)[None, :, None]
# d = a[b, c, torch.randint(3, (3, 2))[None, :, :], :]
# print(f"a.shape:{a.shape}\n"
#       f"b.shape:{b.shape}\n"
#       # f"d_b.shape:{d_b.shape}\n"
#       f"c.shape:{c.shape}\n"
#       # f"d_c.shape:{d_c.shape}\n"
#       # f"d_b_c.shape:{d_b_c.shape}"
#       f"d.shape:{d.shape}\n")

import torch
import torch.nn as nn
a = torch.arange(0, 12).view(2, 6).float()
pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
b = pool(a)
print(a)
print(a.shape)
print(b)
print(b.shape)
