# coding:utf-8
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path_data = 'E:/学习/研究生1/创新竞赛/2018年算法赛/data/xgboost/train_featured_人群分类.csv'
train_full = pd.read_csv(path_data)
train_full_original = train_full.drop(train_full[(train_full['location'] != 1)].index).reset_index(drop=True)
sns.boxplot(x='hour', y="registered", data=train_full_original)
plt.show()

for i in range(1, 34):
    for j in range(24):
        data = train_full.drop(train_full[(train_full['location'] != i) | (train_full['hour'] != j)].index)  # .reset_index(drop=True)
        # print(data.index)
        # print(data['registered'])
        q_point = data['registered'].quantile([0.25, 0.75]).values  # 找到上下四分位点
        up = q_point[1] + 1.5 * (q_point[1] - q_point[0])
        down = q_point[0] - 1.5 * (q_point[1] - q_point[0])
        print('第', i, '个地点', '第', j, '小时', up, down, q_point[0], q_point[1])
        print(data[(data['registered'] < down) | (data['registered'] > up)].index)
        train_full = train_full.drop(data[(data['registered'] < down) | (data['registered'] > up)].index)
        print(len(train_full['registered']))

for i in range(1, 34):
    for j in range(24):
        data = train_full.drop(train_full[(train_full['location'] != i) | (train_full['hour'] != j)].index)  # .reset_index(drop=True)
        # print(data.index)
        # print(data['registered'])
        q_point = data['casual'].quantile([0.25, 0.75]).values  # 找到上下四分位点
        up = q_point[1] + 1.5 * (q_point[1] - q_point[0])
        down = q_point[0] - 1.5 * (q_point[1] - q_point[0])
        print('第', i, '个地点', '第', j, '小时', up, down, q_point[0], q_point[1])
        print(data[(data['casual'] < down) | (data['casual'] > up)].index)
        train_full = train_full.drop(data[(data['casual'] < down) | (data['casual'] > up)].index)
        print(len(train_full['casual']))

data1 = train_full.drop(train_full[(train_full['location'] != 1)].index).reset_index(drop=True)
sns.boxplot(x='hour', y="registered", data=data1)
plt.show()


