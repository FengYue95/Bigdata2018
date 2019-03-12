# coding:utf-8

# 本部分只负责统计9,10月份的registered和casual流量，不负责drop

import pandas as pd
import numpy as np

path_user_cat = 'E:/学习/研究生1/创新竞赛/2018年算法赛/data/xgboost/常驻人口统计_用户地点_出现次数9-10删除部分后.csv'
path_train_9 = "E:/学习/研究生1/创新竞赛/2018年算法赛/data/months1_10/09.csv"
path_train_10 = "E:/学习/研究生1/创新竞赛/2018年算法赛/data/months1_10/10.csv"
user_cat = pd.read_csv(path_user_cat)
print(user_cat.head(3))
train9 = pd.read_csv(path_train_9)
print(train9.head(3))
train10 = pd.read_csv(path_train_10)
print('开始联合两个数据集')
train = pd.concat([train9, train10], axis=0).reset_index(drop=True)
train['location'] = train['location'].astype(str)
# train['time'] = train['time'].astype(str).str.pad(6, side='left', fillchar='0')  # 将时间格式补全
print(len(train9), len(train10), len(train))
print('补全完成')

# 开始匹配两个用户ID号

train.columns = ['phone_id', 'time', 'location']
train['user_location'] = train['phone_id'].map(str)+'-'+train['location'].map(str)
print('原数据userID_location合并完成')
user_cat.columns = ['user_location',  'count']
user_cat['cat'] = 1
# user_cat['user_location'] = user_cat['phone_id'].map(str)+'-'+user_cat['location'].astype(int).map(str)
# user_cat.pop('phone_id')
# user_cat.pop('location')
user_cat.pop('count')
# user_cat.columns = ['user_location', 'cat']

train = pd.merge(train, user_cat, how='left', on=['user_location'])
train.pop('user_location')
print('匹配完成')
print(train.head(3))
train['cat'].fillna(0)
print('开始统计registered')
registered = train.drop(train[(train['cat'] != 1)].index).reset_index(drop=True)
registered['LT'] = registered['time'].map(str) + '/' + registered['location'].map(str)
# registered = registered.sort_values(by=['LT'])
registered_count = registered['LT'].value_counts()
registered_count.to_csv('E:/学习/研究生1/创新竞赛/2018年算法赛/data/rules/registered_count9_10.csv', index=True)
print('开始统计casual')
casual = train.drop(train[(train['cat'] == 1)].index).reset_index(drop=True)
casual['LT'] = casual['time'].map(str) + '/' + casual['location'].map(str)
# casual = casual.sort_values(by=['LT'])
casual_count = casual['LT'].value_counts()
casual_count.to_csv('E:/学习/研究生1/创新竞赛/2018年算法赛/data/rules/casual_count9_10.csv', index=True)

