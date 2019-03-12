# coding:utf-8

# 主要实现了，原始数据 匹配了 用户标签，并且按照Time，Location，Cat进行了流量统计，到train_count文件下
# train_count文件还需要手动处理，分成两个文件train_regis和train_casual

import datetime
import pandas as pd
path_train = 'E:/学习/研究生1/创新竞赛/2018年算法赛/data/train.csv'
path_freq = 'E:/学习/研究生1/创新竞赛/2018年算法赛/data/train_frequency.csv'
train = pd.read_csv(path_train)
user_cat = pd.read_csv(path_freq)
user_cat.drop("freq", axis=1, inplace=True)
train_combine = pd.merge(train, user_cat, how='left')  # 整合casual和在校生用户标签
train_combine['TLC'] = train_combine['time'].map(str) + ',' + train_combine['location'].map(str) + ',' + train_combine['cat'].map(str)  # 进行字符串合并

train_combine.info()
print(train_combine.head(5))
train_count = train_combine['TLC'].value_counts()
print(len(train_count))
print(train_count.head(5))
train_count.to_csv('E:/学习/研究生1/创新竞赛/2018年算法赛/data/train_count.csv', index=True)

# 需要手动excel分开casual和regis为两个文件
'''
week = []
for i in range(len(train_combine['userID'])):
    if i % 1000 == 0:
        print(i)
    month = train_combine['month'][i]
    day = train_combine['day'][i]
    week.append(datetime.date(2017, month, day).isocalendar()[2])
train_combine['weekday'] = week
'''