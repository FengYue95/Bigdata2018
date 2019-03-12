# coding:utf-8

import pandas as pd
import datetime
'''
train_09 = pd.read_csv('E:/学习/研究生1/创新竞赛/2018年算法赛/data/months1_10/10.csv')
train_10 = pd.read_csv('E:/学习/研究生1/创新竞赛/2018年算法赛/data/months1_10/11.csv')
# print(train_full.head(3))
# train_full = train_full.iloc[0:10000, :]
train_full = pd.concat([train_09, train_10], axis=0)
# train_full['date'] = train_full['time'].str.split(' ', expand=True)[0]
# print(len(train_09['time']))
# print(len(train_10['time']))
print(len(train_full['time']))
month = []
day = []
for i in range(len(train_full['time'])):
    if i % 10000 == 0:
        print(i)
    date = datetime.datetime.strptime(train_full.iloc[i, 1].split(' ')[0], '%Y-%m-%d')
    month.append(date.month)
    day.append(date.day)
train_full['month'] = month
train_full['day'] = day
train_full.columns = ['userID', 'time', 'location', 'month', 'day']
train_full.pop('time')
print(train_full.head(3))
train_full = train_full.drop(train_full[(train_full['day'] < 11) & (train_full['month'] == 10)].index).reset_index(drop=True)
print('drop process finished')
train_full.to_csv('E:/学习/研究生1/创新竞赛/2018年算法赛/data/xgboost/联立10-11月所有人访问记录-时间.csv', index=False)
'''
# 每个人一个地点一天只保留一条记录
# print(len(train_full))
# train_full['ULMD'] = train_full['userID'].map(str)+'-'+train_full['location'].map(str)+'-'+train_full['month'].map(str)+'-'+train_full['day'].map(str)
# train_full = train_full.drop_duplicates(subset='ULMD', keep='first', inplace=False)
# print(len(train_full))

train_full = pd.read_csv('E:/学习/研究生1/创新竞赛/2018年算法赛/data/xgboost/联立10-11月所有人访问记录-时间.csv')
train_full['user_location'] = train_full['userID'].map(str)+'-'+train_full['location'].map(str)
print('数据读取完毕')
location_count = train_full['user_location'].value_counts()
print('value_count完毕')
print(location_count.head(3))
location_count.to_csv('E:/学习/研究生1/创新竞赛/2018年算法赛/data/xgboost/常驻人口统计_用户地点_出现次数10-11.csv', index=True, encoding='utf-8')

