# coding:utf-8
# 整合完整时间，补全数据

import pandas as pd

# 读取数据
path_train = 'E:/学习/研究生1/创新竞赛/2018年算法赛/data/rules/train_9_10不完整时间人流量.csv'
path_time = 'E:/学习/研究生1/创新竞赛/2018年算法赛/data/rules/train_9_10完整时间序列.csv'
train_loss = pd.read_csv(path_train)
train_time = pd.read_csv(path_time)
train_loss.columns = ['month', 'day', 'hour', 'location', 'flow']
train_loss = train_loss.drop(train_loss[(train_loss['month'] == 9) & (train_loss['day'] < 13)].index).reset_index(drop=True)
train_loss = train_loss.drop(train_loss[(train_loss['month'] == 10) & (train_loss['day'] < 9) & (train_loss['day'] > 1)].index).reset_index(drop=True)
print(train_loss.head(3))
print(train_time.head(3))
train_time.columns = ['time', 'location']
train_loss['month'] = train_loss['month'].astype(str)
train_loss['day'] = train_loss['day'].astype(str)
train_loss['hour'] = train_loss['hour'].astype(str)
train_loss['location'] = train_loss['location'].astype(str)
train_time['time'] = train_time['time'].astype(str)
train_time['location'] = train_time['location'].astype(str)


# 整合数据
train_loss['time'] = '2017/'+train_loss['month']+'/'+train_loss['day']+' '+train_loss['hour']+':00'
print(train_loss.head(3))
train_loss['TL'] = train_loss['time'] + train_loss['location']
train_time['TL'] = train_time['time'] + train_time['location']
train_full = pd.merge(train_time, train_loss, how='left', on='TL')
train_full.info()

train_full['flow'] = train_full['flow'].fillna(0)
train_full.pop('month')
train_full.pop('day')
train_full.pop('hour')
train_full.pop('location_y')
train_full.pop('time_y')
train_full.pop('TL')
train_full.columns = ['time', 'location', 'flow']
train_full.to_csv('E:/学习/研究生1/创新竞赛/2018年算法赛/data/rules/train_9_10完整时间人流量.csv', index=False)
