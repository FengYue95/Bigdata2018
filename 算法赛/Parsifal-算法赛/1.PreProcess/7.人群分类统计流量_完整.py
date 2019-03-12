# coding:utf-8
# 整合完整时间，补全数据

import pandas as pd

# 读取数据
path_registered = 'E:/学习/研究生1/创新竞赛/2018年算法赛/data/rules/registered_count9_10_不完整.csv'
path_casual = 'E:/学习/研究生1/创新竞赛/2018年算法赛/data/rules/casual_count9_10_不完整.csv'
path_time = 'E:/学习/研究生1/创新竞赛/2018年算法赛/data/rules/train_9_10完整时间序列.csv'
train_registered = pd.read_csv(path_registered)
train_casual = pd.read_csv(path_casual)
train_time = pd.read_csv(path_time)
train_registered.columns = ['month', 'day', 'hour', 'location', 'registered']
train_casual.columns = ['month', 'day', 'hour', 'location', 'casual']
train_registered = train_registered.drop(train_registered[(train_registered['month'] == 9) & (train_registered['day'] < 13)].index).reset_index(drop=True)
train_registered = train_registered.drop(train_registered[(train_registered['month'] == 10) & (train_registered['day'] < 9) & (train_registered['day'] > 1)].index).reset_index(drop=True)
train_casual = train_casual.drop(train_casual[(train_casual['month'] == 9) & (train_casual['day'] < 13)].index).reset_index(drop=True)
train_casual = train_casual.drop(train_casual[(train_casual['month'] == 10) & (train_casual['day'] < 9) & (train_casual['day'] > 1)].index).reset_index(drop=True)

print(train_registered.head(3))
print(train_casual.head(3))
print(train_time.head(3))
train_time.columns = ['time', 'location']
train_registered['month'] = train_registered['month'].astype(str)
train_registered['day'] = train_registered['day'].astype(str)
train_registered['hour'] = train_registered['hour'].astype(str)
train_registered['location'] = train_registered['location'].astype(str)

train_casual['month'] = train_casual['month'].astype(str)
train_casual['day'] = train_casual['day'].astype(str)
train_casual['hour'] = train_casual['hour'].astype(str)
train_casual['location'] = train_casual['location'].astype(str)
train_time['time'] = train_time['time'].astype(str)
train_time['location'] = train_time['location'].astype(str)


# 整合数据registered
train_registered['time'] = '2017/'+train_registered['month']+'/'+train_registered['day']+' '+train_registered['hour']+':00'
print(train_registered.head(3))
train_registered['TL'] = train_registered['time'] + train_registered['location']
train_time['TL'] = train_time['time'] + train_time['location']
train_full_registered = pd.merge(train_time, train_registered, how='left', on='TL')
train_full_registered.info()

train_full_registered['registered'] = train_full_registered['registered'].fillna(0)
train_full_registered.pop('month')
train_full_registered.pop('day')
train_full_registered.pop('hour')
train_full_registered.pop('location_y')
train_full_registered.pop('time_y')
train_full_registered.pop('TL')
train_full_registered.columns = ['time', 'location', 'registered']


# 整合数据casual
train_casual['time'] = '2017/'+train_casual['month']+'/'+train_casual['day']+' '+train_casual['hour']+':00'
print(train_casual.head(3))
train_casual['TL'] = train_casual['time'] + train_casual['location']
# train_time['TL'] = train_time['time'] + train_time['location']
train_full_casual = pd.merge(train_time, train_casual, how='left', on='TL')
train_full_casual.info()

train_full_casual['casual'] = train_full_casual['casual'].fillna(0)
train_full_casual.pop('month')
train_full_casual.pop('day')
train_full_casual.pop('hour')
train_full_casual.pop('location_y')
train_full_casual.pop('time_y')
train_full_casual.pop('TL')
train_full_casual.columns = ['time', 'location', 'casual']

# 整合registered和casual
train_full = pd.merge(train_full_registered, train_full_casual, how='inner', on=['time', 'location'])
print(len(train_full['time']), len(train_full_registered['time']), len(train_full_casual['time']))
train_full.to_csv('E:/学习/研究生1/创新竞赛/2018年算法赛/data/rules/train_9_10完整时间人流量_count_registered.csv', index=False)