# coding:utf-8

import numpy as np
import pandas as pd
import datetime

# 读取数据
# path_train = 'E:/学习/研究生1/创新竞赛/2018年算法赛/data/xgboost/train_9_10不完整时间人流量.csv'
path_test = 'E:/学习/研究生1/创新竞赛/2018年算法赛/data/xgboost/test完整时间序列.csv'
# X_train_original = pd.read_csv(path_train)
X_test_original = pd.read_csv(path_test)

path_train = 'E:/学习/研究生1/创新竞赛/2018年算法赛/data/xgboost/train_9_10完整时间人流量.csv'
train = pd.read_csv(path_train)
print(train.head(3))

train['month'] = train['time'].str.split('/', expand=True)[1]
train['day2'] = train['time'].str.split('/', expand=True)[2]
train['day'] = train['day2'].str.split(' ', expand=True)[0]
train['day'] = train['day'] .astype(int)
train['hour2'] = train['day2'].str.split(' ', expand=True)[1]
train['hour'] = train['hour2'].str.split(':', expand=True)[0]
train.pop('day2')
train.pop('hour2')
temp_location = train['location']
train.pop('location')
train.insert(5, 'location', temp_location)
temp_flow = train['flow']
train.pop('flow')
train.insert(5, 'flow', temp_flow)
train.pop('time')
print(train.head(3))
X_train_original = train

# 先处理test数据集，和train保持一致，然后两个模型一起训练
X_test_original['month'] = X_test_original['time'].str.split('/', expand=True)[1]
X_test_original['day2'] = X_test_original['time'].str.split('/', expand=True)[2]
X_test_original['day'] = X_test_original['day2'].str.split(' ', expand=True)[0]
X_test_original['day'] = X_test_original['day'] .astype(int)
X_test_original['hour2'] = X_test_original['day2'].str.split(' ', expand=True)[1]
X_test_original['hour'] = X_test_original['hour2'].str.split(':', expand=True)[0]
X_test_original.pop('day2')
X_test_original.pop('hour2')
temp_location = X_test_original['location']
X_test_original.pop('location')
X_test_original.insert(4, 'location', temp_location)
X_test_original.pop('time')
print(X_train_original.head(3))
print(X_test_original.head(3))


# test整理完毕，开始一起添加特征
y_train_original = X_train_original.pop('flow')
X_train_original.columns = ['month', 'day', 'hour', 'location']
X_test_original.columns = ['month', 'day', 'hour', 'location']
print(len(X_train_original), len(X_test_original))
train_test = pd.concat([X_train_original, X_test_original], axis=0).reset_index(drop=True)  # 整合训练集和测试集，一起变更特征。
print(len(train_test))
print(train_test.head(3))  # 顺序保持不变

# 添加weekday信息
weekday = []
print(len(train_test['month']))
for i in range(len(train_test['month'])):
    if i % 1000 == 0:
        print(i)
    date = datetime.date(2017, int(train_test.iloc[i, 0]), int(train_test.iloc[i, 1]))
    # print(date, type(date))
    weekday.append(date.weekday()+1)
train_test['weekday'] = weekday
print(train_test.head(3))

# 添加hour分类信息
train_test_hour_cat = train_test['hour']  # 在保留hour数值特征基础上，新增一个分类特征
train_test['hour_cat'] = train_test_hour_cat

# 添加地点分类信息
shitang = [12, 8, 29, 10]
jiaoxuelou = [24, 27, 33, 16, 22, 14, 15]
big_loc = pd.DataFrame(np.arange(1, 34), columns=['location'])  # 小数据匹配，大数据merge，速度更快
big_loc['big_loc'] = 'sushe'

for i in range(len(big_loc)):
    print(i)
    location = int(big_loc.iloc[i, 0])
    if location in shitang:
        big_loc.iloc[i, 1] = 'shitang'
    elif location in jiaoxuelou:
        big_loc.iloc[i, 1] = 'jiaoxuelou'
print(big_loc)
train_test = pd.merge(train_test, big_loc, how='left', on='location')
print(train_test.head(3))

# 添加规则法中的flow信息
path_rule = 'E:/学习/研究生1/创新竞赛/2018年算法赛/data/xgboost/规则法结论.csv'
flow = pd.read_csv(path_rule)
train_test['LWH'] = train_test['location'].map(str) + '-' + train_test['weekday'].map(str) + '-' + train_test['hour'].map(str)
flow['LWH'] = flow['location'].map(str) + '-' + flow['weekday'].map(str) + '-' + flow['hour'].map(str)
flow.pop('location')
flow.pop('weekday')
flow.pop('hour')
train_test = pd.merge(train_test, flow, how='left', on='LWH')
train_test.pop('LWH')

# 添加学长规则法的flow2信息
path_rule = 'E:/学习/研究生1/创新竞赛/2018年算法赛/data/xgboost/规则法结论_小时.csv'
flow2 = pd.read_csv(path_rule)
train_test['LWH'] = train_test['location'].map(str) + '-' + train_test['weekday'].map(str) + '-' + train_test['hour'].map(str)
flow2['LWH'] = flow2['location'].map(str) + '-' + flow2['weekday'].map(str) + '-' + flow2['hour'].map(str)
flow2.pop('location')
flow2.pop('weekday')
flow2.pop('hour')
train_test = pd.merge(train_test, flow2, how='left', on='LWH')
train_test.pop('LWH')

# 添加地点的flow3
path_rule = 'E:/学习/研究生1/创新竞赛/2018年算法赛/data/xgboost/规则法结论_地点.csv'
flow3 = pd.read_csv(path_rule)
train_test['LWH'] = train_test['location'].map(str) + '-' + train_test['weekday'].map(str) + '-' + train_test['hour'].map(str)
flow3['LWH'] = flow3['location'].map(str) + '-' + flow3['weekday'].map(str) + '-' + flow3['hour'].map(str)
flow3.pop('location')
flow3.pop('weekday')
flow3.pop('hour')
train_test = pd.merge(train_test, flow3, how='left', on='LWH')
train_test.pop('LWH')

# 添加天气信息
weather = pd.read_csv('E:/学习/研究生1/创新竞赛/2018年算法赛/data/xgboost/weather_all.csv')
weather.columns = ['date', 'weather1', 'weather2', 'max_temp', 'min_temp', 'wind1', 'wind2', 'air',  'AQI']
weather['month'] = weather['date'].str.split('/', expand=True)[1]
weather['day'] = weather['date'].str.split('/', expand=True)[2]
weather['M+D'] = weather['month'].map(str) + '-' + weather['day'].map(str)
weather.pop('month')
weather.pop('day')
weather.pop('date')
train_test['M+D'] = train_test['month'].map(str) + '-' + train_test['day'].map(str)
train_test = pd.merge(train_test, weather, how='left', on='M+D')
train_test.pop('M+D')
# print(train_test.head(3))

# 增加working
working = [1, 2, 3, 4, 5]
notworking = [6, 7]
work = pd.DataFrame(np.arange(1, 8), columns=['weekday'])
work['working'] = 'yes'
for i in range(len(work['weekday'])):
    print(i)
    weekday = int(big_loc.iloc[i, 0])
    if weekday in working:
        work.iloc[i, 1] = 'yes'
    else:
        work.iloc[i, 1] = 'no'
print(work)
train_test = pd.merge(train_test, work, how='left', on='weekday')
# print(train_test.head(7))


# 分割训练集和测试集合
train = train_test.iloc[0:33264, :]
test = train_test.iloc[33264:, :]
train['real_flow'] = y_train_original
print(train.head(3))
print(test.head(3))
print(len(train), len(test))
train.to_csv('E:/学习/研究生1/创新竞赛/2018年算法赛/data/xgboost/train_featured.csv', index=False)
test.to_csv('E:/学习/研究生1/创新竞赛/2018年算法赛/data/xgboost/test_featured.csv', index=False)
