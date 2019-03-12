# coding:utf-8
# 本部分完成与测试集时间的融合，生成最终结果

import pandas as pd
import numpy as np
import datetime

# 读取数据
path_test_half = 'E:/学习/研究生1/创新竞赛/2018年算法赛/data/rules/10_11规则一_8周.csv'
path_test_time = 'E:/学习/研究生1/创新竞赛/2018年算法赛/data/rules/12月test完整时间序列.csv'
test_half = pd.read_csv(path_test_half)
test_time = pd.read_csv(path_test_time)

# 添加共有时间信息，做key进行匹配
print(test_half.head(3))
print(test_time.head(3))
test_half.columns = ['flow', 'location1', 'weekday1', 'hour1']
test_time.columns = ['time', 'location']
test_half['weekday1'] = test_half['weekday1'].astype(str)
test_half['hour1'] = test_half['hour1'].astype(str)
test_half['location1'] = test_half['location1'].astype(str)
test_time['time'] = test_time['time'].astype(str)
test_time['location'] = test_time['location'].astype(str)
test_time['day'] = test_time['time'].str.split(' ', expand=True)[0]
test_time['hour2'] = test_time['time'].str.split(' ', expand=True)[1]
test_time['hour'] = test_time['hour2'].str.split(':', expand=True)[0]
test_time.pop('hour2')


# 需要将test的时间序列分解出weekday信息
weekday = []
print(len(test_time['day']))
for i in range(len(test_time['day'])):
    if i % 1000 == 0:
        print(i)
    date = datetime.datetime.strptime(test_time.iloc[i, 2], '%Y/%m/%d')
    # print(date, type(date))
    weekday.append(date.weekday()+1)
test_time['weekday'] = weekday
test_time['weekday'] = test_time['weekday'].astype(str)
test_time.columns = ['time', 'location', 'day', 'hour', 'weekday']
print(test_time.head(3))
test_half['Location+weekday+hour'] = test_half['location1'] + '-' + test_half['weekday1'] + '-' + test_half['hour1']
test_time['Location+weekday+hour'] = test_time['location'] + '-' + test_time['weekday'] + '-' + test_time['hour']
test_full = pd.merge(test_time, test_half, how='left', on='Location+weekday+hour')
test_full.info()

# 填充hour和day的长度
test_full['hour'] = test_full['hour'].astype(str).str.pad(2, side='left', fillchar='0')
test_full['date'] = test_time['day'].str.split('/', expand=True)[2].astype(str).str.pad(2, side='left', fillchar='0')
test_full['month'] = test_time['day'].str.split('/', expand=True)[1]
test_full['year'] = test_time['day'].str.split('/', expand=True)[0]
test_full['daytime'] = test_full['year'] + '-' + test_full['month'] + '-' + test_full['date']
test_full.pop('date')
test_full.pop('month')
test_full.pop('year')
test_full.pop('day')
print(test_full.head(3))


test_full['time_stamp'] = test_full['daytime'] + ' ' + test_full['hour']
test_full.pop('time')
test_full.pop('daytime')
test_full.pop('hour')
test_full.pop('weekday')
test_full.pop('Location+weekday+hour')
test_full.pop('location1')
test_full.pop('weekday1')
test_full.pop('hour1')
print(len(test_time), len(test_full), len(test_half))
test_full.columns = ['loc_id', 'num_of_people', 'time_stamp']
print(test_full.head(3))
test_full.info()
print(len(test_full['time_stamp'].value_counts()))
print(len(test_half['Location+weekday+hour'].value_counts()))
print(len(test_time['Location+weekday+hour'].value_counts()))
test_full.to_csv('E:/学习/研究生1/创新竞赛/2018年算法赛/data/rules/12月规则一_全时间8.csv', index=False)
# loc_id,time_stamp,num_of_people



