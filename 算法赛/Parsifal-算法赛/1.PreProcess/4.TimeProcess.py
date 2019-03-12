# coding:utf-8

# 处理Time信息，实现分割时间，找出星期几。

import pandas as pd
import datetime
path_count = 'E:/学习/研究生1/创新竞赛/2018年算法赛/data/train_count_real.csv'
path_regis = 'E:/学习/研究生1/创新竞赛/2018年算法赛/data/train_regis.csv'
path_casual = 'E:/学习/研究生1/创新竞赛/2018年算法赛/data/train_casual.csv'
count = pd.read_csv(path_count)
regis = pd.read_csv(path_regis)
casual = pd.read_csv(path_casual)
print(count.head(2))
print(regis.head(2))
print(casual.head(2))
count['time'] = count['time'].astype(str).str.pad(6, side='left', fillchar='0')
casual['time'] = casual['time'].astype(str).str.pad(6, side='left', fillchar='0')
regis.columns = ['time', 'location', 'cat', 'flow']
regis['time'] = regis['time'].astype(str).str.pad(6, side='left', fillchar='0')

month = []
day = []
hour = []
weekday = []
for i in range(len(count['time'])):  #
    # print(str(train.iloc[i, 1])[:2], train['month'][i])
    if i % 1000 == 0:
        print(i,)
    time = str(count.iloc[i, 0])
    month.append(str(time[:2]))
    day.append(str(time[2:4]))
    hour.append(str(time[4:]))
    weekday.append(datetime.date(2017, int(month[-1]), int(day[-1])).isoweekday())
count['month'] = month
count['day'] = day
count['hour'] = hour
count['weekday'] = weekday
print('count process finished')
print(count.head(5))

month = []
day = []
hour = []
weekday = []
for j in range(len(regis['time'])):  #
    # print(str(train.iloc[i, 1])[:2], train['month'][i])
    if j % 1000 == 0:
        print(j,)
    time = str(regis.iloc[j, 0])
    month.append(str(time[:2]))
    day.append(str(time[2:4]))
    hour.append(str(time[4:]))
    weekday.append(datetime.date(2017, int(month[-1]), int(day[-1])).isoweekday())
regis['month'] = month
regis['day'] = day
regis['hour'] = hour
regis['weekday'] = weekday
print('regis process finished')

month = []
day = []
hour = []
weekday = []
for k in range(len(casual['time'])):  #
    # print(str(train.iloc[i, 1])[:2], train['month'][i])
    if k % 1000 == 0:
        print(k,)
    time = str(casual.iloc[k, 0])
    month.append(str(time[:2]))
    day.append(str(time[2:4]))
    hour.append(str(time[4:]))
    weekday.append(datetime.date(2017, int(month[-1]), int(day[-1])).isoweekday())
casual['month'] = month
casual['day'] = day
casual['hour'] = hour
casual['weekday'] = weekday
print('casual process finished')

count.to_csv(path_count)
regis.to_csv(path_regis)
casual.to_csv(path_casual)
