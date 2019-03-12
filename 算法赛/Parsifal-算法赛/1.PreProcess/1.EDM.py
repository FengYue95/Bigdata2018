# coding:utf-8

import pandas as pd
path_train = 'E:/学习/研究生1/创新竞赛/2018年算法赛/data/train.csv'
train = pd.read_csv(path_train)
train.info()
# train = train.head(5)
train_frequency = train['userID'].value_counts()
print(train_frequency.head(3))
print(train.head(3))
train['time'] = train['time'].astype(str).str.pad(6, side='left', fillchar='0')

print(train.head(3))
train['month'] = "None"
train['day'] = "None"
train['hour'] = "None"
month = []
day = []
hour = []
for i in range(len(train['userID'])):  #
    # print(str(train.iloc[i, 1])[:2], train['month'][i])
    if i % 1000 == 0:
        print(i,)
    time = str(train.iloc[i, 1])
    month.append(str(time[:2]))
    day.append(str(time[2:4]))
    hour.append(str(time[4:]))
train['month'] = month
train['day'] = day
train['hour'] = hour
print(train.head(5))
train.to_csv('E:/学习/研究生1/创新竞赛/2018年算法赛/data/train_date.csv', index=False)