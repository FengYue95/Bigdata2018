# coding:utf-8
# 本部分将数据分解，下放到具体地点目录/周/星期几，每个文件存放一个csv

import pandas as pd
import numpy as np
import datetime
'''
path_train_full = 'E:/学习/研究生1/创新竞赛/2018年算法赛/data/rules/train_9_10完整时间人流量.csv'
train_full = pd.read_csv(path_train_full)
print(train_full.head(3))
train_full['month'] = train_full['time'].str.split('/', expand=True)[1]
train_full['day'] = train_full['time'].str.split('/', expand=True)[2]
train_full['day2'] = train_full['day'].str.split(' ', expand=True)[0]
train_full['day2'] = train_full['day2'] .astype(int)
train_full['hour'] = train_full['day'].str.split(' ', expand=True)[1]
train_full['hour2'] = train_full['hour'].str.split(':', expand=True)[0]
train_full.pop('day')
train_full.pop('hour')
train_full.columns = ['time', 'location', 'flow', 'month', 'day', 'hour']
print(train_full.head(3))


# 排除掉除了9月25号到9月26号
train_full = train_full.drop(train_full[(train_full['month'].map(int) != 10) & (train_full['day'].map(int) < 27)].index).reset_index(drop=True)

# 处理time，增加weekday,week信息
weekday = []
week = []
print(len(train_full['time']))
for i in range(len(train_full['time'])):
    if i % 1000 == 0:
        print(i)
    date = datetime.date(2017, int(train_full.iloc[i, 3]), int(train_full.iloc[i, 4]))
    # print(date, type(date))
    weekday.append(date.weekday()+1)
    week.append((date.isocalendar()[1]))
train_full['week'] = week
train_full['weekday'] = weekday
print(train_full.head(3))


# 对周数进行替换37->1
train_full['week'] = train_full['week'].replace(37, 1)
train_full['week'] = train_full['week'].replace(38, 2)
train_full['week'] = train_full['week'].replace(39, 3)
train_full['week'] = train_full['week'].replace(41, 4)
train_full['week'] = train_full['week'].replace(42, 5)
train_full['week'] = train_full['week'].replace(43, 6)
train_full['week'] = train_full['week'].replace(44, 3)

print(train_full['week'].value_counts())
train_full.to_csv('E:/学习/研究生1/创新竞赛/2018年算法赛/data/rules/train_9_10完整时间人流量_时间分解_4周数据.csv', index=False)
'''

train_full = pd.read_csv('E:/学习/研究生1/创新竞赛/2018年算法赛/data/rules/train_9_10完整时间人流量_时间分解_4周数据.csv')
print(train_full.head(3))


# 开始分析流量
result_location_weekday = pd.DataFrame()
for i in range(1, 34):  # 指明33个location
    print(i)
    for j in range(1, 8):  # 指明周一到周日
        location_weekday = train_full.drop(train_full[(train_full['location'] != i) | (train_full['weekday'] != j)].index).reset_index(drop=True)
        # print(location_weekday.head(3))
        # print(len(location_weekday['hour']))  # 查看取出的地点，在星期一的情况下，是否有24*3=72个数据点
        zhongwei = np.zeros((4, 24), dtype=float)
        # reslut = [0]*24  # 用来存储，6个星期中每一个小时的中位数结果
        for k in range(3, 7):  # 指明周数
            location_weekday_week = location_weekday.drop(location_weekday[(location_weekday['week'] != k)].index).reset_index(drop=True)  # 得到24个时间点
            location_weekday_week = location_weekday_week.sort_values(by=['hour'])  # 按照时间顺序从小到大排列
            # print(location_weekday_week.head(3))
            # print(len(location_weekday_week['hour']))
            temp = location_weekday_week['flow'].values
            print(temp)
            # print(np.median(temp))
            if np.sum(temp) != 0:
                for m in range(24):  # 记录每小时的中位数占比
                    zhongwei[k-3][m] = (location_weekday_week.iloc[m, 2])/np.sum(temp)  # 记录每小时的中位数占比

            else:
                for n in range(24):  # 记录每小时的中位数占比
                    zhongwei[k-3][n] = 0  # 记录每小时的中位数占比
        # print(zhongwei)
        result = (np.median(zhongwei, axis=0)) * (np.sum(temp))
        print('第', i, '个地点', '  星期', j)
        # print(result)
        temp = pd.DataFrame(result, columns=['flow'])
        temp['location'] = i
        temp['weekday'] = j
        temp['hour'] = np.arange(24)
        result_location_weekday = pd.concat([result_location_weekday, temp], axis=0).reset_index(drop=True)
        # print(result_location_weekday)
print(len(result_location_weekday))
# result_location_weekday.columns = ['location', 'weekday', 'hour', 'flow']
result_location_weekday.to_csv('E:/学习/研究生1/创新竞赛/2018年算法赛/data/rules/9_10规则一_4周.csv', index=False)
