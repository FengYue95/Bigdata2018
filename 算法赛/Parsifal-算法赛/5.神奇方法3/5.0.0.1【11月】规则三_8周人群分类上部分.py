# coding:utf-8
# 本部分将数据分解，下放到具体地点目录/周/星期几，每个文件存放一个csv

import pandas as pd
import numpy as np
import datetime
'''
path_train_full = 'E:/学习/研究生1/创新竞赛/2018年算法赛/data/rules/train_10_11完整时间人流量_count_registered.csv'
train_full = pd.read_csv(path_train_full)
train_full.columns = ['time', 'location', 'registered', 'casual']
train_full['month'] = train_full['time'].str.split('/', expand=True)[1]
train_full['day'] = train_full['time'].str.split('/', expand=True)[2]
train_full['day2'] = train_full['day'].str.split(' ', expand=True)[0]
train_full['day2'] = train_full['day2'] .astype(int)
train_full['hour'] = train_full['day'].str.split(' ', expand=True)[1]
train_full['hour2'] = train_full['hour'].str.split(':', expand=True)[0]
train_full.pop('day')
train_full.pop('hour')
train_full.columns = ['time', 'location', 'registered', 'casual', 'month', 'day', 'hour']


# 处理time，增加weekday,week信息
weekday = []
week = []
print(len(train_full['time']))
for i in range(len(train_full['time'])):
    if i % 1000 == 0:
        print(i)
    date = datetime.date(2017, int(train_full.iloc[i, 4]), int(train_full.iloc[i, 5]))
    # print(date, type(date))
    weekday.append(date.weekday()+1)
    week.append((date.isocalendar()[1]))
train_full['week'] = week
train_full['weekday'] = weekday
print(train_full.head(3))


# 对周数进行替换
train_full['week'] = train_full['week'].replace(41, 1)
train_full['week'] = train_full['week'].replace(42, 2)
train_full['week'] = train_full['week'].replace(43, 3)
train_full['week'] = train_full['week'].replace(44, 4)
train_full['week'] = train_full['week'].replace(45, 5)
train_full['week'] = train_full['week'].replace(46, 6)
train_full['week'] = train_full['week'].replace(47, 7)
train_full['week'] = train_full['week'].replace(48, 8)

# 第一周数据10月9号和10月10号有较多异常值，抛弃不用，使用16号和17号进行填补
train_full = train_full.drop(train_full[(train_full['month'].map(int) == 10) & (train_full['day'].map(int) > 8) & (train_full['day'].map(int) < 11)].index).reset_index(drop=True)
train_sup = train_full[(train_full['month'].map(int) == 10) & (train_full['day'].map(int) > 15) & (train_full['day'].map(int) < 18)]
print(train_sup.head(3), len(train_sup))
train_sup['day'] = train_sup['day'].replace(16, 9)  # 日期更换为9,10号
train_sup['day'] = train_sup['day'].replace(17, 10)
train_sup['week'] = train_sup['week'].replace(2, 1)
print(len(train_full))
train_full = pd.concat([train_full, train_sup], axis=0)
print(len(train_full))

# 最后一周数据不能填到第一周，用前一周的数据补足最后一周
train_sup = train_full[(train_full['month'].map(int) == 11) & (train_full['day'].map(int) > 23) & (train_full['day'].map(int) < 27)]
print(train_sup.head(3), len(train_sup))
train_sup['day'] = train_sup['day'].replace(24, 31)  # 日期更换为31,32,33号
train_sup['day'] = train_sup['day'].replace(25, 32)
train_sup['day'] = train_sup['day'].replace(26, 33)
train_sup['week'] = train_sup['week'].replace(7, 8)
print(len(train_full))
train_full = pd.concat([train_full, train_sup], axis=0)
print(len(train_full))


# 异常值变为中位数
for i in range(1, 34):
    for j in range(24):
        data = train_full.drop(train_full[(train_full['location'].map(int) != i) | (train_full['hour'].map(int) != j)].index)  # .reset_index(drop=True)
        # print(data.index)
        # print(data['registered'])
        q_point = data['registered'].quantile([0.25, 0.75, 0.5]).values  # 找到上下四分位点
        up = q_point[1] + 1.5 * (q_point[1] - q_point[0])
        down = q_point[0] - 1.5 * (q_point[1] - q_point[0])
        middle = q_point[2]
        print('第', i, '个地点', '第', j, '小时', up, down, q_point[0], q_point[1])
        print(middle)
        temp = data[(data['registered'] < down) | (data['registered'] > up)].index
        print(train_full.loc[temp]['registered'])
        for item in temp:
            train_full['registered'][item] = middle
        print(train_full.loc[temp]['registered'])

for i in range(1, 34):
    for j in range(24):
        data = train_full.drop(train_full[(train_full['location'].map(int) != i) | (train_full['hour'].map(int) != j)].index)  # .reset_index(drop=True)
        # print(data.index)
        # print(data['registered'])
        q_point = data['casual'].quantile([0.25, 0.75, 0.5]).values  # 找到上下四分位点
        up = q_point[1] + 1.5 * (q_point[1] - q_point[0])
        down = q_point[0] - 1.5 * (q_point[1] - q_point[0])
        middle = q_point[2]
        print('第', i, '个地点', '第', j, '小时', up, down, q_point[0], q_point[1])
        print(middle)
        temp = data[(data['casual'] < down) | (data['casual'] > up)].index
        print(train_full.loc[temp]['casual'])
        for item in temp:
            train_full['casual'][item] = middle
        print(train_full.loc[temp]['casual'])


print(train_full['week'].value_counts())
train_full.to_csv('E:/学习/研究生1/创新竞赛/2018年算法赛/data/rules/train_10_11完整时间人流量_时间分解_8周数据_人群分类.csv', index=False)
'''

train_full = pd.read_csv('E:/学习/研究生1/创新竞赛/2018年算法赛/data/rules/train_10_11完整时间人流量_时间分解_8周数据_人群分类.csv')
print(train_full.head(3))
# train_full.columns = ['time', 'location', 'registered', 'casual', 'month', 'day', 'hour', 'week', 'weekday']

# 开始分析流量
result_location_hour = pd.DataFrame()
for i in range(1, 8):  # 指明周一到周日
    print(i)
    for j in range(0, 24):  # 指明24个小时
        weekday_hour = train_full.drop(train_full[(train_full['weekday'] != i) | (train_full['hour'] != j)].index).reset_index(drop=True)
        # print(location_weekday.head(3))
        print(len(weekday_hour['location']))  # 查看取出的weekday，在location1的情况下，是否有33*6=198个数据点
        zhongwei_registered = np.zeros((8, 33), dtype=float)
        zhongwei_casual = np.zeros((8, 33), dtype=float)

        for k in range(1, 9):  # 指明周数
            weekday_hour_week = weekday_hour.drop(weekday_hour[(weekday_hour['week'] != k)].index).reset_index(drop=True)  # 得到24个时间点
            weekday_hour_week = weekday_hour_week.sort_values(by=['location'])  # 按照时间顺序从小到大排列
            # print(location_weekday_week.head(3))
            # print(len(location_weekday_week['hour']))
            registered = weekday_hour_week['registered'].values
            casual = weekday_hour_week['casual'].values

            if np.sum(registered) != 0:
                for m in range(33):  # 记录每小时的中位数占比
                    zhongwei_registered[k-1][m] = (weekday_hour_week.iloc[m, 2])/np.sum(registered)  # 记录每小时的中位数占比

            else:
                for n in range(33):  # 记录每小时的中位数占比
                    zhongwei_registered[k-1][n] = 0  # 记录每小时的中位数占比

            if np.sum(casual) != 0:
                for m in range(33):  # 记录每小时的中位数占比
                    zhongwei_casual[k - 1][m] = (weekday_hour_week.iloc[m, 3]) / np.sum(casual)  # 记录每小时的中位数占比

            else:
                for n in range(33):  # 记录每小时的中位数占比
                    zhongwei_casual[k - 1][n] = 0  # 记录每小时的中位数占比
        # print(zhongwei)
        result_registered = (np.median(zhongwei_registered, axis=0)) * (np.sum(registered))
        result_casual = (np.median(zhongwei_casual, axis=0)) * (np.sum(casual))
        print('星期', i, '  第', j, '小时')
        # print(result)
        temp = pd.DataFrame(result_registered, columns=['registered'])
        temp['casual'] = result_casual
        temp['weekday'] = i
        temp['hour'] = j
        temp['location'] = np.arange(33)+1
        result_location_hour = pd.concat([result_location_hour, temp], axis=0).reset_index(drop=True)
        # print(result_location_weekday)
print(len(result_location_hour))

result_location_hour['flow'] = result_location_hour['registered'].map(float) + result_location_hour['casual'].map(float)
result_location_hour.pop('registered')
result_location_hour.pop('casual')
# result_location_weekday.columns = ['location', 'weekday', 'hour', 'flow']
result_location_hour.to_csv('E:/学习/研究生1/创新竞赛/2018年算法赛/data/rules/10_11规则三_8周_人群分类.csv', index=False)
