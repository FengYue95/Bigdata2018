# coding:utf-8

import pandas as pd
import numpy as np
import datetime

path_train_full = 'E:/学习/研究生1/创新竞赛/2018年算法赛/data/rules/train_9_10完整时间人流量_count_registered.csv'
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


# 排除掉除了9月13号之前的数据
# train_full = train_full.drop(train_full[(train_full['month'].map(int) == 9) & (train_full['day'].map(int) < 27)].index).reset_index(drop=True)

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


# 对周数进行替换37->1
train_full['week'] = train_full['week'].replace(37, 1)
train_full['week'] = train_full['week'].replace(38, 2)
train_full['week'] = train_full['week'].replace(39, 3)
train_full['week'] = train_full['week'].replace(41, 4)
train_full['week'] = train_full['week'].replace(42, 5)
train_full['week'] = train_full['week'].replace(43, 6)
train_full['week'] = train_full['week'].replace(44, 1)


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
train_full.to_csv('E:/学习/研究生1/创新竞赛/2018年算法赛/data/rules/train_9_10完整时间人流量_时间分解_6周数据_人群分类.csv', index=False)

train_full = pd.read_csv('E:/学习/研究生1/创新竞赛/2018年算法赛/data/rules/train_9_10完整时间人流量_时间分解_6周数据_人群分类.csv')
print(train_full.head(3))
# train_full.columns = ['time', 'location', 'registered', 'casual', 'month', 'day', 'hour', 'week', 'weekday']

# 开始分析流量
result_location_weekday = pd.DataFrame()
for i in range(1, 34):  # 指明33个location
    print(i)
    for j in range(1, 8):  # 指明周一到周日
        location_weekday = train_full.drop(train_full[(train_full['location'] != i) | (train_full['weekday'] != j)].index).reset_index(drop=True)
        # print(location_weekday.head(3))
        # print(len(location_weekday['hour']))  # 查看取出的地点，在星期一的情况下，是否有24*3=72个数据点
        zhongwei_registered = np.zeros((6, 24), dtype=float)
        zhongwei_casual = np.zeros((6, 24), dtype=float)
        # reslut = [0]*24  # 用来存储，6个星期中每一个小时的中位数结果
        for k in range(1, 7):  # 指明周数
            location_weekday_week = location_weekday.drop(location_weekday[(location_weekday['week'] != k)].index).reset_index(drop=True)  # 得到24个时间点
            location_weekday_week = location_weekday_week.sort_values(by=['hour'])  # 按照时间顺序从小到大排列
            # print(location_weekday_week.head(3))
            # print(len(location_weekday_week['hour']))
            registered = location_weekday_week['registered'].values
            casual = location_weekday_week['casual'].values
            print(registered)
            print(casual)
            # print(np.median(temp))
            if np.sum(registered) != 0:
                for m in range(24):  # 记录每小时的中位数占比
                    zhongwei_registered[k-1][m] = (location_weekday_week.iloc[m, 2])/np.sum(registered)  # 记录每小时的中位数占比
                    # zhongwei_casual[k-1][m] = (location_weekday_week.iloc[m, 3]) / np.sum(casual)  # 记录每小时的中位数占比
            else:
                for n in range(24):  # 记录每小时的中位数占比
                    zhongwei_registered[k-1][n] = 0  # 记录每小时的中位数占比
                    # zhongwei_casual[k - 1][n] = 0  # 记录每小时的中位数占比

            if np.sum(casual) != 0:
                for m in range(24):  # 记录每小时的中位数占比
                    # zhongwei_registered[k - 1][m] = (location_weekday_week.iloc[m, 2]) / np.sum(registered)  # 记录每小时的中位数占比
                    zhongwei_casual[k-1][m] = (location_weekday_week.iloc[m, 3]) / np.sum(casual)  # 记录每小时的中位数占比
            else:
                for n in range(24):  # 记录每小时的中位数占比
                    # zhongwei_registered[k - 1][n] = 0  # 记录每小时的中位数占比
                    zhongwei_casual[k - 1][n] = 0  # 记录每小时的中位数占比
        # print(zhongwei)
        result_registered = (np.median(zhongwei_registered, axis=0)) * (np.sum(registered))
        result_casual = (np.median(zhongwei_casual, axis=0)) * (np.sum(casual))
        print('第', i, '个地点', '  星期', j)
        # print(result)
        temp = pd.DataFrame(result_registered, columns=['registered'])
        temp['casual'] = result_casual
        temp['location'] = i
        temp['weekday'] = j
        temp['hour'] = np.arange(24)
        result_location_weekday = pd.concat([result_location_weekday, temp], axis=0).reset_index(drop=True)
        # print(result_location_weekday)
print(len(result_location_weekday))
# result_location_weekday.columns = ['location', 'weekday', 'hour', 'flow']
result_location_weekday['flow'] = result_location_weekday['registered'].map(float) + result_location_weekday['casual'].map(float)
result_location_weekday.pop('registered')
result_location_weekday.pop('casual')
result_location_weekday.to_csv('E:/学习/研究生1/创新竞赛/2018年算法赛/data/rules/9_10规则一_6周_人群分类.csv', index=False)
