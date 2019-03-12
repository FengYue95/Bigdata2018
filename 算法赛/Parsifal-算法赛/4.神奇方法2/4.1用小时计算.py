# coding:utf-8

import pandas as pd
import numpy as np
import datetime

train_full = pd.read_csv('E:/学习/研究生1/创新竞赛/2018年算法赛/data/rules/train_9_10完整时间人流量_时间分解.csv')
print(train_full.head(3))


# 异常值变为中位数
for i in range(1, 34):
    for j in range(24):
        data = train_full.drop(train_full[(train_full['location'] != i) | (train_full['hour'] != j)].index)  # .reset_index(drop=True)
        # print(data.index)
        # print(data['registered'])
        q_point = data['flow'].quantile([0.25, 0.75, 0.5]).values  # 找到上下四分位点
        up = q_point[1] + 1.5 * (q_point[1] - q_point[0])
        down = q_point[0] - 1.5 * (q_point[1] - q_point[0])
        middle = q_point[2]
        print('第', i, '个地点', '第', j, '小时', up, down, q_point[0], q_point[1])
        print(middle)
        temp = data[(data['flow'] < down) | (data['flow'] > up)].index
        print(train_full.loc[temp]['flow'])
        for item in temp:
            train_full['flow'][item] = middle
        print(train_full.loc[temp]['flow'])

# 开始分析流量
result_location_hour = pd.DataFrame()
for i in range(1, 34):  # 指明33个location
    print(i)
    for j in range(0, 24):  # 指明24个小时
        location_hour = train_full.drop(train_full[(train_full['location'] != i) | (train_full['hour'] != j)].index).reset_index(drop=True)
        # print(location_weekday.head(3))
        print(len(location_hour['weekday']))  # 查看取出的地点，在00点的情况下，是否有7*6=42个数据点
        zhongwei = np.zeros((6, 7), dtype=float)

        for k in range(1, 7):  # 指明周数
            location_hour_week = location_hour.drop(location_hour[(location_hour['week'] != k)].index).reset_index(drop=True)  # 得到24个时间点
            location_hour_week = location_hour_week.sort_values(by=['weekday'])  # 按照时间顺序从小到大排列
            # print(location_weekday_week.head(3))
            # print(len(location_weekday_week['hour']))
            temp = location_hour_week['flow'].values
            # print(temp)
            # print(np.median(temp))
            if np.mean(temp) != 0:
                for m in range(7):  # 记录weekday的中位数
                    zhongwei[k-1][m] = (location_hour_week.iloc[m, 2])/np.mean(temp)  # 记录每小时的中位数占比

            else:
                for n in range(7):  # 记录weekday的中位数
                    zhongwei[k-1][n] = 0  # 记录weekday的中位数
        # print(zhongwei)
        result = (np.median(zhongwei, axis=0)) * (np.mean(temp))
        print('第', i, '个地点', '  第', j, '小时')
        # print(result)
        temp = pd.DataFrame(result, columns=['flow2'])
        temp['location'] = i
        temp['hour'] = j
        temp['weekday'] = np.arange(7)+1
        result_location_hour = pd.concat([result_location_hour, temp], axis=0).reset_index(drop=True)
        # print(result_location_weekday)
print(len(result_location_hour))
# result_location_weekday.columns = ['location', 'weekday', 'hour', 'flow']
result_location_hour.to_csv('E:/学习/研究生1/创新竞赛/2018年算法赛/data/rules/9_10规则二_6周.csv', index=False)