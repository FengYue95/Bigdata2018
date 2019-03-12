# coding:utf-8
# 预处理三周信息，整合成
# Location, Time, FLow
# 对于部分没有流量的信息，补全，进行可视化查看

# B2=IF(MOD(D1,36)=0,B1+(1/24),B1)  本excel公式要仔细想
# D2=MOD(C2,36)+1
import pandas as pd


# 数据读取，整合
path_train1 = "E:/学习/研究生1/创新竞赛/2018年算法赛/data/months1_10/10.csv"
path_train2 = "E:/学习/研究生1/创新竞赛/2018年算法赛/data/months1_10/11.csv"
train1 = pd.read_csv(path_train1)
train2 = pd.read_csv(path_train2)
train = pd.concat([train1, train2], axis=0).reset_index(drop=True)
# train['location'] = train['location'].astype(str)
# train['time'] = train['time'].astype(str).str.pad(6, side='left', fillchar='0')  # 将时间格式补全
print('补全完成')

train['LT'] = train['time'] + '/' + train['location'].map(str)
# train['LT'] = train['LT'].astype(int)
train = train.sort_values(by=['LT'])  # 进行排序
print(train.head(3))


# 统计各个地点，时间的人流量
train_count = train['LT'].value_counts()  # 进行统计
print(len(train_count), train_count.head(3))
train_count.to_csv('E:/学习/研究生1/创新竞赛/2018年算法赛/data/rules/train_count10_11.csv', index=True)

