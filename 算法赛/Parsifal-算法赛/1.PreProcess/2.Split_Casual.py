# coding:utf-8

# 主要实现了建立用户标签，是常驻用户还是casual用户

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
path_train = 'E:/学习/研究生1/创新竞赛/2018年算法赛/data/train.csv'
train = pd.read_csv(path_train)
train.info()
train_frequency = train['userID'].value_counts()  # 统计每个用户在总记录中出现的次数
print(type(train_frequency), len(train_frequency))
print(train_frequency.head(5))
cat = []
train_frequency = pd.DataFrame(train_frequency)
train_frequency['cat'] = '2'  # 因为frequency是按照从大到小顺序排列，所以能够直接按照次序来安排分类
train_frequency['cat'][0:10000] = '1'

train_frequency.columns = ['freq', 'cat']
print(train_frequency.head(5))
print(train_frequency.iloc[10001, :])
train_frequency.to_csv('E:/学习/研究生1/创新竞赛/2018年算法赛/data/train_frequency.csv', index=True)

'''
for i in range(len(train_frequency['freq'])):

    freq = train_frequency.iloc[i, 0]
    if 0 <= freq <= 100:
        cat.append('0-100')
    elif 100 < freq <= 500:
        cat.append('100-500')
    elif 500 < freq <= 1000:
        cat.append('500-1000')
    elif 1000 < freq <= 3000:
        cat.append('1000-3000')
    elif 3000 < freq:
        cat.append('>3000')
    if i % 1000 == 0:
        print(i, freq, cat[-1])
train_frequency['cat'] = cat
train_frequency.head(5)
sns.boxplot(x='cat', y="freq", data=train_frequency, order=['0-100', '100-500', '500-1000', '1000-3000', '>3000'])
plt.show()
'''