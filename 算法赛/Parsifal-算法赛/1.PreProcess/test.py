# coding:utf-8
import pandas as pd
import datetime

people = pd.read_csv('E:/学习/研究生1/创新竞赛/2018年算法赛/data/xgboost/常驻人口统计_用户标签_总次数_改变编码.csv')
print(people.head(3))
result = people['count_次数'].value_counts()
result.to_csv('E:/学习/研究生1/创新竞赛/2018年算法赛/data/xgboost/频数统计.csv', index=True, encoding='utf-8')