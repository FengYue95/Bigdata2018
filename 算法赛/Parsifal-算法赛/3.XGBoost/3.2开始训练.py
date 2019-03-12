# coding:utf-8
import numpy as np
import pandas as pd
from pandas import DataFrame
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error

from hyperopt import fmin, tpe, hp,space_eval,rand,Trials,partial,STATUS_OK
import matplotlib.pyplot as plt
# 读取数据
path_train = 'E:/学习/研究生1/创新竞赛/2018年算法赛/data/xgboost/train_featured.csv'
path_test = 'E:/学习/研究生1/创新竞赛/2018年算法赛/data/xgboost/test_featured.csv'
train = pd.read_csv(path_train)
test = pd.read_csv(path_test)
rule1 = train['flow'].values
rule2 = train['flow2'].values

X_test_divided = train.drop(train[(train['month'] != 10) | (train['day'] < 25)].index).reset_index(drop=True)
X_train_divided = train.drop(train[(train['month'] == 10) & (train['day'] > 24)].index).reset_index(drop=True)
print(len(X_train_divided))
'''
train_full = X_train_divided

# 异常值处理

for i in range(1, 34):
    for j in range(24):
        data = train_full.drop(train_full[(train_full['location'] != i) | (train_full['hour'] != j)].index)  # .reset_index(drop=True)
        # print(data.index)
        # print(data['registered'])
        q_point = data['real_flow'].quantile([0.25, 0.75, 0.5]).values  # 找到上下四分位点
        up = q_point[1] + 1.5 * (q_point[1] - q_point[0])
        down = q_point[0] - 1.5 * (q_point[1] - q_point[0])
        middle = q_point[2]
        print('第', i, '个地点', '第', j, '小时', up, down, q_point[0], q_point[1])
        # print(middle)
        temp = data[(data['real_flow'] < down) | (data['real_flow'] > up)].index
        # print(train_full.loc[temp]['real_flow'])
        for item in temp:
            train_full['real_flow'][item] = middle
        # print(train_full.loc[temp]['real_flow'])
        # train_full.loc[data[(data['registered'] < down) | (data['registered'] > up)].index]['registered'] = middle

        # print(train_full.loc[data[(data['registered'] < down) | (data['registered'] > up)].index]['registered'])

        # train_full = train_full.drop(data[(data['registered'] < down) | (data['registered'] > up)].index)
        # print(len(train_full['registered']))

X_train_divided = train_full

print('异常值处理完毕')
X_train_divided.to_csv('E:/学习/研究生1/创新竞赛/2018年算法赛/data/xgboost/X_train_divided_3.2.csv', index=False)
'''
X_train_divided = pd.read_csv('E:/学习/研究生1/创新竞赛/2018年算法赛/data/xgboost/X_train_divided_3.2.csv')


y_train_divided = X_train_divided.pop('real_flow')
y_test_divided = X_test_divided.pop('real_flow')


# 测试规则法专用区
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
y_rule = X_test_divided['flow'].values
y_rule2 = X_test_divided['flow2'].values
# print(score)
score2 = rmse(y_rule, y_test_divided.values)
score3 = rmse(y_rule2, y_test_divided.values)
score4 = rmse((y_rule*0.5)+(y_rule2*0.5), y_test_divided.values)
print(score2, score3, score4)


# print(X_train_divided.head(3))
# print(X_test_divided.head(3))
y_train = train.pop('real_flow')
# print(train.head(3))
# print(train.head(3))

# print(rmse(train['flow'].values, y_train.values))
# y_train = np.log1p(y_train)
# train['flow'] = np.log1p(train['flow'])
# test['flow'] = np.log1p(test['flow'])

train.pop('day')
test.pop('day')

# train.pop('flow')
# test.pop('flow')


# 将分类的数值特征进行one-hot编码
cols = ('location', 'weekday', 'hour_cat', 'big_loc', 'weather1', 'weather2', 'air', 'working')
for c in cols:
    train[c] = train[c].astype(str)
    test[c] = test[c].astype(str)
    X_train_divided[c] = X_train_divided[c].astype(str)
    X_test_divided[c] = X_test_divided[c].astype(str)
train = pd.get_dummies(train)
test = pd.get_dummies(test)
X_train_divided = pd.get_dummies(X_train_divided)
X_test_divided = pd.get_dummies(X_test_divided)

# 分割训练集和验证集合
# X_train_divided, X_test_divided, y_train_divided, y_test_divided = train_test_split(train, y_train, test_size=0.3, random_state=0)

features = ['month', 'hour', 'max_temp', 'min_temp', 'location_1', 'location_10', 'location_11', 'location_12', 'location_13',
            'location_14', 'location_15', 'location_16', 'location_17', 'location_18', 'location_19', 'location_2', 'location_20', 'location_21', 'location_22',
            'location_23', 'location_24', 'location_25', 'location_26', 'location_27', 'location_28', 'location_29', 'location_3', 'location_30', 'location_31',
            'location_32', 'location_33', 'location_4', 'location_5', 'location_6', 'location_7', 'location_8', 'location_9', 'weekday_1', 'weekday_2', 'weekday_3',
            'weekday_4', 'weekday_5', 'weekday_6', 'weekday_7', 'hour_cat_0', 'hour_cat_1', 'hour_cat_10', 'hour_cat_11', 'hour_cat_12', 'hour_cat_13', 'hour_cat_14',
            'hour_cat_15', 'hour_cat_16', 'hour_cat_17', 'hour_cat_18', 'hour_cat_19', 'hour_cat_2', 'hour_cat_20', 'hour_cat_21', 'hour_cat_22', 'hour_cat_23',
            'hour_cat_3', 'hour_cat_4', 'hour_cat_5', 'hour_cat_6', 'hour_cat_7', 'hour_cat_8', 'hour_cat_9', 'big_loc_jiaoxuelou', 'big_loc_shitang', 'big_loc_sushe',
            'weather1_1', 'weather1_2', 'weather1_4', 'weather2_1', 'weather2_2', 'working_no', 'working_yes']
# 删除的特征有：'weather1_3' 'weather2_3' 'weather2_4' 'air4', 'air_1', 'air_2', 'air_3', 'wind1', 'wind2' 'flow', 'flow2', 'flow3',

X_train_divided = X_train_divided[features]
X_test_divided = X_test_divided[features]
'''
# X_train_divided.pop('flow')
# y_rule = X_test_divided.pop('flow')
# print(y_train_divided.head(3))
# print(y_test_divided.head(3))
# 真正预测部分
gbm_count = XGBRegressor(nthread=4,  # 进程数
                         n_estimators=310,  # 树的数量
                         learning_rate=0.03,  # 学习率
                         subsample=0.3,  # 采样数
                         max_depth=2,  # 最大深度
                         min_child_weight=4,  # 孩子数
                         )  # max_delta_step=10)  # 10步不降则停止
gbm_count.fit(X_train_divided[features], y_train_divided)
y_pred = gbm_count.predict(X_test_divided[features])
# y_rule = X_test_divided['flow'].values
# y_rule2 = X_test_divided['flow2'].values
# y_rule3 = X_test_divided['flow3'].values
score = rmse(y_pred, y_test_divided.values)
print(score)
# score2 = rmse(y_rule, y_test_divided.values)
# score3 = rmse(y_rule2, y_test_divided.values)
# score4 = rmse(y_rule3, y_test_divided.values)
# score4_1 = rmse((y_rule*0.5)+(y_rule2*0.5), y_test_divided.values)
# score5 = rmse((y_rule*0.5)+(y_rule3*0.5), y_test_divided.values)
# score6 = rmse((y_rule2*0.5)+(y_rule3*0.5), y_test_divided.values)
# score7 = rmse((y_rule*(0.3))+(y_rule2*(0.3)+(y_rule3*(0.4))), y_test_divided.values)
# print(score, score2, score3, score4, score4_1, score5, score6, score7)
# y_pred = pd.DataFrame(y_pred)

# y_pred.to_csv('E:/学习/研究生1/创新竞赛/2018年算法赛/data/xgboost/纯XGB.csv')

'''



def GBM(argsDict):

    global X_train_divided, y_train_divided, y_test_divided, X_test_divided
    gbm = XGBRegressor(nthread=4,    # 进程数
                       max_depth=argsDict["max_depth"],  # 最大深度
                       n_estimators=argsDict['n_estimators'] * 10 + 50,   # 树的数量
                       learning_rate=argsDict["learning_rate"] * 0.01 + 0.02,  # 学习率
                       subsample=argsDict["subsample"] * 0.1 + 0.1,  # 采样数
                       min_child_weight=argsDict["min_child_weight"],   # 孩子数
                       )# max_delta_step=10)  # 10步不降则停止

    # gbm.fit(X_train_divided, y_train_divided)
    # y_pred = gbm.predict(X_test_divided)
    # score = rmse(y_pred, y_test_divided.values)
    score = np.mean(np.sqrt(-cross_val_score(gbm, X_train_divided, y_train_divided, cv=2, scoring='neg_mean_squared_error', verbose=0)))
    print(score)
    return score


space = {"max_depth": hp.randint("max_depth", 15),
         "n_estimators": hp.randint("n_estimators", 30),  # [0,1,2,3,4,5] -> [50,]
         "learning_rate": hp.randint("learning_rate", 10),  # [0,1,2,3,4,5] -> 0.05,0.06
         "subsample": hp.randint("subsample", 10),  # [0,1,2,3] -> [0.7,0.8,0.9,1.0]
         "min_child_weight": hp.randint("min_child_weight", 10)
         }
algo = partial(tpe.suggest, n_startup_jobs=10)
best = fmin(GBM, space, algo=algo, max_evals=100)

print(best)
print(GBM(best))

