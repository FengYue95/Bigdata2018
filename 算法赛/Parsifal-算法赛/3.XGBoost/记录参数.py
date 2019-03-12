'''
# ---------------------------------------------食堂预测部分--------------------------------------------------------#
# 真正预测部分
gbm_shitang_registered = XGBRegressor(nthread=4,  # 进程数
                                      n_estimators=80,  # 树的数量
                                      learning_rate=0.08,  # 学习率
                                      subsample=0.3,  # 采样数
                                      max_depth=10,  # 最大深度
                                      min_child_weight=8,  # 孩子数
                                      )  # max_delta_step=10)  # 10步不降则停止
gbm_shitang_registered.fit(X_train_divided_shitang[features_shitang], y_train_divided_shitang_registered)   # X_train_divided[features], y_train_divided_registered  # train[features], y_train_registered
y_pred_shitang_registered = gbm_shitang_registered.predict(X_test_divided_shitang[features_shitang])  # X_test_divided[features]  # test[features]


print('registered predicted !')
gbm_shitang_casual = XGBRegressor(nthread=4,  # 进程数
                                  n_estimators=140,  # 树的数量
                                  learning_rate=0.06,  # 学习率
                                  subsample=0.3,  # 采样数
                                  max_depth=4,  # 最大深度
                                  min_child_weight=3,  # 孩子数
                                  )  # max_delta_step=10)  # 10步不降则停止
gbm_shitang_casual.fit(X_train_divided_shitang[features_shitang], y_train_divided_shitang_casual)  # X_train_divided[features], y_train_divided_casual  # train[features], y_train_casual
y_pred_shitang_casual = gbm_shitang_casual.predict(X_test_divided_shitang[features_shitang])  # X_test_divided[features]  # test[features]
print('casual predicted !')

# xgb.plot_importance(gbm_registered)
# plt.show()
# xgb.plot_importance(gbm_casual)
# plt.show()
y_pred_shitang = y_pred_shitang_registered + y_pred_shitang_casual
zero_count = 0
print('食堂测试集一共', len(y_pred_shitang), '条数据')
for i in range(len(y_pred_shitang)):
    # print(i)
    if y_pred_shitang[i] < 0:
        y_pred_shitang[i] = 0
        zero_count += 1
print('一共统计出', zero_count, '个0')

X_test_divided_shitang0['pred_flow'] = y_pred_shitang

# ---------------------------------------------其余地点预测部分--------------------------------------------------------#
# 真正预测部分
gbm_else_registered = XGBRegressor(nthread=4,  # 进程数
                                   n_estimators=220,  # 树的数量
                                   learning_rate=0.04,  # 学习率
                                   subsample=0.2,  # 采样数
                                   max_depth=6,  # 最大深度
                                   min_child_weight=2,  # 孩子数
                                   )  # max_delta_step=10)  # 10步不降则停止
gbm_else_registered.fit(X_train_divided_else[features_else], y_train_divided_else_registered)   # X_train_divided[features], y_train_divided_registered  # train[features], y_train_registered
y_pred_else_registered = gbm_else_registered.predict(X_test_divided_else[features_else])  # X_test_divided[features]  # test[features]
print('registered predicted !')
gbm_else_casual = XGBRegressor(nthread=4,  # 进程数
                               n_estimators=130,  # 树的数量
                               learning_rate=0.08,  # 学习率
                               subsample=0.4,  # 采样数
                               max_depth=8,  # 最大深度
                               min_child_weight=4,  # 孩子数
                               )  # max_delta_step=10)  # 10步不降则停止
'''
