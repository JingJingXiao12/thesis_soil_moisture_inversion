import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import KFold
import pickle
import math
import os


def get_train_val(train_data):
    '''
    将训练数据格式化为一个特征向量x和结果y
    :param train_data:
    :return: 特征向量x，结果y
    '''
    y = train_data['Soil Moisture'].tolist()
    col = train_data.columns.drop(["OBJECTID", "ID" , 'Latitude' , 'Longitude' , 'Latitude_D' , 'Longitude_'  , 'Soil Moisture'])
    x = train_data[col].values  # 剩下的列作为训练数据
    assert len(x) == len(y)
    return x , y

def lgbm_train(train,valid):
    '''
    用gridsearch搜索lgbm的最好参数
    :param train:
    :param valid:
    :return:
    '''
    parameters = {
                'bagging_fraction' : [0.6],
                'bagging_freq': [5],
                'cat_smooth': [1],
                'feature_fraction': [0.95],
                'lambda_l1': [1e-5,1e-3,1e-1,0.0,0.1],
                'lambda_l2': [1e-5,1e-3,1e-1,0.0,0.1],
                'learning_rate': [0.02,0.01,1e-2,1e-3],
                'num_leaves': [15,10,5],
                'subsample': [0.6, 0.7, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 1.0],
                "max_depth" : [30,40,50,60,70,80,90],
            }
    gbm = lgb.LGBMRegressor(
    boosting_type='gbdt',
    num_leaves=30,
    learning_rate=0.1,
    n_estimators=10,
    max_bin=255,
    subsample_for_bin=500,
    objective='regression',
    min_split_gain=0,
    min_child_weight=5,
    min_child_samples=10,
    subsample=1,
    subsample_freq=1,
    colsample_bytree=1,
    reg_alpha=0,
    reg_lambda=0,
    seed=0,
    silent=True,
    huber_delta=1.0,
    fair_c=1.0,
    poisson_max_delta_step=0.7,
    drop_rate=0.1,
    skip_drop=0.5,
    max_drop=50,
    uniform_drop=False,
    xgboost_dart_mode=False,
    verbose_eval=100
)
#有了gridsearch我们便不需要fit函数
    gsearch = GridSearchCV(gbm, param_grid=parameters,  cv=3 , verbose = 1 , n_jobs  = -1 ,scoring = 'r2')
    gsearch.fit(train, valid)

    #print("Best score: %0.3f" % gsearch.best_score_)
    #print("Best parameters set:")
    best_parameters = gsearch.best_estimator_.get_params()
    #for param_name in sorted(parameters.keys()):
    #    print("\t%s: %r" % (param_name, best_parameters[param_name]))


def Kfold_cv(k, X, y, show, V, pol , random_seed = 707):
    '''

    :param k: 分成几个ford
    :param X: 训练特征向量
    :param y: 结果（土壤含水量）
    :param show: 是否显示结果
    :param V: 使用哪种VI，
    :param pol: 使用哪种极化方式
    :param random_seed: 随机种子
    :return: kford验证的rmse的均值，方差，kford验证的r方的均值，方差
    '''
    kf = KFold(n_splits=k, shuffle=True, random_state= random_seed)
    rmse = []
    r2 = []
    oof = np.zeros(len(y))
    fold = 0
    for train_index, test_index in kf.split(X):
        fold += 1
        train_x = X[train_index]
        train_y = y[train_index]
        test_x = X[test_index]
        test_y = y[test_index]

        model = lgb.LGBMRegressor(
            boosting_type='gbdt',
            num_leaves=7,
            max_depth=39,
            learning_rate=0.1,
            n_estimators=100,
            max_bin=255,
            subsample_for_bin=5000,
            objective='regression',
            min_split_gain=0,
            min_child_weight=5,
            min_child_samples=10,
            subsample=1,
            subsample_freq=1,
            colsample_bytree=1,
            reg_alpha=0,
            reg_lambda=0,
            seed=0,
            fair_c=1.0,
            poisson_max_delta_step=0.1,
            drop_rate=0.1,
            skip_drop=0.5,
            max_drop=50,
            uniform_drop=False,
            xgboost_dart_mode=False,
        )
        model.fit(train_x, train_y)
        predict = model.predict(test_x)
        cnt = 0
        for item in test_index:
            oof[item] = predict[cnt]
            cnt += 1

        testing_r2 = r2_score(test_y, predict)
        testing_rmse = np.sqrt(mean_squared_error(predict, test_y))
        if show == True:
            print("-" * 10, "fold_{}".format(fold))
            print("testing r2:", testing_r2)
            print("testing rmse:", testing_rmse)
        rmse.append(testing_rmse)
        r2.append(testing_r2)


        if os.path.exists('save') == False:
            os.mkdir('./save')
        if os.path.exists('save/lgbm_checkpoint') == False:
            os.mkdir('./save/lgbm_checkpoint')

        with open('save/lgbm_checkpoint/lgbm_result{}_{}_{}.pickle'.format(fold, V, pol), 'wb') as f:
            pickle.dump(model, f)

    if show == True:
        print("rmse cv result {}±{}".format(np.mean(rmse), np.std(rmse)))
        print("r2 cv result {}±{}".format(np.mean(r2), np.std(r2)))

    return np.mean(rmse), np.std(rmse), np.mean(r2), np.std(r2)


def get_factors(df, VI_label, polar):
    '''
    获取水云模型的参数
    :param df:
    :param VI_label: 使用哪种VI，目前使用有['EVI', 'PVI', 'RVI', 'SAVI', 'NDWI', 'NDVI']
    :param polar: 使用哪种极化方式，有VV和VH两种
    :return: 水云模型的参数
    '''
    Pi = math.acos(-1)
    X = []
    for index, row in df.iterrows():
        VI = row[VI_label]
        sigma0 = row[polar]
        sec_theta = 1 / math.cos(row['incident angle'] * Pi / 180)

        factors = [sigma0, VI, VI * VI, VI * VI * VI, VI * VI * VI * VI,
                   sigma0 * sec_theta, sigma0 * VI * sec_theta, sigma0 * VI * VI * sec_theta]

        assert len(factors) == 8
        X.append(factors)
    return X

def pred(paras , x):
    '''
    使用计算好的水云模型得到预测的结果
    :param paras: 水云模型的参数
    :param x: 需要预测的特征向量
    :return: 水云模型的预测含水量
    '''
    ret = []
    for row in x:
        ret.append( np.dot(row , paras[1:]) + paras[0])
    return ret

def solve():
    '''
    pipeline
    :return:
    '''
    df_paras = pd.read_csv("stat_linear_regression.csv")
    df = pd.read_excel("newpointdata.xlsx")

    list_rmse_mean = []
    list_rmse_std = []
    list_r2_mean = []
    list_r2_std = []

    for i, row in df_paras.iterrows():
        paras = []
        for j in range(1, 10):
            name = 'k{}'.format(j)
            paras.append(row[name])

        factors_X = get_factors(df, row['VI'], row['polarization'])
        linear_pred = pred(paras, factors_X)

        df['oof'] = linear_pred
        x, y = get_train_val(df)

        x = np.array(x)
        y = np.array(y)
        rmse_mean, rmse_std, r2_mean, r2_std = Kfold_cv(5, x, y, False, row['VI'], row['polarization'])
        list_rmse_mean.append(rmse_mean)
        list_rmse_std.append(rmse_std)
        list_r2_mean.append(r2_mean)
        list_r2_std.append(r2_std)
    df_paras['lgbm_rmse_mean'] = list_rmse_mean
    df_paras['lgbm_rmse_std'] = list_rmse_std
    df_paras['lgbm_r2_mean'] = list_r2_mean
    df_paras['lgbm_r2_std'] = list_r2_std

    if os.path.exists('lgbm_model_result') == False:
        os.mkdir('lgbm_model_result')

    local_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    saving_filename = local_time + "_lgbm_cvresult.csv"

    print(df_paras)
    df_paras.to_csv("stat_lgbm.csv")
    df_paras.to_csv("./lgbm_model_result/" + saving_filename)
    print(df_paras[['rmse-mean', 'lgbm_rmse_mean']])
if __name__ == '__main__':
    solve()