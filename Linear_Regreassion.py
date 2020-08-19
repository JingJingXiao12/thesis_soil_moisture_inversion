import pandas as pd
import numpy as np
import math
from sklearn.model_selection import KFold
import pickle
from sklearn.linear_model import Ridge,LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
import time
import os


def get_factors(df, VI_label, polar):

    '''
    :param df: 训练数据集
    :param VI_label:  属于VH极化还是VV极化
    :param polar:  VH或者VV极化的值
    :return: 水云模型经过泰勒展开后的8个因子
    '''

    Pi = math.acos(-1)

    X = []
    Y = []
    for index, row in df.iterrows():
        VI = row[VI_label]
        sigma0 = row[polar]
        sec_theta = 1 / math.cos(row['incident angle'] * Pi / 180)

        factors = [sigma0, VI, VI * VI, VI * VI * VI, VI * VI * VI * VI,
                   sigma0 * sec_theta, sigma0 * VI * sec_theta, sigma0 * VI * VI * sec_theta]
        result = row['Soil Moisture']

        assert len(factors) == 8

        X.append(factors)
        Y.append(result)

    return X, Y


def Ridge_regression(X, y):
    '''
    岭回归（Ridge Regression）是回归方法 的一种，属于统计方法。在机器学习中也称作权重衰减。也有人称之为Tikhonov正则化。

    岭回归主要解决的问题是两种：一是当预测变量的数量超过观测变量的数量的时候（预测变量相当于特征，观测变量相当于标签），
                        二是数据集之间具有多重共线性，即预测变量之间具有相关性。

    相比于线性回归，岭回归的模型在均方根误差中还加入一项所有参数的平方和来减小所有参数
    :param X: 水云模型经过泰勒展开后的8个因子
    :param y: 土壤含水量
    :return: 训练好的Ridge regression模型
    '''

    model = Ridge(alpha=0.05)
    model.fit(X, y)

    # 这部分代码用来显示岭回归的准确度
    # print('coefs:\n',model.coef_)
    # print('intercept' , model.intercept_)
    # predicted = model.predict(X)

    # print("training r2:" , r2_score(y , predicted))
    # print("training rmse:", np.sqrt(mean_squared_error(predicted,y)) )

    # print(r2_score(y , predicted))
    return model


def linear_regression(X, y):
    '''
    :param X: 水云模型经过泰勒展开后的8个因子
    :param y: 土壤含水量
    :return: 训练好的linear_regression模型
    '''
    model = LinearRegression()
    model.fit(X, y)
    print('coefs:\n', model.coef_)
    print('intercept', model.intercept_)
    predicted = model.predict(X)

    print("training r2:", r2_score(y, predicted))
    print("training rmse:", np.sqrt(mean_squared_error(predicted, y)))
    return model


def Kfold_cv(k, X, y, random_seed=781, save=True):
    '''

    :param k: kford中分成几个fold
    :param X: 水云模型经过泰勒展开后的8个因子
    :param y: 土壤含水量
    :param random_seed: kford分割的随机种子
    :param save: 是否保存结果，默认为保存
    :return: kford的oof，kford验证的rmse的均值，方差，kford验证的r方的均值，方差
    '''
    kf = KFold(n_splits=k, shuffle=True, random_state=random_seed)

    rmse = []
    r2 = []

    oof = np.zeros(len(y))
    fold = 0
    paras = np.zeros(9)
    for train_index, test_index in kf.split(X):
        fold += 1
        train_x = X[train_index]
        train_y = y[train_index]
        test_x = X[test_index]
        test_y = y[test_index]
        model = Ridge_regression(train_x, train_y)
        predict = model.predict(test_x)

        cnt = 0
        for item in test_index:
            oof[item] = predict[cnt]
            cnt += 1

        testing_r2 = r2_score(test_y, predict)
        testing_rmse = np.sqrt(mean_squared_error(predict, test_y))

        rmse.append(testing_rmse)
        r2.append(testing_r2)

        if os.path.exists('save') == False:
            os.mkdir('./save')
        if os.path.exists('save/linear_model_checkpoint') == False:
            os.mkdir('./save/linear_model_checkpoint')

        if save == True:
            with open('save/linear_model_checkpoint/linear_fold{}.pickle'.format(fold), 'wb') as f:
                pickle.dump(model, f)
        w = model.coef_
        b = model.intercept_
        paras[0] += b
        paras[1:] += w

    # print("cv result {}±{}".format(np.mean(rmse) , np.std(rmse)))

    return oof, paras / 5, np.mean(rmse), np.std(rmse), np.mean(r2), np.std(r2)


def cross_validation(df , VIS, POL, random_seed = 781):
    '''

    :param df: 采集好的训练数据
    :param VIS: 使用哪种VI进行训练，如['EVI', 'PVI', 'RVI', 'SAVI', 'NDWI', 'NDVI']
    :param POL: 使用哪种极化后的数据进行训练，选项有vh极化和vv极化
    :param random_seed: kford分割的随机种子
    :return:
    '''
    VIs = []
    POLs = []
    PARAs = []
    RMSEs = []
    R2s = []
    std_rmse = []
    std_r2 = []
    cv_result = pd.DataFrame()
    for V in VIS:
        for P in POL:
            x, y = get_factors(df, V, P)
            x = np.array(x)
            y = np.array(y)

            oof, paras, rmse_mean, rmse_std, r2_mean, r2_std = Kfold_cv(5, x, y, random_seed)
            VIs.append(V)
            POLs.append(P)
            PARAs.append(paras)
            RMSEs.append(rmse_mean)
            R2s.append(r2_mean)
            std_rmse.append(rmse_std)
            std_r2.append(r2_std)

    cv_result['VI'] = VIs
    cv_result['polarization'] = POLs
    cv_result['rmse-mean'] = RMSEs
    cv_result['r2-mean'] = R2s
    cv_result['rmse-std'] = std_rmse
    cv_result['r2-std'] = std_r2
    PARAs = np.array(PARAs)
    for cnt in range(1, 10):
        name = 'k{}'.format(cnt)
        cv_result[name] = PARAs[:, cnt - 1]
    return cv_result

def solve():
    '''
    pipeline
    :return:
    '''
    df = pd.read_excel("./newpointdata.xlsx")
    print((df.head(2)))
    VIS = ['EVI', 'PVI', 'RVI', 'SAVI', 'NDWI', 'NDVI']
    POL = ['sigma0vh', 'sigma0vv']
    cv_result = cross_validation(df , VIS, POL)
    local_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    saving_filename = local_time + "_linear_regression_cvresult.csv"

    print(cv_result)

    if os.path.exists('linear_model_result') == False:
        os.mkdir('linear_model_result')

    cv_result.to_csv('./linear_model_result/' + saving_filename , index=False)
    cv_result.to_csv("./stat_linear_regression.csv" , index = False)

if __name__ == '__main__':
    solve()