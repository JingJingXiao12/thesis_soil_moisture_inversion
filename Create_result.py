import cv2
import os
import numpy as np
import pandas as pd
np.set_printoptions(suppress=True)
import pickle
import math
from PIL import Image



def load_tif_files():
    '''
    读取tif图片
    :return:
    '''
    root = "./data/clipnewdata/"
    pics = []
    names = []
    for filename in os.listdir(root):
        if filename.endswith("tif"):
            full_path = os.path.join(root, filename)
            pic = cv2.imread(full_path, 2)
            pics.append(pic)
            names.append(filename.replace(".tif", ''))

    print('used features = ' , names)

    n = pics[0].shape[0]
    m = pics[0].shape[1]
    mp = {}
    for i, pic in enumerate(pics):
        mp[names[i]] = pic

    return n , m , mp

def linear_model_predict(VI, VV, paras, n , m , mp):
    '''
    使用已经计算好的线性模型预测整张图的结果
    :param VI: 使用哪种VI
    :param VV: 使用哪种极化方式
    :param paras: 水云模型的参数
    :param n: 图片的横坐标大小
    :param m: 图片的纵坐标大小
    :param mp: 读入的tif图片
    :return: 用水云模型预测的土壤含水量
    '''
    Pi = math.acos(-1)
    result = np.zeros((n, n))
    pic_VI = mp[VI]
    pic_incidentan = mp['incidentangle']
    pic_sigma0 = mp[VV]
    stat = []
    for i in range(n):
        for j in range(m):
            VI = pic_VI[i][j]
            sec_theta = 1 / math.cos(pic_incidentan[i][j] * Pi / 180)
            sigma0 = pic_sigma0[i][j]
            factors = [sigma0, VI, VI * VI, VI * VI * VI, VI * VI * VI * VI,
                       sigma0 * sec_theta, sigma0 * VI * sec_theta, sigma0 * VI * VI * sec_theta]
            factors = np.array(factors).reshape(1, -1)
            pred = np.dot(factors, paras[1:]) + paras[0]
            result[i][j] = pred
            stat.append(pred)
    return result, stat


def get_linear_model_result( n , m , mp , df_paras):
    '''
    批量生成水云模型的土壤含水量结果
    :param n: 图片的横坐标大小
    :param m: 图片的纵坐标大小
    :param mp: 读入的tif图片
    :param df_paras: 水云模型的参数
    :return: 生成的土壤含水量的一些统计结果，土壤含水量图
    '''
    stat_min = []
    stat_max = []
    stat_mean = []

    oof_results = []
    for i, row in df_paras.iterrows():
        paras = []
        for j in range(1, 10):
            name = 'k{}'.format(j)
            paras.append(row[name])
        print(row['VI'], row['polarization'])
        result, stat = linear_model_predict(row['VI'], row['polarization'], paras, n , m , mp)
        oof_results.append(result)
        im = Image.fromarray(result)


        if os.path.exists('linear_model_result') == False:
            os.mkdir('linear_model_result')

        if os.path.exists('linear_model_result/pics') == False:
            os.mkdir('linear_model_result/pics')

        im.save('linear_model_result/pics/linear_reg_result_{}_{}.tif'.format(row['VI'], row['polarization']))

        stat_min.append(np.min(stat))
        stat_max.append(np.max(stat))
        stat_mean.append(np.mean(stat))

    df_paras['stat_min'] = stat_min
    df_paras['stat_max'] = stat_max
    df_paras['stat_mean'] = stat_mean
    df_paras.to_csv("stat_linear_regression.csv")

    return df_paras , oof_results


def predict_lgbm( n , m , mp , r, V, pol):
    '''
    使用梯度提升树得到的土壤含水量预测结果
    :param n: 图片的横坐标大小
    :param m: 图片的纵坐标大小
    :param mp: 读入的tif图片
    :param r: 水云模型的计算结果
    :param V: 使用哪种VI
    :param pol: 使用哪种极化方式
    :return: lgbm预测的土壤含水量图
    '''
    cols = ['incidentangle', 'sigma0vh', 'sigma0vv', 'DEM', 'Solar Radiation',
            'Sunshine Duration', 'Surface Roughness',
            'Relief Degree of Land Surface', 'Temperature', 'NDVI', 'RVI', 'EVI',
            'PVI', 'NDWI', 'SAVI', 'Elevation Difference', 'aspect', 'Slope', 'DSM',
            'oof_watercloud']

    mp['oof_watercloud'] = r
    all_result = np.zeros((n, n))
    for fold in range(1, 6):
        lgb_model_name = 'save/lgbm_checkpoint/lgbm_result{}_{}_{}.pickle'.format(fold, V, pol)
        with open(lgb_model_name, 'rb') as f:
            model = pickle.load(f)
            result = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    factors = []
                    for feature in cols:
                        f = mp[feature][i][j]
                        factors.append(f)
                    factors = np.array(factors).reshape(1, -1)
                    result[i][j] = model.predict(factors)
        print(np.min(result), np.max(result), np.mean(result))
        all_result += result
    all_result /= 5
    return all_result

def get_lgbm_result(n , m , mp , df_paras , oof_results):
    '''
    批量生成梯度提升树的结果
    :param n: 图片的横坐标大小
    :param m: 图片的纵坐标大小
    :param mp: 读入的tif图片
    :param df_paras: 水云模型的参数
    :param oof_results: 水云模型结果图
    :return:
    '''
    for i, row in df_paras.iterrows():
        VI = row['VI']
        pol = row['polarization']
        result_lgbm = predict_lgbm( n , m , mp , oof_results[i], VI, pol)

        print(VI, pol)
        im = Image.fromarray(result_lgbm)

        if os.path.exists('lgbm_model_result') == False:
            os.mkdir('lgbm_model_result')

        if os.path.exists('lgbm_model_result/pics') == False:
            os.mkdir('lgbm_model_result/pics')

        im.save('lgbm_model_result/pics/lgbm_result_{}_{}.tif'.format(VI, pol))

def solve():
    n , m , mp = load_tif_files()
    df_paras = pd.read_csv("stat_linear_regression.csv")
    df_paras , oof_results = get_linear_model_result(n,m,mp , df_paras)
    print("linear model stat:")
    print(df_paras)

    get_lgbm_result(n , m , mp , df_paras , oof_results)


if __name__ == '__main__':
    solve()