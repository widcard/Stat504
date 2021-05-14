import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('Real estate.csv')
df.info(); df.head()


def outlier_test(data, column, method=None, z=2):
    """ 以某列为依据，使用 上下截断点法 检测异常值(索引) """
    """ 
    full_data: 完整数据
    column: full_data 中的指定行，格式 'x' 带引号
    return 可选; outlier: 异常值数据框 
    upper: 上截断点;  lower: 下截断点
    method：检验异常值的方法（可选, 默认的 None 为上下截断点法），
            选 Z 方法时，Z 默认为 2
    """
    # ================== 上下截断点法检验异常值 ==============================
    if method == None:
        print(f'以 {column} 列为依据，使用 上下截断点法(iqr) 检测异常值...')
        print('=' * 70)
        # 四分位点；这里调用函数会存在异常
        column_iqr = np.quantile(data[column], 0.75) - np.quantile(data[column], 0.25)
        # 1，3 分位数
        (q1, q3) = np.quantile(data[column], 0.25), np.quantile(data[column], 0.75)
        # 计算上下截断点
        upper, lower = (q3 + 1.5 * column_iqr), (q1 - 1.5 * column_iqr)
        # 检测异常值
        outlier = data[(data[column] <= lower) | (data[column] >= upper)]
        print(f'第一分位数: {q1}, 第三分位数：{q3}, 四分位极差：{column_iqr}')
        print(f"上截断点：{upper}, 下截断点：{lower}")
        return outlier, upper, lower
    # ===================== Z 分数检验异常值 ==========================
    if method == 'z':
        """ 以某列为依据，传入数据与希望分段的 z 分数点，返回异常值索引与所在数据框 """
        """ 
        params
        data: 完整数据
        column: 指定的检测列
        z: Z分位数, 默认为2，根据 z分数-正态曲线表，可知取左右两端的 2%，
           根据您 z 分数的正负设置。也可以任意更改，知道任意顶端百分比的数据集合
        """
        print(f'base on {column} column，use Z score，z score {z} to detect outlier...')
        print('=' * 70)
        # 计算两个 Z 分数的数值点
        mean, std = np.mean(data[column]), np.std(data[column])
        upper, lower = (mean + z * std), (mean - z * std)
        print(f" greater than {upper} less than {lower} can be treated as outlier。")
        print('=' * 70)
        # 检测异常值
        outlier = data[(data[column] <= lower) | (data[column] >= upper)]
        return outlier, upper, lower

outlier, upper, lower = outlier_test(data=df, column='Y house price of unit area', method='z')
print(outlier.info())
df.drop(index=outlier.index, inplace=True)
df.to_csv('Real estate.csv')
