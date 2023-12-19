import pandas as pd
import numpy as np
from numpy_ext import rolling_apply
import copy, math

# 关于pandas经常使用到的一些操作 以及对现有函数的改进

# dataframe, 查找的列， 为value时删除，包括np.nan 
# return 操作后df 与被删除的df
def drop_row(df, col, value_list):
    drop_all = pd.DataFrame(columns = df.columns)
    for value in value_list:
        # nan特殊处理
        if type(value) == type(np.nan):
            if np.isnan(value):
                # 重新排序
                df = df.reset_index(drop = True)
                beishanchu = df[df[col].isnull() == True]
                df = df.drop(df.index[df[col].isnull() == True].values)
                df = df.reset_index(drop = True)
        else:
            df = df.reset_index(drop = True)
            beishanchu = df[df[col] == value]
            df = df.drop(df.index[df[col] == value].values)
            df = df.reset_index(drop = True)
        drop_all = pd.concat([drop_all, beishanchu], ignore_index=True)
    return df, drop_all

# merge的改进， 避免列被重新命名(新加的为a_，并且保持on的列不变
def nmerge(df, df_add, on):
    new_name = ['a_' + x for x in df_add.columns]
    new_name = dict(zip(df_add.columns,new_name))
    df_add = df_add.rename(columns = new_name)
    # 将 on 列 改回
    new_name = ['a_' + x for x in on]
    new_name = dict(zip(new_name,on))
    # on
    df_add = df_add.rename(columns = new_name)

    # merge
    return df.merge(df_add, on=on)

# 将df中所有行按照年份划分，返回一个列表包含各年度的行，从最早开始
def divide_row(df):
    first_year = df['date'].min().year
    last_year = df['date'].max().year
    result = []
    for y in range(first_year, last_year+1):
        select = list(map(lambda x: x.year==y, df['date']))
        df_ = df[select]
        result.append(df_)
    return result

# 按sortby排序，使用unique_id列第一次出现的行组成新的df
def extract_first(df, unique_id = 'thscode', sortby = 'date'):
    df_result = pd.DataFrame(columns = df.columns)
    df = df.sort_values(by = sortby)
    for i in df[unique_id].unique():
        # 默认按时间第一行
        df_ = df[df[unique_id]==i].iloc[0]
        df_ = df_.to_frame().T
        df_result = pd.concat([df_result, df_], ignore_index = True)
    
    return df_result

# 按unique_col将所有combine_col的值合并为所有values的list
def combine_row(df, unique_col='date', combine_col='order_status'):
    df_result = pd.DataFrame(columns = list(df.columns))
    for date in list(df[unique_col].unique()):
        list_status = list(df[df[unique_col] == date][combine_col].values)
        df_ = pd.DataFrame({unique_col:date, combine_col:[list_status]})
        df_result = pd.concat([df_result, df_])
    return df_result

# x, y 自变量与因变量的列名（单自变量）
# 返回DataFrame  0，1分别为回归系数与截距
def rolling_reg(df, x_name, y_name, n):
    # 如果这一字段的df长度小于n，直接返回nan,对应index
    if df.shape[0]<n:
        result_nan = np.ones(df.shape[0])*np.nan
        result = pd.Series(result_nan, df.index)
        return result
    # 回归的x和y
    x = df[x_name]
    y = df[y_name]
#    def func(x, y):
#        # 可能出现连续0值造成错误，这时使用np.nan填充
#        try:
#            result = np.polyfit(x, y, deg=1)
#        except:
#            # 一元回归情况
#            result = np.ones(2)*np.nan
#        return result
    def func(x, y):
        lxx = ((x-x.mean())**2).sum()
        #lyy = ((y-y.mean())**2).sum()
        lxy = ((x-x.mean())*(y-y.mean())).sum()
    # 斜率与截距
        beta = lxy/lxx
        alpha = y.mean() - beta*x.mean()
    # Sum of Reg/Error/Total  r2
        #SSE2 = ((y-(alpha+x*beta))**2).sum()
        #SSR2 = ((alpha+x*beta - y.mean())**2).sum()
        #SST2 = SSR2 + SSE2
        #r2 = SSR2/SST2 
        r = x.corr(y)
    # corr**2  = r2 一元线性回归
        #corr = x.corr(y)
        return beta, alpha, r 
    result = rolling_apply(func, n, x, y)
# 添加index
    result = pd.DataFrame(result, index=df.index)
    return result

# 并行计算   可以保留顺序
def parallel(df, func, n_core=12):
    from joblib import Parallel, delayed
    len_df = len(df)
    sp = list(range(len_df)[::int(len_df/n_core+0.5)])[:-1] # 最后一个节点改为末尾
    sp.append(len_df)
    slc_gen = (slice(*idx) for idx in zip(sp[:-1],sp[1:]))
    results = Parallel(n_jobs=n_core)(delayed(func)(df[slc]) for slc in slc_gen)
    return pd.concat(results)

def parallel_group(df, func, n_core=12, sort_by='code'):
    from joblib import Parallel, delayed
    results = Parallel(n_jobs=n_core)(delayed(func)(group) for name, group in df.groupby(sort_by))
    return pd.concat(results)



# 行情数据的常用计算
# 数据格式为 mutiindex (date,code)   close...



# ta-lib
# SMA 简单移动平均
# EMA 指数移动平均
# WMA 加权移动平均
# DEMA 双指数移动平均
# TEMA 三重指数移动平均


# 计算时间序列值
# 数据df，函数名(Max, Min, Skew, Kurt, MA, Std)
# 作用字段，滚动时间窗口长度，是否并行，并行核数
# rolling输入series而不是df速度会明显提升
def cal_TS(df, func_name='Max', cal='close', period=20, parallel=False, n_core=12):
    df = copy.deepcopy(df)
# inde必须为 'code'和'date'，并且code内部的date排序
    df = df.reset_index().sort_values(by='code').set_index(['code', 'date']).sort_index(level=['code', 'date'])
    new_col = cal + '_' + func_name + '_' + str(period)
    if parallel:
        def func(df):
            if func_name=='Max':
                return df[cal].rolling(period, min_periods=1).max()
            elif func_name=='Min':
                return df[cal].rolling(period, min_periods=1).min()
            elif func_name=='Skew':
                return df[cal].rolling(period, min_periods=1).skew()
            elif func_name=='Kurt':
                return df[cal].rolling(period, min_periods=1).kurt()
            elif func_name=='MA':
                return df[cal].rolling(period, min_periods=1).mean()
            elif func_name=='Std':
                return df[cal].rolling(period, min_periods=1).std()
            elif func_name=='Sum':
                return df[cal].rolling(period, min_periods=1).sum()
            elif func_name=='Zscore':
                return (df[cal]-df[cal].rolling(period, min_periods=1).mean())/df[cal].rolling(period, min_periods=1).std().replace(0, np.nan)
            elif func_name=='HV':
                returns = (df[cal]/(df[cal].shift())).apply(lambda x: np.log(x))
                return np.exp(returns.rolling(period, min_periods=1).std() * np.sqrt(252))-1
        df[new_col] =  parallel_group(df, func, n_core=n_core).values
    else:
        if func_name=='Max':
            df[new_col] =  df.groupby('code', sort=False)[cal].rolling(period, min_periods=1).max().values
        elif func_name=='Min':
            df[new_col] =  df.groupby('code', sort=False).rolling(period, min_periods=1)[cal].min().values
        elif func_name=='Skew':
            df[new_col] =  df.groupby('code', sort=False).rolling(period, min_periods=1)[cal].skew().values
        elif func_name=='Kurt':
            df[new_col] =  df.groupby('code', sort=False).rolling(period, min_periods=1)[cal].kurt().values
        elif func_name=='MA':
            df[new_col] =  df.groupby('code', sort=False).rolling(period, min_periods=1)[cal].mean().values
        elif func_name=='Std':
            df[new_col] =  df.groupby('code', sort=False).rolling(period, min_periods=1)[cal].std().values
        elif func_name=='Sum':
            df[new_col] =  df.groupby('code', sort=False).rolling(period, min_periods=1)[cal].sum().values
        elif func_name=='Zscore':
            df[new_col] =  (df[cal].values - df.groupby('code', sort=False)[cal].rolling(period, min_periods=1).mean().values)\
                             /df.groupby('code', sort=False)[cal].rolling(period, min_periods=1).std().replace(0, np.nan).values
            df[new_col] = df[new_col].fillna(0)
        elif func_name=='HV':
            df['returns'] = (df[cal]/(df[cal].shift())).apply(lambda x: np.log(x))
            df[new_col] =  np.exp(df.groupby('code', sort=False)['returns'].rolling(period, min_periods=1).std().values * np.sqrt(252))-1

# 将index变回 date code
    df = df.reset_index().sort_values(by='date').set_index(['date', 'code'])
    return df

# 获得df中x_name列为自变量 y_name列为因变量的线性回归结果 
def cal_reg(df, x_name, y_name, n, parallel=False, n_core=12):
    df = copy.deepcopy(df)
    # inde必须为 'code'和'date'，并且code内部的date排序
    df = df.reset_index()
    df = df.sort_values(by='code')
    df = df.set_index(['code','date'])
    df = df.sort_index(level=['code','date'])
    # 命名规则
    name_beta = x_name + '-' + y_name + '--beta' +str(n)
    name_alpha = x_name + '-' + y_name + '--alpha'+str(n)
    name_r = x_name + '-' + y_name + '--r'+str(n)
    if parallel:
        def func(df):
            return rolling_reg(df.reset_index('code'), x_name, y_name, n)
        df[[name_beta, name_alpha, name_r]] =  parallel_group(df, func, n_core=n_core).values
    else:
        # 回归    去掉二级index中的code
        df_reg = df.groupby('code', sort=False).apply(lambda df: rolling_reg(df.reset_index('code'), x_name, y_name, n))
        df[[name_beta,name_alpha, name_r]] = df_reg
    # 将index变回 date code
    df = df.reset_index()
    df = df.sort_values(by='date')
    df = df.set_index(['date','code'])
    df = df.sort_index(level=['date','code']) 

    return df

def cal_CrossReg(df_, x_name, y_name, series=False):
    df = copy.copy(df_)
    name = y_name + '-' + x_name + '--alpha'
    beta = df.groupby('date').apply(lambda x: ((x[y_name]-x[y_name].mean())*(x[x_name]-x[x_name].mean())).sum()/((x[x_name]-x[x_name].mean())**2).sum())
    gamma = df.groupby('date').apply(lambda x: x[y_name].mean() - beta[x.index[0][0]]*x[x_name].mean())
    r = df.groupby('date').apply(lambda x: np.sqrt(((gamma[x.index[0][0]]+x[x_name]*beta[x.index[0][0]] - x[y_name].mean())**2).sum()/(((gamma[x.index[0][0]]+x[x_name]*beta[x.index[0][0]] - x[y_name].mean())**2).sum() + ((x[y_name]-(gamma[x.index[0][0]] + x[x_name]*beta[x.index[0][0]]))**2).sum()))) 

    if series:
        return df, beta, gamma, r
    else:
        df[name] = df.groupby('date').apply(lambda x: x[y_name] - beta[x.index[0][0]]*x[x_name] - gamma[x.index[0][0]]).values
        return df