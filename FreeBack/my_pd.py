import pandas as pd
import numpy as np
import statsmodels.api as sm
from numpy_ext import rolling_apply
from joblib import Parallel, delayed
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

def rolling_corr(df, x_name, y_name, n):
    # 如果这一字段的df长度小于n，直接返回nan,对应index
    if df.shape[0]<n:
        result_nan = np.ones(df.shape[0])*np.nan
        result = pd.Series(result_nan, df.index)
        return result
    # 相关性的x和y
    x = df[x_name]
    y = df[y_name]
    def func(x, y):
        r = x.corr(y)
        return r 
    result = rolling_apply(func, n, x, y)
# 添加index
    result = pd.DataFrame(result, index=df.index)
    return result

# 并行计算df, func为对df/ser的操作,返回df/ser 可以保留顺序
def parallel(df, func, n_core=12):
    len_df = len(df)
    sp = list(range(len_df)[::int(len_df/n_core+0.5)])[:-1] # 最后一个节点改为末尾
    sp.append(len_df)
    slc_gen = (slice(*idx) for idx in zip(sp[:-1],sp[1:]))
    results = Parallel(n_jobs=n_core)(delayed(func)(df[slc]) for slc in slc_gen)
    return pd.concat(results)

# 并行group
def parallel_group(df, func, n_core=12, sort_by='code'):
    results = Parallel(n_jobs=n_core)(delayed(func)(group) for name, group in df.groupby(sort_by))
    return pd.concat(results)

# 行情数据的常用计算
# 数据格式为 mutiindex (date,code)   close...

# 输入series返回series
# 有并行选项的函数默认并行
# a 函数可选参数


def cal_ts(ser, func_name='Max', period=20, a=1, parallel=True, n_core=12):
    if func_name=='MA':
        return ser.groupby(level='code').apply(lambda x: x.droplevel( 'code').rolling(\
            period, min_periods=1).mean()).reset_index().sort_values(by='date').\
                set_index(['date', 'code']).loc[ser.index].iloc[:,0]
    elif func_name=='EMA':
        return ser.groupby(level='code').apply(lambda x: x.droplevel( 'code').ewm(\
            span=period, min_periods=1).mean()).reset_index().sort_values(by='date').\
                set_index(['date', 'code']).loc[ser.index].iloc[:,0]
    elif func_name=='Max':
        return ser.groupby(level='code').apply(lambda x: x.droplevel( 'code').rolling(\
            period, min_periods=1).max()).reset_index().sort_values(by='date').\
                set_index(['date', 'code']).loc[ser.index].iloc[:,0]
    elif func_name=='Min':
        return ser.groupby(level='code').apply(lambda x: x.droplevel( 'code').rolling(\
            period, min_periods=1).min()).reset_index().sort_values(by='date').\
                set_index(['date', 'code']).loc[ser.index].iloc[:,0]
    elif func_name=='Std':
        return ser.groupby(level='code').apply(lambda x: x.droplevel( 'code').rolling(\
            period, min_periods=1).std()).reset_index().sort_values(by='date').\
                set_index(['date', 'code']).loc[ser.index].iloc[:,0]
    elif func_name=='Skew':
        return ser.groupby(level='code').apply(lambda x: x.droplevel( 'code').rolling(\
            period, min_periods=1).skew()).reset_index().sort_values(by='date').\
                set_index(['date', 'code']).loc[ser.index].iloc[:,0]
    elif func_name=='Kurt':
        return ser.groupby(level='code').apply(lambda x: x.droplevel( 'code').rolling(\
            period, min_periods=1).kurt()).reset_index().sort_values(by='date').\
                set_index(['date', 'code']).loc[ser.index].iloc[:,0]
    elif func_name=='Sum':
        return ser.groupby(level='code').apply(lambda x: x.droplevel( 'code').rolling(\
            period, min_periods=1).sum()).reset_index().sort_values(by='date').\
                set_index(['date', 'code']).loc[ser.index].iloc[:,0]
    elif func_name=='rank':
        return ser.groupby(level='code').apply(lambda x: x.droplevel( 'code').rolling(\
            period, min_periods=1).rank()).reset_index().sort_values(by='date').\
                set_index(['date', 'code']).loc[ser.index].iloc[:,0] 
    elif func_name=='quantile':
        return ser.groupby(level='code').apply(lambda x: x.droplevel( 'code').rolling(\
            period, min_periods=1).quantile(a)).reset_index().sort_values(by='date').\
                set_index(['date', 'code']).loc[ser.index].iloc[:,0] 
    elif func_name=='Zscore':
        MA = ser.groupby(level='code').apply(lambda x: x.droplevel( 'code').rolling(\
            period, min_periods=1).mean()).reset_index().sort_values(by='date').\
                set_index(['date', 'code']).loc[ser.index].iloc[:,0]
        Std = ser.groupby(level='code').apply(lambda x: x.droplevel( 'code').rolling(\
            period, min_periods=1).std()).reset_index().sort_values(by='date').\
                set_index(['date', 'code']).loc[ser.index].iloc[:,0]
        return ((ser-MA)/Std).fillna(0)
    #elif func_name=='HV':
    #    returns = np.log(ser/ser.groupby(level='code').shift()).fillna(0)
    #    return np.exp(returns.groupby(level='code').apply(lambda x: x.droplevel('code').rolling(\
    #        period, min_periods=1).std()*np.sqrt(250)).reset_index().sort_values(\
    #            by='date').set_index(['date', 'code']).loc[ser.index].iloc[:,0])-1
    elif func_name in ['WMA']:
        if parallel:
            def func(ser):
                return ser.groupby(level='code').apply(lambda x: x.droplevel( 'code').rolling(\
                    period, min_periods=1).apply(lambda x: x[::-1].cumsum().sum()/(period*(1+period)/2)))
            result = parallel_group(ser, func, n_core=n_core)
            return result.reset_index().sort_values(by='date').set_index(['date', 'code']).loc[ser.index].iloc[:,0]
        return ser.groupby(level='code').apply(lambda x: x.droplevel( 'code').rolling(\
            period, min_periods=1).apply(lambda x: x[::-1].cumsum().sum()/(period*(1+period)/2))).reset_index().\
                sort_values(by='date').set_index(['date', 'code']).loc[ser.index].iloc[:,0]
    # 在rolling中无法直接调用的函数只能通过这种方法，速度慢几个数量级
    elif func_name in ['argmin', 'argmax', 'prod']:
        if parallel:
            def func(ser):
                return ser.groupby(level='code').apply(lambda x: x.droplevel( 'code').rolling(\
                    period, min_periods=1).apply(getattr(np, func_name)))
            result = parallel_group(ser, func, n_core=n_core)
            return result.reset_index().sort_values(by='date').set_index(['date', 'code']).loc[ser.index].iloc[:,0]
        return ser.groupby(level='code').apply(lambda x: x.droplevel( 'code').rolling(\
            period, min_periods=1).apply(getattr(np, func_name))).reset_index().sort_values(by='date').set_index(\
                ['date', 'code']).loc[ser.index].iloc[:,0]


# 按照look列的阈值筛选得到cal列，求和
def cal_select_sum(df, key, threshold_left, threshold_right, sum, n, n_core=12):
    # 按条件筛选滚动求和
    def rolling_select_sum(df, key, threshold_left, threshold_right, sum, n):
        # 如果这一字段的df长度小于n，直接返回nan,对应index
        if df.shape[0]<n:
            result_nan = np.ones(df.shape[0])*np.nan
            result = pd.Series(result_nan, df.index)
            return result
        # 相关性的x和y
        x = df[key]
        tl = df[threshold_left]
        tr = df[threshold_right]
        y = df[sum]
        def func(x, tl, tr, y):
            return ((x>=tl.iloc[-1])*(x<=tr.iloc[-1])*y).sum()
        result = rolling_apply(func, n, x, tl, tr, y)
# 添加  index
        result = pd.DataFrame(result, index=df.index)
        return result
    # 按代码并行
    df = copy.deepcopy(df)
    # inde必须为 'code'和'date'，并且code内部的date排序
    df = df.reset_index()
    df = df.sort_values(by='code')
    df = df.set_index(['code','date'])
    df = df.sort_index(level=['code','date'])
    # 命名规则
    name_r = str(sum) + '_' + 'Sum' + str(n) + '-' + str(key) + ',' + str(threshold_left) + ';' + str(threshold_right)
    def func(df):
        return rolling_select_sum(df.reset_index('code'), key, threshold_left, threshold_right, sum, n)
    df[[name_r]] =  parallel_group(df, func, n_core=n_core).values
    # 将index变回 date code
    df = df.reset_index()
    df = df.sort_values(by='date')
    df = df.set_index(['date','code'])
    df = df.sort_index(level=['date','code'])
    return df[name_r]


# 获得df中x_name列为自变量 y_name列为因变量的线性回归结果 
def cal_reg(df, x_name, y_name, n, parallel=True, n_core=12):
    df = copy.deepcopy(df)
    # inde必须为 'code'和'date'，并且code内部的date排序
    df = df.reset_index()
    df = df.sort_values(by='code')
    df = df.set_index(['code','date'])
    df = df.sort_index(level=['code','date'])
    # 命名规则
    name_beta = str(x_name) + '-' + str(y_name) + '--beta' +str(n)
    name_alpha = str(x_name) + '-' + str(y_name) + '--alpha'+str(n)
    name_r = str(x_name) + '-' + str(y_name) + '--r'+str(n)
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

def cal_corr(df, x_name, y_name, n, parallel=True, n_core=12):
    df = copy.deepcopy(df)
    # inde必须为 'code'和'date'，并且code内部的date排序
    df = df.reset_index()
    df = df.sort_values(by='code')
    df = df.set_index(['code','date'])
    df = df.sort_index(level=['code','date'])
    # 命名规则
    name_r = str(x_name) + '-' + str(y_name) + '--r'+str(n)
    if parallel:
        def func(df):
            return rolling_corr(df.reset_index('code'), x_name, y_name, n)
        df[[name_r]] =  parallel_group(df, func, n_core=n_core).values
    else:
        # 回归    去掉二级index中的code
        df_reg = df.groupby('code', sort=False).apply(lambda df: \
                        rolling_corr(df.reset_index('code'), x_name, y_name, n))
        df[[name_r]] = df_reg
    # 将index变回 date code
    df = df.reset_index()
    df = df.sort_values(by='date')
    df = df.set_index(['date','code'])
    df = df.sort_index(level=['date','code']) 
    return df


# x_name list, y_nmae column
# 截面多元线性回归
def cal_CrossReg(df, x_name, y_name, residual=False):
    # 使用sm模块
    result = df.groupby('date', sort=False).apply(lambda d:\
                 sm.OLS(d[y_name], sm.add_constant(d[x_name])).fit())
    
    # 如果d[x_name]中所有数相同为C且不为零，这时params中没有const，x_name为d[y_name].mean()/C
    # rsquared为0
    # 当d[x_name]全为0时，params['const']为0，params[x_name]为d[y_name].mean()
    # rsquared可能为极小的负数
    def func(x, name):
        try:
            return x.params[name]
        except:
            print('sm reg warning')
            return 0
    #if type(x_name)!=type([]):
    r = result.map(lambda x: np.sqrt(abs(x.rsquared)))

    data = []
    index = []
    for i in result.items():
        data.append(dict(i[1].params))
        index.append(i[0])
    params = pd.DataFrame(data, index=index)

    if residual:
        return pd.Series(df.groupby('date').apply(lambda x: x[y_name] - \
                    (params.loc[x.index[0][0]][x_name]*x[x_name]).sum(axis=1) -\
                        params.loc[x.index[0][0]]['const']).values, index=df.index)
    else:
        return params, r
