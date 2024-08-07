import scipy.optimize as sco
from itertools import product
import FreeBack as FB
import numpy as np
import pandas as pd

# 投资组合优化模块

class Result():
    # 组合优化结果（returns, weight)
    def __init__(self):
        self.store = {}
    class smallResult():
        def __init__(self, df):
            self.df = df
            self.returns = (self.df['returns']*self.df['weight']).groupby('date').sum()
    def add(self, method, df):
        self.store[method] = self.smallResult(df)
class Opt():
    # 待优化组合, index:(date,code) values: 简单收益率序列
    def __init__(self, ser, interval=20, window=250):
        self.ser = ser
        self.interval = interval
        self.window = window
        # key为优化方法如'max_sharpe'
        self.result = Result()
     # 计算某段时期最优化权重
    def calculate_weights(self, ser, method):
        # 计算组合内股票的期望收益率和协方差矩阵
        pf = ser.unstack().fillna(0)
        # 简单平均/几何平均
        mean_return = [pf.mean()]
        #mean_return = [np.exp(np.log(pf+1).mean())-1]
        cov_matrix = pf.cov().values * 250
        # 计算给定权重下的投资组合表现
        def target(weights, method):
            weights = np.array(weights)
            pred_var = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))  # 组合的期望波动
            pred_return = (np.prod(1+ mean_return*weights)-1)*250  # 组合的期望收益
            # 最小波动
            if method=='最小波动':
                return pred_var
            # 风险平价
            elif method=='风险平价':
                MRC = np.dot(cov_matrix, weights)
                TRC = weights*MRC
                return sum([(i[0]-i[1])**2 for i in list(product(TRC, repeat=2))])
            # 最大收益
            elif method=='最大收益':
                return -pred_return
            # 最大夏普
            elif method=='最大夏普':
                return -pred_return/pred_var
        # 投资组合内股票数目
        num = pf.shape[1]
        if method != '等权':
            def min_func(weights):
                return target(weights, method)
            # 约束是所有参数(权重)的总和为1
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            # 参数值(权重)限制在0和1之间
            bnds = tuple((0, 1) for x in range(num))
            # 调用最优化函数，对初始权重使用平均分布
            opt = sco.minimize(min_func, num * [1. / num], method='SLSQP', bounds=bnds, constraints=cons)
            w = opt['x']
        else:
            w = num * [1. / num]
        return pd.Series(w, index=pf.columns)
    # 每隔inteval天优化持仓权重，当interval为'week','month'时为每周初、每月初优化持仓权重。 使用前window天数据 method 优化方法
    def process_method(self, method='等权'):
        if method=='all':
            for method in ['等权', '最小波动', '风险平价', '最大收益', '最大夏普']:
                self.process_method(method)
        else:
            df = self.ser.fillna(0)
            date_range = df.index.get_level_values(0).unique()
            weights = []
            for i in range(self.window, len(date_range) - 1, self.interval):
                weight = self.calculate_weights(df.loc[date_range[i-self.window:i]], method=method)
                weight = pd.DataFrame(weight)
                weight['date'] = date_range[i-1]
                weight = weight.reset_index().set_index(['date', 'code'])[0]
                weights.append(weight)
            weights = pd.DataFrame(pd.concat(weights)).rename(columns={0:'weight'})
            df = pd.DataFrame(df).join(weights).groupby('code').ffill().dropna()
            self.result.add(method, df)
    # 查看历史表现，比较对象其他方法或者子策略
    def pnl(self, method='等权', compare='sub'):
        if compare=='sub':
            benchmark = self.ser.unstack()
        elif compare=='method':
            benchmark = pd.DataFrame()
            for k,v in self.result.store.items():
                if k!=method:
                    benchmark[k] = v.returns
        post0 = FB.post.ReturnsPost(self.result.store[method].returns, benchmark=benchmark, stratname=method)
        post0.pnl()
    # 查看历史权重
    def weight(self, method='等权'):
        plt, fig, ax = FB.display.matplot() 
        codes = self.result.store[method].df['weight'].index.get_level_values(1).unique()
        all_weights = [self.result.store[method].df['weight'].loc[:, i, :] for i in codes]
        plt.stackplot(self.result.store[method].df.index.get_level_values(0).unique(), all_weights,\
                                labels=codes)
        plt.legend(bbox_to_anchor=(0.5, -0.2), loc=8, ncol=5)
        FB.post.check_output()
        plt.savefig('./output/'+method+'-weight')
        plt.show()