'''
基于信号的择时框架
'''

import pandas as pd
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from FreeBack.display import matplot
from FreeBack.post import ReturnsPost
from FreeBack.my_pd import parallel_group

'''
信号生成模块->根据因子生成信号
输入: market.index: date code;  columns:open, low, high, close
factor, Series, index: date code
'''
class SignalGenerate():
    def __init__(self, market, return_type='open'):
        self.market = market
        # 预期当天触发信号，次日所获得收益
        if return_type =='open':
            self.sr = market.groupby('code')['open'].apply(lambda x: x.shift(-2)/x.shift(-1) - 1).droplevel(0)
        elif return_type == 'close':
            self.sr = market.groupby('code')['close'].apply(lambda x: x.shift(-1)/x - 1).droplevel(0)
        elif return_type == 'overnight':
            self.sr = market.groupby('code').apply(lambda x: x['open'].shift(-1)/x['close'] - 1).droplevel(0)
        elif return_type == 'inday':
            self.sr = market.groupby('code').apply(lambda x: x['close'].shift(-1)/x['open'].shift(-1) - 1).droplevel(0)
        self.sr = self.sr.dropna()

    ### 快速计算并展示因子的择时效果
    def fast_post(self, factor, cal_type=0, cal_param=[0.3, 0.7,  '9999d','365d'], \
                  cut_type=0, cut_param=None, n_core=10):
        io_df = self.get_oi_df(factor=factor, cal_type=cal_type, cal_param=cal_param)
        signals = self.get_signals(io_df=io_df, cut_type=cut_type, cut_param=cut_param, n_core=n_core)
        pos = SignalPost(signals=signals, sr=self.sr)
        pos.position_post(compose_type=0)
    

    ### 给出因子值，计算进出场信号
    # factor: 因子值，index:date
    # cal_type计算类型:
    # 0: 按照历史分位数, 参数: [0.1, 0.9, '9999d', '200d'], 最小10%做空, 90%后做多, 窗口9999d,最小窗口'200d'
    # 1: 绝对数值, 参数：[a, b, factor2, c. d], 小于a空开, 小于b多开， factor2平仓因子，None表示没有平仓因子
    def get_oi_df(self, factor, cal_type=0, cal_param=[0.3, 0.7,  '9999d','365d']):
        def fun0(x):  # 历史分位数类型
            x = x.set_index('date')
            min_date = x.index.min() + pd.Timedelta(cal_param[3])
            x['short'] = x['mo11'].rolling(cal_param[2]).quantile(cal_param[0])
            x['long'] = x['mo11'].rolling(cal_param[2]).quantile(cal_param[1])
            x = x.loc[min_date: ].reset_index().set_index(['date', 'code'])
            return  x

        factor = factor.dropna()
        fname = factor.name
        if cal_type == 0:
            factor = factor.dropna()
            f = factor.reset_index().groupby('code')[['date', 'code', fname]].apply(lambda x: fun0(x)).droplevel(0)
            io_df = pd.DataFrame(index=self.market.index, columns=['lo', 'so'])
            io_df['lo'] = f['long'] < f[fname]
            io_df['so'] = f['short'] > f[fname]
        elif cal_type == 1:
            lo = factor >= cal_param[1]
            so = factor <= cal_param[0]
            if cal_param[2] is None:
                io_df = pd.DataFrame(index=self.market.index, columns=['lo', 'so'])
            else:
                io_df = pd.DataFrame(index=self.market.index, columns=['lo', 'so', 'sc', 'lc'])
                factor2 = cal_param[2]
                io_df['sc'] = factor2 <= cal_param[3]
                io_df['lc'] = factor2 >= cal_param[4]
            io_df['so'] = so
            io_df['lo'] = lo
        else:
            print('cal_type格式错误')
        io_df = io_df.dropna()
        return io_df


    ### 给出进场出场条件，生成信号
    # 信号： 当天收盘后触发、次日执行
    # oi_df: index: date, columns:[lo, so, lc, sc](多开，空开，多平，空平)
    # cut_type止损类型： None：没有止损, fix: 持有固定天数
    def get_signals(self, io_df, cut_type=0, cut_param=None, n_core=10):
        def fun(x):
            return self.get_one_signal(io_df=x, cut_type=cut_type, cut_param=cut_param)
        return parallel_group(io_df, func=fun, n_core=n_core, sort_by='code')

    def get_one_signal(self, io_df, cut_type=0, cut_param=None):
        if cut_type == 0:
            if set(io_df.columns) == set(('lo', 'so')): # 默认按照因子值平仓
                io_df['lc'] = io_df['lo']
                io_df['sc'] = io_df['lo']
            signal = self.compose_io1(df=io_df)
        elif cut_type == 1:
            if set(io_df.columns) == set(('lo', 'so')): # 默认不主动平仓
                io_df['lc'] = False
                io_df['sc'] = False
            signal = self.compose_io2(df=io_df, n=cut_param)
        else:
            print('输入止损格式错误，程序终止')
            return None
        return signal


    # 合成信号方式一
    # 所有持仓均按照lc, sc执行平仓
    def compose_io1(self, df):
        last_signal = 0
        for t in df.index:
            if last_signal == 0:
                if df.loc[t, 'lo']:
                    df.loc[t, 'signal'] = 1
                elif df.loc[t, 'so']:
                    df.loc[t, 'signal'] = -1
                else:
                    df.loc[t, 'signal'] = 0
            elif last_signal == 1:
                if df.loc[t, 'lc']:
                    if df.loc[t, 'so']:
                        df.loc[t, 'signal'] = -1
                    else:
                        df.loc[t, 'signal'] = 0
                else:
                    df.loc[t, 'signal'] = 1
            elif last_signal == -1:
                if df.loc[t, 'sc']:
                    if df.loc[t, 'lo']:
                        df.loc[t, 'signal'] = 1
                    else:
                        df.loc[t, 'signal'] = 0
                else:
                    df.loc[t, 'signal'] = -1
            last_signal = df.loc[t, 'signal']
        return df['signal']


    # 合成信号方式二
    # 只能仓位是1， -1， 0
    # 按照最大回撤止损, 止损后空仓3天
    def compose_io2(self, df, n):
        sr = self.sr
        last_signal = 0
        in_date = None
        cut_flag = 0
        for t in df.index:
            if last_signal == 0:  # 上次空仓情况
                if cut_flag > 0: # 止损空仓期
                    df.loc[t, 'signal'] = 0
                    cut_flag += 1
                    if cut_flag == 3:
                        cut_flag = 0
                else:
                    if df.loc[t, 'lo']:
                        df.loc[t, 'signal'] = 1
                        in_date = t
                    elif df.loc[t, 'so']:
                        df.loc[t, 'signal'] = -1
                        in_date = t
                    else:
                        df.loc[t, 'signal'] = 0
            ## 在场情况
            else: # 上次有持仓
                ## 止损情况，空仓三天
                net = (1 + last_signal*sr.loc[in_date: t]).prod()
                if net < 1-n:
                    df.loc[t, 'signal'] = 0
                    in_date = None
                    cut_flag = 1
                else: ## 未触发止损情况
                    if last_signal == 1:
                        if df.loc[t, 'lc']:
                            if df.loc[t, 'so']:
                                in_date = t
                                df.loc[t, 'signal'] = -1
                            else:
                                df.loc[t, 'signal'] = 0
                        else:
                            df.loc[t, 'signal'] = 1
                    elif last_signal == -1:
                        if df.loc[t, 'sc']:
                            if df.loc[t, 'lo']:
                                in_date = t
                                df.loc[t, 'signal'] = 1
                            else:
                                df.loc[t, 'signal'] = 0
                        else:
                            df.loc[t, 'signal'] = -1
            last_signal = df.loc[t, 'signal']
        return df['signal']



# sr 次日可以获取的收益率: Series, index: date or index: date code
# signals: 次日持仓信号
class SignalPost():
    def __init__(self, signals, sr, comm=2/1e4):
        self.signals = signals
        self.sr = sr
        self.comm = comm

    # 根据信号计算持仓df
    # compose_type信号组合方式
    # 0: 默认满仓等权
    def get_position(self, compose_type=0):
        if compose_type == 0:
            postion_df = self.signals.unstack().fillna(0)
            postion_df = postion_df.div(postion_df.abs().sum(axis=1), axis=0)
        else:
            print('输入compose_type错误')
        self.position_df = postion_df

    def position_post(self, compose_type=0):
        self.get_position(compose_type=compose_type)
        self.turnover = (self.position_df - self.position_df.shift(1)).fillna(0)
        sr_df = self.sr.unstack()*self.position_df - self.turnover*self.comm
        sr_compose = sr_df.loc[self.turnover.index].sum(axis=1)

        ReturnsPost(returns=sr_compose).pnl()

