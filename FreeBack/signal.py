'''
基于信号的择时框架
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from FreeBack.display import matplot
from FreeBack.post import ReturnsPost as rp

'''
信号生成模块
目前限制: 单一品种
输入: market.index: date
'''
class SignalGenerate():
    def __init__(self, market, return_type='open'):
        self.market = market
        # 预期当天触发信号，次日所获得收益
        if return_type =='open':
            self.sr = market['open'].shift(-2)/market['open'].shift(-1) - 1
        elif return_type == 'close':
            self.sr = market['close'].shift(-1)/market['open'] - 1
        # 存储信号
        self.signals = pd.DataFrame(index=self.market.index)
    

    ### 根据factor 快速展示择时效果
    def post_fast(self, factor, cal_type=0, cal_param=[0.3, 0.7, '100d'],\
                    cut_type=0, cut_param=None ):
        oi_df = self.get_oi_df(factor=factor, cal_type=cal_type, cal_param=cal_param)
        signal = self.get_one_signal(oi_df=oi_df, cut_type=cut_type, cut_param=cut_param)
        self.signal_post(signal=signal)

    
    
    ### 给出因子值，计算进出场信号
    # factor: 因子值，index:date
    # cal_type计算类型:
    # 0: 按照历史分位数, 参数: [0.1, 0.9, '200d'], 最小10%做空, 90%后做多, 最小窗口'200d'
    def get_oi_df(self, factor, cal_type=0, cal_param=[0.3, 0.7, '100d']):
        if cal_type == 0:
            factor = factor.dropna()
            min_date = factor.index[0] + pd.Timedelta(cal_param[2])
            oi_df = pd.DataFrame(index=self.market.index, columns=['lo', 'so'])
            short = factor.index.map(lambda x: factor.loc[x] <= factor.loc[: x].quantile(cal_param[0]))
            oi_df['so'] = pd.Series(short, index=factor.index).loc[min_date:]
            long = factor.index.map(lambda x: factor.loc[x] >= factor.loc[: x].quantile(cal_param[1]))
            oi_df['lo'] = pd.Series(long, index=factor.index).loc[min_date:]
        else:
            print('cal_type格式错误')
        oi_df = oi_df.dropna()
        return oi_df



    ### 给出进场出场条件，生成信号
    # 信号： 当天收盘后触发、次日执行
    # oi_df: index: date, columns:[lo, so, lc, sc](多开，空开，多平，空平)
    # cut_type止损类型： None：没有止损, fix: 持有固定天数
    def get_one_signal(self, oi_df, cut_type=0, cut_param=None):
        if set(oi_df.columns) == set(('lo', 'so')): # 默认不主动平仓
            oi_df['lc'] = oi_df['lo']
            oi_df['sc'] = oi_df['so']
        elif set(oi_df.columns) != set(['lo', 'so', 'lc', 'sc']):
            print('输入格式错误，程序终止')
            return None
        if cut_type == 0:
            signal = self.compose_signal1(df=oi_df)
        elif cut_type == 1:
            signal = self.compose_signal2(df=oi_df, n=cut_param)
        else:
            print('输入止损格式错误，程序终止')
            return None
        return signal


    # 合成信号方式一
    # 所有持仓均按照lc, sc执行平仓
    def compose_signal1(self, df):
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
    def compose_signal2(self, df, n, price_type='open'):
        if price_type == 'open':
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



    # 展示模块，未来放入Post中
    def signal_post(self, signal):
        sr = self.sr
        sr = sr*signal
        sr = sr.shift(1)
        post = rp(returns=sr)
        post.pnl()
        return post


        
    
