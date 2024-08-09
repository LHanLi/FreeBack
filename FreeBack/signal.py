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


"""
根据开仓信号和平仓类，分析开仓信号的优劣、得到持仓状态、展示择时效果。
"""





import FreeBack as FB
from plottable import ColumnDefinition, ColDef, Table


# SAR跟踪止损离场
# SAR参数
initAF = deltaAF = 0.005
maxAF = 0.05
class Trail():
    def __init__(self, after_market, direct):
        self.after_market = after_market
        self.direct = direct
    # 初始化，在第一根bar上运行
    def init(self):
        # 运行中全部记录指标
        self.care =  (lambda x: 'high' if x else 'low')(self.direct==1)
        self.edge = self.after_market.iloc[0][self.care]
        # 比如做多的话选取最小值作为初始止损位
        self.SAR = self.after_market.iloc[0][(lambda x: 'low' if x else 'high')(self.direct==1)]
        self.AF = initAF
        self.after_market.loc[self.after_market.index[0], 'SAR'] = self.SAR
    # 沿着date顺序检查,输入为after_market的逐行index和value 
    def check(self, i, v):
        if self.direct*(v[self.care]-self.edge)>0:
            self.edge = v[self.care]
            self.AF = min(self.AF+deltaAF, maxAF)
        self.SAR = self.SAR+self.AF*(self.edge-self.SAR)
        self.after_market.loc[i, 'SAR'] = self.SAR
        # 价格低于（做多）SAR离场
        ifleave = self.direct*(self.after_market.loc[i, 'close']-self.SAR)<0
        return ifleave

class Signal():
    # 开仓信号坐标，开仓方向
    def __init__(self, market, oloc, direct):
        self.market = market
        self.oloc = oloc
        self.direct = direct
    # 信号分析
    def analysis(self):
        self.mean_return = self.result['returns'].mean()
        self.mean_dur = self.result['dur'].mean()
        self.maxdraw = self.result['maxd'].max()
        self.winrate = (self.result['returns']>0).mean()
        self.mean_win = self.result[self.result['returns']>0]['returns'].mean()
        self.mean_loss = self.result[self.result['returns']<0]['returns'].mean()
        self.odds = -self.mean_win/self.mean_loss
        self.potential_odds = (self.result['maxr']/self.result['maxd']).mean()

        col0 = pd.DataFrame(columns=['col0'])
        col0.loc[0] = '信号次数' 
        col0.loc[1] = len(self.oloc)
        col0.loc[2] = '开仓次数' 
        col0.loc[3] = len(self.result)
        col0.loc[4] = '平均持有时长' 
        col0.loc[5] = self.result['dur'].mean().round(1)
        col1 = pd.DataFrame(columns=['col1'])
        col1.loc[0] = '空仓时间占比(%)'
        col1.loc[1] = round(100-100*len(self.result_hold.index.get_level_values(0).unique())/\
                        len(self.market.index.get_level_values(0).unique()), 1)
        col1.loc[2] = '最大重叠信号数' 
        col1.loc[3] = self.result_hold.reset_index().groupby('date').count().max().values[0]
        col1.loc[4] = '平均重叠信号数' 
        col1.loc[5] = self.result_hold.reset_index().groupby('date').count().mean()\
            .round(1).values[0]
        col2 = pd.DataFrame(columns=['col2'])
        col2.loc[0] = '平均收益（万）'  
        col2.loc[1] = self.result['returns'].mean().round(1)
        col2.loc[2] = '最大收益（万）'
        col2.loc[3] = self.result['returns'].max().round(1)
        col2.loc[4] = '最大潜在收益（万）'
        col2.loc[5] = self.result['maxr'].max().round(1)
        col3 = pd.DataFrame(columns=['col3'])
        col3.loc[0] = '平均最大回撤（万）'
        col3.loc[1] = self.result['maxd'].mean().round(1)
        col4 = pd.DataFrame(columns=['col4'])
        col5 = pd.DataFrame(columns=['col5'])
        col6 = pd.DataFrame(columns=['col6'])
        col7 = pd.DataFrame(columns=['col7'])
        df_details = pd.concat([col0, col1, col2, col3, \
                col4, col5, col6, col7], axis=1).fillna('')
        self.df_details = df_details
        plt, fig, ax = FB.display.matplot(w=22)
        column_definitions = [ColumnDefinition(name='col0', group="基本参数"), \
                              ColumnDefinition(name='col1', group="基本参数"), \
                            ColumnDefinition(name='col2', group='收益能力'), \
                            ColumnDefinition(name='col3', group='风险水平'), \
                            ColumnDefinition(name="col4", group='风险调整'), \
                            ColumnDefinition(name="col5", group='风险调整'), \
                            ColumnDefinition(name="col6", group='策略执行'),
                            ColumnDefinition(name="col7", group='业绩持续性分析')] + \
                             [ColDef("index", title="", width=0, textprops={"ha":"right"})]
        tab = Table(self.df_details, row_dividers=False, col_label_divider=False, 
                    column_definitions=column_definitions,
                    odd_row_color="#e0f6ff", even_row_color="#f0f0f0", 
                    textprops={"ha": "center"})
        #ax.set_xlim(2,5)
        # 设置列标题文字和背景颜色(隐藏表头名)
        tab.col_label_row.set_facecolor("white")
        tab.col_label_row.set_fontcolor("white")
        # 设置行标题文字和背景颜色
        tab.columns["index"].set_facecolor("white")
        tab.columns["index"].set_fontcolor("white")
        tab.columns["index"].set_linewidth(0)
        FB.post.check_output()
        plt.savefig('./output/details.png')
        plt.show()
    # 从开仓信号得到信号强度(result)、持仓状态(result_hold)和跟踪指标(result_after)
    def run(self, comm=0):
        result = pd.DataFrame(index=self.oloc)
        result_hold = pd.DataFrame()
        result_after = {}
        for start in self.oloc:
            #print('从', start, '开始')
            if start in result_hold.index:
                #print('开仓信号', start, '在持有状态，忽略。')
                continue
            # 信号触发后的market
            after_market = self.market.loc[start[0]:, start[1], :]
            # 框架
            high =  after_market.iloc[0]['high']
            low = after_market.iloc[0]['low']
            trail0 = Trail(after_market, self.direct)
            trail0.init()
            this_hold = [after_market.index[0]]
            for i,v in after_market.iterrows():
                # 忽略第一根bar
                if i==after_market.index[0]:
                    continue
                high = max(after_market.loc[i, 'high'], high)
                low = min(after_market.loc[i, 'low'], low)
                # 离场信号发出的下一根bar离场
                this_hold.append(i)
                if trail0.check(i, v):
                    break
            this_hold = pd.DataFrame(index=after_market.loc[:i].index)
            returns = self.direct*(1e4*after_market.loc[i, 'close']\
                                   /after_market.iloc[0]['close']-1e4)
            returns = returns-2*comm
            dur = len(this_hold)
            maxr = 1e4*high/after_market.iloc[0]['close']-1e4
            maxd = 1e4-1e4*low/after_market.iloc[0]['close']
            if self.direct==-1:
                maxr,maxd = maxd, maxr
            maxr = maxr-comm
            maxd = maxd+comm
            result.loc[start, ['returns', 'dur', 'maxr', 'maxd']] = [returns, dur, maxr, maxd]
            result_hold = pd.concat([result_hold, this_hold])
            result_after[start] = after_market
        self.result = result.dropna()
        self.result_hold = result_hold
        self.result_after = result_after
    # 观察单标的择时情况，code可以输入代码或者整数当输入整数时展示单次最大收益的代码，\
    # indicators after_market中指标
    def lookcode(self, code=0, indicators=[]):
        if type(code)==type(0):
            code = self.result.sort_values(by='returns', \
                                ascending=False).index.get_level_values(1).unique()[code]
        plt, fig, ax = FB.display.matplot()
        # 跟踪价格，收盘价
        l0, = ax.plot(self.market.loc[:, code, :]['close'])
        # 开仓信号
        ax.scatter(pd.DataFrame(index=self.oloc).loc[:, code, :].index, \
                self.market.loc[:, code, :]['close'].loc[\
                    pd.DataFrame(index=self.oloc).loc[:, code, :].index],\
                      c='C3', s=10, marker='*', alpha=1)
        lines = []
        for date in self.result.loc[:, code, :].index:
            ax.vlines(date, self.market.loc[:, code, :]['close'].min(),\
                       self.market.loc[:, code, :]['close'].max(), colors='C6', linestyle='--')
            for indicator in indicators:
                l, = ax.plot(self.result_after[(date, code)].loc[:, code, :][indicator], c='C1')
                lines.append(l)
        plt.legend([l0]+lines, ['收盘价']+indicators)













