import pandas as pd
import numpy as np
import FreeBack as FB

class Event():
    '''
    事件驱动测试
    参数格式:
    multi——index: date, code
    signal: 入场信号,index格式
    price: series
    before: int, 事件前天数
    after: int, 事件后天数
    bench_type: zero:零, equal:等权
    '''
    def __init__(self, signal, price, before=5, after=30, bench_type='zero',\
                  custom_bench=None, n_core=6, fast=True):
        self.signal = signal
        self.price = price
        self.before = before
        self.after = after
        self.n_core = n_core
        self.bench_type = bench_type
        self.custom_bench = custom_bench
        if fast:
            self.fast_init()
        else:
            self.init_param()
    
    def sr(self, x):
        sr = (x - x.shift(1))/x.shift(1)
        return sr
        
    def init_param(self):
        def fun1(x):
            r = pd.DataFrame()
            for i in cols:
                r.loc[:, i] = x.shift(-i)
            return r

        self.length = self.before + self.after
        self.sr = self.price.groupby('code').apply(lambda x: self.sr(x)).droplevel(0).sort_index(level=0)
        # 基准按照等权计算
        self.bench_sr = self.sr.groupby('date').mean()
        if self.bench_type == 'zero':
            self.bench_sr = pd.Series(index=self.bench_sr.index)
            self.bench_sr.fillna(0, inplace=True)
        elif self.bench_type == 'equal':
            self.bench_sr = self.bench_sr
        if type(self.custom_bench)==type(None):
            self.bench_sr = self.custom_bench.loc[self.bench_sr.index]
        self.sr = self.sr - self.bench_sr
        self.sr.name = 'sr'
        cols = [i-self.before+1 for i in range(self.length)]
        self.signal_sr_df = FB.my_pd.parallel_group(self.sr, fun1, n_core=self.n_core).loc[self.signal]
        self.number = self.signal_sr_df[0].groupby(level='date').count()
        self.bench_net = (self.bench_sr + 1).cumprod()
        self.net = (self.signal_sr_df+1).cumprod(axis=1)

    def fast_init(self):
        self.length = self.before + self.after
        self.sr = (self.price/self.price.groupby('code').shift() - 1).fillna(0)
        # 基准收益率，默认为0
        self.bench_sr = self.sr.groupby('date').mean()
        if self.bench_type == 'zero':
            self.bench_sr = pd.Series(index=self.bench_sr.index)
            self.bench_sr.fillna(0, inplace=True)
        elif self.bench_type == 'equal':
            self.bench_sr = self.bench_sr
        self.sr = self.sr - self.bench_sr
        self.sr.name = 'sr'
        # 前后观察收益率
        cols = [i-self.before+1 for i in range(self.length)]
        signal_sr_df = pd.concat([self.sr.groupby('code').shift(-i) for i in cols], axis=1)
        signal_sr_df.columns = cols
        self.signal_sr_df = signal_sr_df.loc[self.signal].copy()
        self.signal_sr_df.fillna(0, inplace=True)
        # 触发次数
        self.number = self.signal_sr_df[0].groupby(level='date').count()
        # 净值
        self.bench_net = (self.bench_sr + 1).cumprod()
        self.net = (self.signal_sr_df+1).cumprod(axis=1)

    # 每日触发信号数量, bench_type zero时没有bench
    def draw_turnover(self):
        plt0, fig0, ax0 = FB.display.matplot()
        ax1 = ax0.twinx()
        num = self.number
        ax1.plot(num.cumsum(), color='C2', label='累计样本量（右）')
        # 触发次数过多的截断
        index = num[num > (num.mean() + 5*num.std())].index
        num.loc[index] = num.mean() + 5*num.std()
        ax0.bar(num.index, num.values, color='grey', label='每日样本量')
        #if self.bench_type != 'zero':
            #ax1.plot(self.bench_net, color='steelblue', label='基准净值（右）')
        #fig0.legend(bbox_to_anchor=(0.5, 0), loc=10, ncol=2)
        fig0.legend(loc='lower center', ncol=2)
        plt0.show()
    
    # 每日超额, 事件净值(取均值)
    def draw_net(self):
        plt0, fig0, ax0 = FB.display.matplot()
        sr = self.signal_sr_df.mean(axis=0)
        ax0.bar(sr.index, sr.values, width=0.5,  label='单日超额', color='darkgoldenrod')
        ax1 = ax0.twinx()
        net = self.net.mean()
        net = net/net.loc[1]
        ax1.plot(net, color='crimson', label='累计净值（右）', linewidth=2.0)
        ax1.hlines(1, sr.index[0], sr.index[-1], colors='k', linestyles='--')
        fig0.legend(loc='lower center', ncol=2)
        plt0.show()
    
    def draw_Kelly(self, direct='long'):
        # 信号触发后价格变化结果
        trade_result = (self.signal_sr_df.iloc[:, self.before+1:]+1).cumprod(axis=1)-1
        # 胜率
        winrate = (trade_result>0).sum()/len(trade_result.index)
        # 赔率 按最大值
        win = (trade_result*(trade_result>0)).replace(0, np.nan).max()
        loss = (abs(trade_result)*(trade_result<0)).replace(0, np.nan).max()
        odds = win/(loss+win)
        # 根据交易多空修改赔率胜率
        if direct=='short':
            win,loss = loss,win
            odds = win/(loss+win)
            winrate = 1-winrate
        ## Kelly公式确定仓位，负仓位为0
        ## 收益率分布
        ##plt, fig, ax = post.matplot()
        ##sns.histplot(trade_result[2])
        ##plt.show()
        position = (winrate*win - (1-winrate)*loss)/(win*loss)
        position[position<0] = 0
        # 作图
        plt, fig, ax = FB.display.matplot()
        ax.plot(100*winrate, c='C2', label='胜率')
        ax.plot(100*odds, c='C0', label='赔率')
        ax1 = ax.twinx()
        ax1.plot(position, c='C3', label='最佳仓位')
        ax.set_ylabel('（%）')
        ax.set_xlabel('bar')
        fig.legend(loc='lower center', ncol=3)
        plt.show()


    # 净值累计加减一个方差
    def draw_std_net(self):
        plt1, fig1, ax1 = FB.display.matplot()
        net = self.net.loc[:, 1:]
        net_mean = net.mean()
        net_up = net_mean + self.net.std()
        net_low = net_mean - self.net.std()
        ax1.plot(net_mean, color='darkblue', linewidth=2.0, label='均值')
        ax1.plot(net_up, color='darkred', linewidth=2.0, label='均值+方差')
        ax1.plot(net_low, color='darkgreen', linewidth=2.0, label='均值-方差')
        ax1.legend(loc='upper left')
        plt1.show()
    
    # 净值累计最大值&净值最小值
    def draw_e_ratio(self):
        plt1, fig1, ax1 = FB.display.matplot()
        net = self.net.loc[:, 1:]
        net_max = net.cummax(axis=1).mean()
        net_min = net.cummin(axis=1).mean()
        e_ratio = net_max/net_min
        ax1.plot(e_ratio, color='darkred', linewidth=2.0)
        ax1.set_xlabel('set_xlabel')
        plt1.show()
    
    # 仅一个信号
    # i是siganl中第i个信号
    def draw_one_signal_net(self, date, code):
        plt0, fig0, ax0 = FB.display.matplot()
        sr = self.signal_sr_df.loc[date, code]
        ax0.bar(sr.index, sr.values, width=0.5,  label='单日超额', color='darkgoldenrod')
        ax1 = ax0.twinx()
        net = self.net.loc[date, code]
        net = net/net.loc[1]
        ax1.plot(net, color='crimson', label='累计净值', linewidth=2.0)
        fig0.legend(loc='lower center')
        plt0.show()




# # 在事件发生后，持有n日后(事件次bar open至+Tbar收盘价（T=0为次bar开盘至收盘的收益）)，收益率
# #dict_date:  key thscode, value list of date(事件后可以买入的日期)
# # hold_days = list(range(61))
# #df_price:   date thscode  open close
# def event_return(dict_date, df_price, hold_days):
#     # 建立dataframe 前两列为合约与日期
#     columns = ['thscode', 'date']
#     # return_1 代表持有1天，即当天(0)的收盘价与开盘价之比
#     for i in hold_days:
#         columns.append('return_%d'%(i+1))
#     df_return = pd.DataFrame(columns = columns)
#     # 每个合约
#     for i in list(dict_date.keys()):
#         # 事件日期列表
#         list_date = dict_date[i]
#         # 如果在行情数据中没有,输出，跳过
#         if(i not in df_price.thscode.unique()):
#             print('not found',i)
#             continue
#         # 筛选出此合约行情
#         df_ = df_price[df_price.thscode == i]
#         # 按日期排序
#         df_ = df_.sort_values(by = 'date')
#         df_ = df_.reset_index(drop=True)
#         # 每一次事件
#         for start_date in list_date:
#             # 公告日期为实际发布公告日期后次日，在此时可以直接买入
#             # 为交易日则直接买入 
#             if start_date in df_.date.values:
#                 start = df_[df_.date == start_date]
#             # 如果不是交易日则后延
#             else:
#                 # 最多尝试30天
#                 try_num = 0
#                 while try_num < 30:
#                     try_num += 1
#                     start_date += datetime.timedelta(1)
#                     if start_date in df_.date.values:
#                         start = df_[df_.date == start_date]
#                         break
#                 # 没有找到则下一个日期或转债
#                 if(try_num==30):
#                     print('fail: ', i, start_date)
#                     continue
#             # 持有到end，需存在行情数据
#             dur = [start.index[0]+dur_i for dur_i in hold_days if (start.index[0]+dur_i) < len(df_.index)]
#             end = df_.loc[dur]
#     #       公告日开盘价到持有日收盘价 收益率
#             return_list = list((end.close/start.open.values[0]).apply(lambda x: math.log(x)))
#         # 字典 value
#             dict_values = [i,start_date]
#             dict_values.extend(return_list)
#             append_dict = dict(zip(columns, dict_values))
#             df_return = df_return.append(append_dict, ignore_index=True)
        
#     return df_return