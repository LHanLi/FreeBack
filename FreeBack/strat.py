import FreeBack as FB
import numpy as np
import pandas as pd
import time



####################################################################
########################## 常用策略框架 ##############################
###################################################################

# 修正冻结交易日（停牌、涨跌停等）的returns,market
def frozen_correct(code_returns, market, buy_frozen_days, sell_frozen_days=None):
    if code_returns.name:
        returns_name = code_returns.name
    else:
        returns_name = 0
    # code_returns 调整：连续冻结交易日（涨跌停/停牌）收益转移到第一个冻结交易日
    code_returns = code_returns.reindex(market.index).fillna(0) # 收益对齐至market
    if type(sell_frozen_days)==type(None):
        sell_frozen_days = buy_frozen_days
    from numba import njit
    @njit
    def compute_freeze_blocks(buy_arr, sell_arr):
        freeze = np.zeros(buy_arr.shape[0], dtype=np.int32)
        block = 0
        is_frozen = False
        for i in range(len(freeze)):
            # 如果当天买入信号为 True 或者前一天已冻结，则当天属于冻结状态
            if buy_arr[i] or is_frozen:
                if not is_frozen:
                    block += 1  # 启动新冻结区间
                is_frozen = True
                freeze[i] = block
            else:
                freeze[i] = 0 
            # 如果处于冻结状态，且当天卖出信号为 False，则冻结状态在当天结束（当天仍归入该区间），下一天不再冻结
            if is_frozen and (not sell_arr[i]):
                is_frozen = False
        return freeze
    def process_series(buy_series, sell_series):
        # 转为 numpy array
        res = compute_freeze_blocks(buy_series.values, sell_series.values)
        return pd.Series(res, index=buy_series.index)
    frozen_days_labels = []
    for code in buy_frozen_days.index.get_level_values(1).unique():
        res = process_series(buy_frozen_days.loc[:, code, :], sell_frozen_days.loc[:, code, :])
        res = res.reset_index()
        res['code'] = code
        res = res.set_index(['date', 'code'])[0]
        frozen_days_labels.append(res)
    frozen_days_labels = pd.concat(frozen_days_labels).sort_index()
    frozen_days_start = (frozen_days_labels!=0)&(frozen_days_labels!=frozen_days_labels.groupby('code').shift())
    frozen_days_start = frozen_days_start.map(lambda x: 1 if x else 0)
    frozen_days_labels = frozen_days_labels[frozen_days_labels>0]
    frozen_days_labels = pd.Series(frozen_days_labels.index.get_level_values(1), \
                index=frozen_days_labels.index)+frozen_days_labels.astype(str)               # 每一个冻结区块有唯一值
    code_returns_frozen = (1+code_returns).groupby(frozen_days_labels).prod()-1  # 计算冻结交易日的收益率
    code_returns_frozen = code_returns_frozen.reset_index().merge(\
        frozen_days_labels.loc[frozen_days_start[frozen_days_start==1].index]\
           .reset_index().rename(columns={0:'index'}), on='index')\
            .set_index(['date', 'code'])[returns_name].sort_index()         # 收益率对齐到冻结首日
    code_returns_frozen = code_returns_frozen.reindex(frozen_days_labels.index).fillna(0)  # 其后冻结日收益为0
    code_returns.loc[code_returns_frozen.index] = code_returns_frozen # 修正code_returns
    return code_returns, market[~buy_frozen_days]


# 择股策略、元策略
class MetaStrat():
    # 'inexclude':,
        # True,  不排除
        # 'include'， 'include'列为bool值，为True为符合条件的证券
        # 'exclude'  'exclude列为bool值, 为True为排除的证券
        # 格式为 ['code0', 'code1', ] 时为等权持有固定证券组合，'cash'表示持有现金
    # 'score':float,   按score列由大到小选取证券,等权持有
    # 'hold_num':float,    取前hold_num（大于1表示数量，小于1小于百分比）只
    # market，pd.DataFrame, 需要包括策略需要调取的列，可以先不加
    # price，当前日期可以获得的价格数据,可以使用 'close'收盘价（有一点未来信息），或者下根bar开盘价/TWAP/VWAP
    # hold_weight 权重, code_returns 标的收益(T-1日持有标的的收益),   MultiIndex, Seires
    def __init__(self, market, inexclude=None, score=None, hold_num=None,\
                            price='close', interval=1, direct=1, hold_weight=None, code_returns=None):
        self.inexclude = inexclude
        self.score = score
        self.hold_num = hold_num
        # 记录是否是上市最后一天
        #market['Z'] = market.index.get_level_values(1).duplicated(keep='last')
        #market['Z'] = ~market['Z']
        market.loc[:, 'Z'] = ~market.index.get_level_values(1).duplicated(keep='last')
        market.loc[market.index[-1][0], 'Z'] = False
        self.market = market
        self.price = price
        self.interval = interval
        self.direct = direct
        self.hold_weight = hold_weight
        self.code_returns = code_returns.loc[market.index[0][0]:market.index[-1][0]]
    # 为market添加cash品种
    def add_cash(self):
        cash = pd.DataFrame(index=self.market.index.get_level_values(0).unique())
        cash['code']  = 'cash'
        cash['name'] = '现金'
        cash[self.price] = 1
        cash = cash.reset_index().set_index(['date', 'code'])
        self.market = pd.concat([self.market, cash]).sort_index()
        cash['returns'] = 0
        if type(self.code_returns)!=type(None):
            self.code_returns = pd.concat([self.code_returns, cash['returns']]).sort_index()
    # 获得虚拟持仓表(价格每一时刻货值都为1的持仓张数)
    def get_hold(self):
        if type(self.inexclude)==list:  # 按列表持股
            if 'cash' in (self.inexclude):
                self.add_cash()
            df_hold = self.market.loc[:, self.inexclude, :]
            self.keeppool_rank = pd.Series(index=df_hold.index)  # 记录选中顺序
        #elif self.inexclude==None:  # 持仓全部股票
        #    df_hold = self.market 
        #    self.keeppool_rank = pd.Series(index=df_hold.index)
        else: # 避免持有退市前最后一天股票
            if not self.inexclude:
                keeppool_rank = self.market[self.score][~self.market['Z']]
            elif self.inexclude=='include':
                keeppool_rank = self.market[self.score][(~self.market['Z'])&self.market['include']]
            elif self.inexclude=='exclude':
                keeppool_rank = self.market[self.score][(~self.market['Z'])&(~self.market['exclude'])]
            #time0 = time.time()
            keeppool_rank = keeppool_rank.groupby('date').rank(ascending=False, \
                                        pct=(self.hold_num<1), method='first')
            #print('按日期分组排名耗时', time.time()-time0)  
            #time0 = time.time()
            self.keeppool_rank = keeppool_rank[keeppool_rank<=self.hold_num]
            df_hold = self.market[[self.price]].loc[self.keeppool_rank.index] #.copy()
            # 检查有无空仓情形，如果有的话就在空仓日添加现金
            if len(self.market.index.get_level_values(0).unique())!=\
                    len(df_hold.index.get_level_values(0).unique()):
                lost_bars = list(set(self.market.index.get_level_values(0))-\
                                            set(df_hold.index.get_level_values(0)))
                self.add_cash()
                df_hold = pd.concat([self.market.loc[lost_bars, 'cash', :][[self.price]],\
                                      df_hold])
        # 赋权
        if type(self.hold_weight)!=type(None):
            w_ = self.hold_weight   # 如果df_hold中有cash，权重也应该加入cash
        else:
            w_ = 1
        df_hold = ((w_/df_hold[self.price]).dropna().unstack()).fillna(0)
        # 总账户市值1块钱
        df_hold = df_hold.div((df_hold!=0).sum(axis=1), axis=0)
        self.df_hold = self.direct*df_hold
    # 调仓间隔不为1时，需考虑调仓问题
    def get_interval(self, df):
        if type(self.interval)!=int:
            # 选取调仓日
            take_df = [pd.Series(index=sorted(self.interval)).loc[:date].index[-1]\
                        for date in df.index]
        elif self.interval!=1:
            # 以interval为周期 获取df 
            # 选取的index  interval = 3  0,0,0,3,3,3,6...
            take_df = [df.index[int(i/self.interval)*self.interval]\
                                for i in range(len(df.index))]
        else:
            return df
        real_df = df.loc[take_df].copy()
        ## 提取的index非连续，复原到原来的连续交易日index
        real_df.index = df.index
        return real_df
    # 运行策略
    def run(self):
        #time0 = time.time()
        self.get_hold()
        #print('获取持仓矩阵耗时', time.time()-time0) 
        #time0 = time.time()
        df_hold = self.get_interval(self.df_hold)
        keeppool_rank = self.get_interval(self.keeppool_rank.fillna(1).\
                                    groupby('date').cumsum().unstack()).stack()
        self.keeppool_rank = keeppool_rank.reset_index().sort_values(by=['date',0]).\
            set_index(['date', 'code'])[0]
        #print('获取持仓表耗时', time.time()-time0)
        #time0 = time.time()
        always_not_hold = (df_hold==0).all() # 去掉一直持仓为0的品种
        self.df_hold = df_hold[always_not_hold[~always_not_hold].index].copy()
        # 判断cash是否在持仓，如果在的话避免price没有cash列
        if ('cash' in self.df_hold.columns)&('cash' not in self.market.index.get_level_values(1)):
            self.add_cash()
        # 价格矩阵，去掉没有持仓过的标的，缺失价格数据（nan）的日期
        df_price = pd.DataFrame(self.market[self.price]).\
            pivot_table(self.price, 'date' ,'code')[self.df_hold.columns]
        self.df_price = df_price[~df_price.isna().all(axis=1)].copy()
        # 虚拟货值矩阵
        self.df_amount = (self.df_hold*self.df_price).fillna(0)
        #print('获取虚拟货值矩阵耗时', time.time()-time0)  
        #time0 = time.time()
        # 权重矩阵
        self.df_weight = (self.df_amount.apply(lambda x:\
                                        (x/abs(x.sum())).fillna(0), axis=1))
        #print('获取权重矩阵耗时', time.time()-time0)  
        #time0 = time.time()
        # 净值贡献矩阵
        if type(self.code_returns)==type(None):
            self.code_returns = (self.df_price/self.df_price.shift() - 1).fillna(0)
        else:
            self.code_returns = self.code_returns.unstack().fillna(0)[self.df_price.columns]
        self.df_contri = (self.df_weight.shift()*self.code_returns).fillna(0)
        self.returns = self.df_contri.sum(axis=1)
        self.net = (self.returns+1).cumprod()
        #print('获取净值耗时', time.time()-time0)
        #time0 = time.time()
        # 为了准确计算换手率，需要获得真实持仓市值与持仓张数(净值需要interval)
        net = self.get_interval(self.net)
        self.df_amount = self.df_amount.mul(list(net.values), axis=0)
        self.df_hold = self.df_hold.mul(list(net.values), axis=0)
        # 交易金额（注意不能直接用市值相减）
        delta_hold = self.df_hold-self.df_hold.shift().fillna(0)
        self.delta_amount = (delta_hold*self.df_price).fillna(0)
        # cash的变化不会带来换手，可能没有‘cash'列
        self.df_turnover = abs(self.delta_amount.div(self.df_amount.sum(axis=1), axis=0))
        if 'cash' in self.df_hold.columns:
            self.df_turnover['cash'] = 0
        self.turnover = self.df_turnover.sum(axis=1)
        #print('获取换手率耗时', time.time()-time0)



# 根据择时条件选择陪着不同的择股策略
# conds = [满足条件0的交易日（index或lsit），满足条件1的交易日, ..., 满足条件n的交易日]
# strats =[条件0对应策略0（MetaStrat）,   非条件0且条件1对应策略1, ... , 非条件0到条件n-1且条件n对应策略n， 剩余时间执行策略n+1]
class ComboStrat(MetaStrat):
    def __init__(self, conds, strats, market, price='close', code_returns=None, interval=1):
        self.conds = conds
        self.strats = strats
        # 如果状态数和策略数相同，呢么默认空余状态使用最后一个策略
        if len(strats)==len(conds):
            self.strats.append(strats[-1])
        self.market = market
        self.price = price
        self.code_returns = code_returns
        self.interval = interval
    def get_hold(self):
        # 策略择时模块,将全部交易日按择时条件划分
        # 满足条件0为
        all_days = self.market.index.get_level_values(0).unique()
        print('共:', len(all_days), 'bars')
        stat_days = []
        left_days = all_days
        # 从第一个择时条件开始筛选
        for cond in self.conds:
            stati_days = []
            # 新的剩余交易日
            left_days_ = []
            for date in left_days:
                if date in cond:
                    stati_days.append(date)
                else:
                    left_days_.append(date)
            stat_days.append(stati_days)
            left_days = left_days_
        stat_days.append(left_days)
        # 子策略模块
        # stati对应strati对应df_holdi
        df_holds = []
        keeppool_rank = []
        for i in range(len(self.strats)):
            strati = self.strats[i]
            strati.market = self.market.loc[stat_days[i]]
            strati.price = self.price
            print('状态%s bars：'%i, len(stat_days[i]))
            if len(stat_days[i])==0:
                continue
            strati.get_hold()
            df_holds.append(strati.df_hold)
            keeppool_rank.append(strati.keeppool_rank)
        self.keeppool_rank = pd.concat(keeppool_rank).reset_index().\
            sort_values(by=['date', 0]).set_index(['date', 'code'])[0]
        self.keeppool_rank = self.get_interval(self.keeppool_rank)
        self.df_hold = pd.concat(df_holds).sort_values(by='date').fillna(0)



# 根据权重组合不同的MetaStrat
class MixStrat(MetaStrat):
    def __init__(self, weights, strats, market, inexclude=None, score=None, hold_num=None, \
                 price='close', interval=1, direct=1, hold_weight=None, code_returns=None):
        super().__init__(market, inexclude, score, hold_num, price, interval, direct, hold_weight, code_returns)
        self.weights = weights
        self.strats = strats
    def get_hold(self):
        from functools import reduce
        # 虚拟货值矩阵 按权重分配
        # 不同策略隔离运行
        #df_amount = reduce(lambda x, y: x.add(y, fill_value=0), \
        #                     [w*s.df_amount for w,s in zip(theory_weights, select_strats)])
        # 不同策略合并运行
        total_amount = pd.concat([i.df_amount.sum(axis=1) for i in self.strats], axis=1).sum(axis=1)
        df_amount = reduce(lambda x, y: x.add(y, fill_value=0), \
                [w*s.df_weight for w,s in zip(self.weights, self.strats)]).mul(total_amount, axis=0)
        df_price = pd.DataFrame(self.market[self.price]).pivot_table(self.price, 'date' ,'code')
        df_price['cash'] = 1
        self.df_hold = (df_amount/df_price).fillna(0)
        self.keeppool_rank = reduce(lambda x, y: x.add(y, fill_value=0), \
                            [w*s.keeppool_rank for w,s in zip(self.weights, self.strats)])


