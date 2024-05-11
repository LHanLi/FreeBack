import FreeBack as FB
import numpy as np
import pandas as pd



####################################################################
########################## 常用策略框架 ##############################
###################################################################



# 择股策略、元策略
class MetaStrat():
    # 'inexclude':,
        # 当格式为('bool', False)时， 'bool'列为筛选条件, 'bool'为True为符合条件的证券
        # 格式为（False, 'bool'), 'bool'列为排除条件, 'bool'为False为符合条件的证券
        # 格式为 ['code0', 'code1', ] 时为等权持有固定证券组合，'cash'表示持有现金
    # 'score':float,   按score列由大到小选取证券,等权持有
    # 'hold_num':float,    取前hold_num（大于1表示数量，小于1小于百分比）只
    # market，pd.DataFrame, 需要包括策略需要调取的列，可以先不加
    # price，当前日期可以获得的价格数据,可以使用 'close'收盘价（有一点未来信息），或者下根bar开盘价/TWAP/VWAP
    def __init__(self, inexclude, score=None, hold_num=None, market=None, price='close', interval=1):
        self.inexclude = inexclude
        self.score = score
        self.hold_num = hold_num
        self.market = market
        self.price = price
        self.interval = interval
    # 为market添加cash品种
    def add_cash(self):
        cash = pd.DataFrame(index=self.market.index.get_level_values(0).unique())
        cash['code']  = 'cash'
        cash['name'] = '现金'
        cash[self.price] = 1
        cash = cash.reset_index().set_index(['date', 'code'])
        self.market = pd.concat([self.market, cash]).sort_values('date')
    # 获得虚拟持仓表(价格每一时刻货值都为1的持仓张数)
    def get_hold(self):
        # 按列表持股
        if type(self.inexclude)==list:
            if 'cash' in (self.inexclude):
                self.add_cash()
            df_hold = self.market.loc[:, self.inexclude, :]
        # 按排除、排序规则持股
        else:
            keeppool_rank = (lambda x: self.market[self.market[x[0]]] if x[0] \
                                else self.market[~self.market[x[1]]])(self.inexclude)[self.score].\
                                    groupby('date').rank(ascending=False, pct=(self.hold_num<1))
            df_hold = self.market.loc[keeppool_rank[keeppool_rank<=self.hold_num].index].copy()
            # 检查有无空仓情形，如果有的话就添加现金
            lost_bars = list(set(self.market.index.get_level_values(0))-set(df_hold.index.get_level_values(0)))
            if lost_bars!=[]:
                self.add_cash()
                df_hold = pd.concat([self.market.loc[lost_bars, 'cash', :], df_hold])
        # 等权（总账户市值1块钱）
        df_hold = (1/df_hold[self.price].unstack()).fillna(0)
        df_hold = df_hold.apply(lambda x: x/(x!=0).sum(), axis=1)
        # 去掉一直持仓为0的品种
        always_not_hold = (df_hold==0).all()
        self.df_hold = df_hold[always_not_hold[~always_not_hold].index].copy()
    ## 调仓间隔不为1时，需考虑调仓问题
    #def get_interval(self):
    #    if self.interval!=1:
    #        # 以interval为周期 调整持仓的持仓表
    #        # 选取的index  interval = 3  0,0,0,3,3,3,6...
    #        take_hold = [self.df_hold.index[int(i/self.interval)*self.interval]\
    #                            for i in range(len(self.df_hold.index))]
    #        real_hold = self.df_hold.loc[take_hold].copy()
    #        ## 提取的index非连续，复原到原来的连续交易日index
    #        real_hold.index = self.df_hold.index
    #        # 去掉一直持仓为0的品种
    #        always_not_hold = (real_hold==0).all()
    #        self.df_hold = real_hold[always_not_hold[~always_not_hold].index].copy()
    # 调仓间隔不为1时，需考虑调仓问题
    def get_interval(self, df):
        if self.interval!=1:
            # 以interval为周期 获取df 
            # 选取的index  interval = 3  0,0,0,3,3,3,6...
            take_df = [df.index[int(i/self.interval)*self.interval]\
                                for i in range(len(df.index))]
            real_df = df.loc[take_df].copy()
            ## 提取的index非连续，复原到原来的连续交易日index
            real_df.index = df.index
            return real_df
        return False
    # 运行策略
    def run(self):
        self.get_hold()
        #self.get_interval()
        df_hold = self.get_interval(self.df_hold)
        if df_hold:
            # 如果不是每日再平衡的话，可能会有品种一只持仓为0，去掉一直持仓为0的品种
            always_not_hold = (df_hold==0).all()
            self.df_hold = df_hold[always_not_hold[~always_not_hold].index].copy()
        # 判断cash是否在持仓，如果在的话避免price没有cash列
        if 'cash' in self.df_hold.columns:
            self.add_cash()
        # 价格矩阵，去掉全是0的列
        self.df_price = pd.DataFrame(self.market[self.price]).\
            pivot_table(self.price, 'date' ,'code')[self.df_hold.columns].copy()
        # 虚拟货值矩阵
        self.df_amount = (self.df_hold*self.df_price).fillna(0)
        # 权重矩阵
        self.df_weight = (self.df_amount.apply(lambda x: (x/x.sum()).fillna(0), axis=1))
        # 净值贡献矩阵
        returns = (self.df_price/self.df_price.shift() - 1).fillna(0)
        self.df_contri = (self.df_weight.shift()*returns).fillna(0)
        self.returns = self.df_contri.sum(axis=1)
        self.net = (self.returns+1).cumprod()
        # 获得真实持仓市值与持仓张数(净值需要interval)
        net = self.get_interval(self.net)
        if not net:
            net = self.net
        self.df_amount = self.df_amount.mul(list(net.values), axis=0)
        self.df_hold = self.df_hold.mul(list(net.values), axis=0)
        # 交易金额
        delta_hold = self.df_hold-self.df_hold.shift().fillna(0)
        self.delta_amount = (delta_hold*self.df_price).fillna(0)
        # cash的变化不会带来换手，可能没有‘cash'列
        if 'cash' in self.df_hold.columns:
            self.turnover = abs(self.delta_amount).drop(columns='cash').sum(axis=1)/self.net
        else:
            self.turnover = abs(self.delta_amount).sum(axis=1)/self.net



# 组合策略、择时策略
# 根据择时条件选择陪着不同的择股策略
# conds = [满足条件0的交易日（index或lsit），满足条件1的交易日, ..., 满足条件n的交易日]
# strats =[条件0对应策略0（MetaStrat）,   非条件0且条件1对应策略1, ... , 非条件0到条件n-1且条件n对应策略n， 剩余时间执行条件n+1]
class ComboStrat(MetaStrat):
    def __init__(self, conds, strats, market, price='close', interval=1):
        self.conds = conds
        self.strats = strats
        self.market = market
        self.price = price
        self.interval = interval
    def get_hold(self):
        # 策略择时模块,将全部交易日按择时条件划分
        # 满足条件0为
        all_days = self.market.index.get_level_values(0).unique()
        print('全部交易日:', len(all_days))
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
        # 检查状态是否完全覆盖总交易日
        if len(all_days)==sum([len(i) for i in stat_days]):
            pass
        else:
            print('have tradedays not cover')
            return
        # 子策略模块
        # stati对应strati对应df_holdi
        df_holds = []
        for i in range(len(self.strats)):
            strati = self.strats[i]
            strati.market = self.market.loc[stat_days[i]]
            strati.price = self.price
            print('状态%s交易日：'%i, len(stat_days[i]))
            strati.get_hold()
            df_holds.append(strati.df_hold)
        self.df_hold = pd.concat(df_holds).sort_values(by='date').fillna(0)




'''
#===========================  选股策略  ===================================
# market, multiindex(date, code) 必须列为 'bool', 'score', ‘next_return'
def ChooseSecurities(market, strat0):
    # 添加保证金账户（空仓）
    def add_deposit(market):
        deposit = pd.DataFrame(index=market.index.get_level_values(0).unique())
        deposit['code']  = 'deposit'
        deposit['next_returns'] = 0
        deposit = deposit.reset_index(['date', 'code'])
        return pd.concat([market, deposit]).sort_values('date')
    if type(strat0['in/exclude'])==list:
        if (strat0['in/exclude']==[]) | (strat0['in/exclude'][0]==''):
            # 空仓
            deposit = pd.DataFrame(index=market.index.get_level_values(0).unique())
            deposit['code'] = 'deposit'
            deposit['next_returns'] = 0
            return deposit.reset_index().set_index(['date', 'code'])
        else:
            return market.loc[:, strat0['in/exclude'], :][['next_returns']]
    else:
        keeppool_rank = (lambda x: market[market[x[0]]] if x[0] \
                            else market[~market[x[1]]])(strat0['in/exclude'])[strat0['score']].\
                                groupby('date').rank(ascending=False, pct=(strat0['select']<1))
        return market.loc[keeppool_rank[keeppool_rank<=strat0['select']].index][['next_returns']]


#============================  择时选股策略  ===================================
# 根据择时条件选择陪着不同的择股策略
# conds = [满足条件0的交易日，满足条件1的交易日, ..., 满足条件n的交易日]
# strats =[条件0对应策略0,   非条件0且条件1对应策略1, ... , 非条件0到条件n-1且条件n对应策略n， 剩余时间执行条件n+1]
def ChooseStrat(market, conds, strats):
    # 策略择时模块,将全部交易日按择时条件划分
    # 满足条件0为
    all_days = market.index.get_level_values(0).unique()
    print('全部交易日:', len(all_days)) 
    stat_days = []  
    left_days = all_days
    # 从第一个择时条件开始筛选
    for cond in conds:
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
    # 检查状态是否完全覆盖总交易日
    if len(all_days)==sum([len(i) for i in stat_days]):
        pass
    else:
        print('have tradedays not cover')
        return
    # 子策略模块
    # stati对应strati对应df_holdi
    df_holds = []
    for i in range(len(strats)):
        strati = strats[i]
        market_stati = market.loc[stat_days[i]]
        print('状态%s交易日：'%i, len(stat_days[i]))
        df_holdi = ChooseSecurities(market_stati, strati) 
        df_holds.append(df_holdi)
    return pd.concat(df_holds).sort_values(by='date')



###########
'''







