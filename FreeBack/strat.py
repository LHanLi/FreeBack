import FreeBack as FB
import numpy as np
import pandas as pd

####################################################################
########################## 常用策略框架 ##############################
###################################################################



#===========================  选股策略  ===================================
# market, multiindex(date, code) 必须列为 'bool', 'score', ‘next_return'
# 策略使用字典存储，包含以下字段
# 'in/exclude':, 
# 当格式为('bool', False)时， 'bool'列为筛选条件, True为符合条件的证券
# 格式为（False, 'bool'), 'bool'列为排除条件, False为符合条件的证券
# 格式为 ['code0', 'code1', ] 时为持有固定证券组合，空列表则空仓
# 'score':float,   按score列由大到小选取证券
# 'hold_num':float,    取前hold_num（大于1表示数量，小于1小于百分比）只
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








