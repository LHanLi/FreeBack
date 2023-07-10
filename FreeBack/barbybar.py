import pandas as pd
import numpy as np 
from queue import PriorityQueue


# 订单类
class Order():
# 初始化  ('Buy', code, vol, price)   Buy Sell 
    def __init__(self, type_, code, vol, price):
        # 交易方向（开仓 平仓）
        self.type = type_
        # 交易标的
        self.code = code
        # 买卖数量
        self.vol = vol
        # 交易价格  int：限价单   ’split'：平均价成交（最高价+最低价）/2 
        self.price = price
        # 执行优先级
        if type_ == 'Sell':
            self.priority = 0
        else:
            self.priority = 1

    def __lt__(self, other): 
        return self.priority < other.priority



class World():
# 初始化  market(DataFrame)， 初始资金， 交易成本（万）
# 单根k线最大成交量限制，       最大接收订单数（订单queue长度）
    def __init__(self, market, init_cash = 1000000, comm = 7, 
                 max_vol_perbar=1, max_order=100, tradetype='convertible'):
        self.error_log = []
        self.warning_log = []
        self.market = market
        self.init_cash = init_cash
        self.comm = comm/10000
        self.max_vol_perbar = max_vol_perbar
        self.tradetype = tradetype

    # barline 时间线（run函数遍历barline）
        self.barline = market.index.get_level_values(0).unique()
    # 当前bar 日期
        self.bar_n = 0
        self.cur_bar = self.barline[self.bar_n]

    # 当前市场
        self.cur_market = self.market.loc[self.cur_bar]

    # queue_order 订单队列（每个bar开始前检查策略在上一个bar发出的订单，并且执行）
        self.queue_order = PriorityQueue(max_order)
    # df_excute 订单执行记录（所有被执行订单会被记录到df_excute中）
        # columns:
        # 日期(订单执行），代码，买卖类型，执行价，发生数量，
        # 发生金额， 交易成本， 剩余现金， 执行状态， 
        # 订单报价类型（限价单、特定规则（平均价成交等））， 订单报价张数
        temp = ['date', 'code', 'BuyOrSell', 'price', 'occurance_vol', 
                'occurance_amount', 'comm', 'remain_cash', 'stat',
            'orderprice', 'ordervol']
        self.df_excute = pd.DataFrame(columns = temp)
        # 为df添加行、列名
        self.df_excute.columns.name = 'head'
        self.df_excute.index.name = 'date'
        # index: 订单唯一id(每个订单被加入到df_excute之后+1)
        self.unique = 0
        
    # df_hold 持仓表 columns为market中所有标的代码，index为barline，value为持有张数
        all_codes = list(market.index.get_level_values(1).unique())
        df_hold = pd.DataFrame(columns = all_codes)
        df_hold.columns.name = 'code'
        df_hold.index.name = 'date'
        # 初始持仓为0
        df_hold.loc[self.barline[0]] = {}
        self.df_hold = df_hold.fillna(0)
    # cur_hold 当前持仓 当前bar持仓不为0的代码， index为code   vol为持有张数  amount为持有金额
        cur_hold_vol = self.df_hold.loc[self.barline[0]]
        self.cur_hold_vol = cur_hold_vol[cur_hold_vol != 0]
        self.cur_hold_amount = (self.cur_hold_vol*self.cur_market['close']).dropna()  
    # dict_hold 持仓字典 只显示不为0的持仓 key为barline value为cur_hold_(vol/amount)
        self.dict_hold_vol = {}
        self.dict_hold_vol[self.barline[0]] = self.cur_hold_vol
        self.dict_hold_amount = {}
        self.dict_hold_amount[self.barline[0]] = self.cur_hold_amount 
        
    # series_cash 每bar持有现金   index为barline value为float
        self.series_cash = pd.Series(dtype = object)
    # cur_cash 当前现金
        self.cur_cash = init_cash
    # series_net, cur_net 当前净值（持仓（收盘价估算）+现金）
        self.series_net = pd.Series(dtype = object)
        self.cur_net = init_cash
    
    # log函数
    def log(self, notice):
        print(self.cur_bar, notice)
    def log_warning(self, notice):
        self.warning_log.append(str(self.cur_bar) + '    ' + notice)
    def log_error(self, notice):
        self.error_log.append(str(self.cur_bar) + '    ' + notice)
    # 提交订单函数
    def sub_order(self, order):
        self.queue_order.put(order)
    # excute调用
    # 更新订单执行记录函数
    def update_order(self, excute_log):
        self.df_excute.loc[self.unique] = excute_log
        self.unique += 1
    # 更新持仓函数  不更新hold_amount
    def update_hold(self, code, final_vol):
        self.df_hold.loc[self.cur_bar, code] = final_vol
        cur_hold = self.df_hold.loc[self.cur_bar]
        self.cur_hold_vol = cur_hold[cur_hold != 0].sort_values(ascending=False)   
        self.dict_hold_vol[self.cur_bar] = self.cur_hold_vol
    # 更新现金函数
    def update_cash(self, cash):
        self.cur_cash = cash
        self.series_cash.loc[self.cur_bar] = cash
    # 每个bar开始结束时调用
    # 更新净值函数 包含更新hold amount
    def update_net(self):
        self.cur_hold_amount = (self.cur_hold_vol * self.cur_market['close']).loc[self.cur_hold_vol.index].sort_values(ascending=False) 
        self.dict_hold_amount[self.cur_bar] = self.cur_hold_amount
        hold_amount = self.cur_hold_amount
        name_hold = hold_amount.index
        name_notnan = hold_amount[~np.isnan(hold_amount)].index
        name_delist = list(set(name_hold) - set(name_notnan))
        # 如果持有中有退市标的
        if name_delist != []:
            self.log_error('hold lost----delist list: %s'%name_delist)
        # 按照最后一个bar的收盘价计算价值
        lost_amount = 0
        for code in name_delist:
            price = self.market['close'].loc[:,code,:].iloc[-1]
            lost_amount += price * self.cur_hold_vol[code]
        self.cur_net = hold_amount.sum() + lost_amount + self.cur_cash
        self.series_net.loc[self.cur_bar] = self.cur_net

    # 提交订单函数
    def buy(self, code, vol = None, price = 'split'):
        # 默认卖出全部
        if vol == None:
            vol = self.cur_net/self.cur_market['close'].loc[code]
            order = Order('Buy', code, vol, price)
            self.sub_order(order)
        else:
            # amount为正表示做多 为负表示做空
            order = Order('Buy', code, vol, price)
            self.sub_order(order)
    def sell(self, code = None, vol = None, price = 'split'):
        # 无参数时默认清仓
        if code == None:
            for code,vol in self.cur_hold_vol.items():
                order = Order('Sell', code, vol, 'split')
                self.sub_order(order)
        elif vol == None:
            self.cur_hold_vol
            order = Order('Sell', code, self.cur_hold_vol[code], price)
            self.sub_order(order)
        else:
            order = Order('Sell', code, vol, price)
            self.sub_order(order)

    # 执行订单部分
#    @staticmethod
    def rounding(self, vol):
        # 可转债最小交易单位为10张
        if self.tradetype == 'convertible':
            return vol - vol%10
        elif self.tradetype == 'stock':
            return vol - vol%100
        else:
            return vol
        
    # 接收订单对象执行
    def excute(self, order):
        # 保证order.code在当前可交易code中
        inmarket = True
        try:
            #self.cur_market.loc[order.code]
            if self.cur_market['vol'].loc[order.code]==0:
                self.log_warning('excute sus----code: %s, unique:%s'%(order.code, self.unique+1))
                # 交割单
                excute_log = {'date':self.cur_bar, 'code':order.code, 'BuyOrSell':order.type,
                            'price':0, 'occurance_vol':0, 'occurance_amount':0, 
                            'comm':0, 'remain_cash': self.cur_cash, 'stat':'sus code',
                                'orderprice':order.price, 'ordervol':order.vol}
                # update
                self.update_order(excute_log)
                inmarket = False
        except:
            # 没有找到code不发生交易
            self.log_error('excute 404----code: %s, unique:%s'%(order.code, self.unique+1))
            # 交割单
            excute_log = {'date':self.cur_bar, 'code':order.code, 'BuyOrSell':order.type,
                            'price':0, 'occurance_vol':0, 'occurance_amount':0, 
                            'comm':0, 'remain_cash': self.cur_cash, 'stat':'404 code',
                                'orderprice':order.price, 'ordervol':order.vol}
            self.update_order(excute_log)
            inmarket = False
        if inmarket:
            # price 订单执行价
            if order.price == 'split':
                # 平均价执行
                price = (self.cur_market.loc[order.code]['high'] + self.cur_market.loc[order.code]['low'])/2
            elif order.price == 'open':
                price = self.cur_market.loc[order.code]['open']
            elif order.price == 'close':
                price = self.cur_market.loc[order.code]['close']
            else:
                # 限价单
                price = order.price

            # vol  成交量， stat  状态
            # 当前bar最大成交
            max_vol1 = self.max_vol_perbar * self.cur_market.loc[order.code]['vol']
            # 买入并且最低价不高于执行价
            if order.type == 'Buy': 
                # 当前现金cur最大买入量
                max_vol0 = self.cur_cash/price
                if self.cur_market.loc[order.code]['low'] <= price:
                    # 现金限制最大成交量
                    if max_vol0 <= max_vol1:
                        if order.vol > max_vol0:
                            vol = self.rounding(max_vol0)
                            stat = 'not enough cash'
                        else:
                            vol = self.rounding(order.vol)
                            stat = 'normal'
                    # 当前bar最大成交量限制
                    else:
                        if order.vol > max_vol1:
                            vol = self.rounding(max_vol1)
                            stat = 'not enough vol'
                        else:
                            vol = self.rounding(order.vol)
                            stat = 'normal'
                else:
                    vol = 0
                    stat = 'price lower than low'
            # 卖出并且最高价不低于执行价
            if order.type == 'Sell':
                # 持仓最大卖出量
                try:
                    max_vol2 = self.cur_hold_vol[order.code]
                except:
                    max_vol2 = 0
                if self.cur_market.loc[order.code]['high'] >= price:
                    # 最大持仓数量限制
                    if max_vol2 <= max_vol1:
                        if order.vol > max_vol2:
                            # 可以全部卖出
                            vol = max_vol2
                            stat = 'not enough hold'
                        else:
                            vol = self.rounding(order.vol)
                            stat = 'normal'

                    # 当前bar最大成交量限制
                    else:
                        if order.vol > max_vol1:
                            vol = self.rounding(max_vol1)
                            stat = 'not enough vol'
                        else:
                            vol = self.rounding(order.vol)
                            stat = 'normal'
                else:
                    vol = 0
                    stat = 'price higher than high'

            # 执行交易
            if order.type == 'Buy':
                # 交易处理完成后现金、持仓变化。
                cur_cash_ = self.cur_cash - vol*price
                final_vol = self.df_hold.iloc[-1][order.code] + vol
                comm_cost = vol*price*self.comm
                cur_cash_ = cur_cash_ - comm_cost
                # 订单执行记录
                excute_log = {'date':self.cur_bar, 'code':order.code, 'BuyOrSell':order.type,
                        'price':price, 'occurance_vol':vol, 'occurance_amount':vol*price, 
                            'comm':comm_cost, 'remain_cash': cur_cash_, 'stat':stat,
                            'orderprice':order.price, 'ordervol':order.vol}
            # order.type == ‘Sell'
            else:
                cur_cash_ = self.cur_cash + vol*price
                final_vol = self.df_hold.iloc[-1][order.code] - vol
                comm_cost = vol*price*self.comm
                cur_cash_ = cur_cash_ - comm_cost
                excute_log = {'date':self.cur_bar, 'code':order.code, 'BuyOrSell':order.type,
                        'price':price, 'occurance_vol':vol, 'occurance_amount':vol*price, 
                            'comm':comm_cost, 'remain_cash': cur_cash_, 'stat':stat,
                                    'orderprice':order.price, 'ordervol':order.vol}
            # 更新World中信息
            self.update_order(excute_log)     # 交割单
            self.update_cash(cur_cash_)
            self.update_hold(order.code, final_vol)

    # 初始化（定义变量等）
    def init(self):
        pass 
    # 策略
    def strategy(self):
        pass

    # 检查退市未来函数(未来10天中没有此code)
    def future_delist(self, code, n=10):
        i = 1
        future_date = self.barline[self.bar_n]
        while future_date < self.barline[-1] and i<=n:
            future_date = self.barline[self.bar_n + i]
            i += 1
            try:
                # 检查此合约
                self.market.loc[future_date, code]
            except:
                self.log_warning('future delist----delist date: %s, code: %s'%(future_date, code))
                return True
        return False
    # 转债专用
    def convertible_delist(self, code, n=10):
        i = 1
        future_date = self.barline[self.bar_n]
        while future_date < self.barline[-1] and i<=n:
            future_date = self.barline[self.bar_n + i]
            i += 1
            try:
                # 非查无此合约
                vol = self.market['vol'].loc[future_date, code]
                if vol == 0:
                    # 有强赎、退市、兑付公告，则成交量为0代表退市
                    if self.cur_market['announce'].loc[code] == 'Q1' or self.cur_market['announce'].loc[code] == 'F' or self.cur_market['announce'].loc[code] == 'T': 
                        return True
            except:
                # 当期未退市
                try:
                # 没有强赎、退市、兑付公告则使用未来函数
                    if self.cur_market['announce'].loc[code] != 'Q1' and self.cur_market['announce'].loc[code] == 'F' and self.cur_market['announce'].loc[code] != 'T': 
                        self.log_warning('future delist----delist date: %s, code: %s'%(future_date, code))
                    return True
                except:
                    return True
        return False

    # 回测运行
    def run(self):
        self.init()
    # 遍历每个bar
        for bar_ in range(len(self.barline)):
#            self.log('%s'%bar_)
        # regular
            self.log('new bar')
            # 当前bar 日期
            self.bar_n = bar_
            self.cur_bar = self.barline[bar_]
            self.cur_market = self.market.loc[self.cur_bar]
            # 更新账户状态 (默认与之前相同，防止没有订单情况)
            self.df_hold.loc[self.cur_bar] = self.df_hold.iloc[-1]
            self.dict_hold_vol[self.cur_bar] = self.cur_hold_vol
            self.dict_hold_amount[self.cur_bar] = self.cur_hold_amount
            self.update_cash(self.cur_cash)
            self.update_net()

        # broker处理订单（第一个bar不会处理，此时cur_hold和cur_cash为初始值）
            self.log('excute yesterbar order')
            while not self.queue_order.empty():
                # 接收订单
                order = self.queue_order.get()
                self.excute(order)
            
            self.log('run strategy')
        # 策略部分
            self.strategy()
            self.log('end bar')
            
        # 更新过hold之后再次更新净值
            self.update_net()
        
    # 开启作弊 则用当前bar执行交易
    def cheat_run(self):
        self.init()
        # 标的交易可以无限细分
        self.tradetype = None

    # 遍历每个bar
        for bar_ in range(len(self.barline)):
            self.log('new bar')
        # regular
            self.bar_n = bar_
            self.cur_bar = self.barline[bar_]
            self.cur_market = self.market.loc[self.cur_bar]
            # 更新账户状态 (默认与之前相同，防止没有订单情况)
            self.df_hold.loc[self.cur_bar] = self.df_hold.iloc[-1]
            self.dict_hold_vol[self.cur_bar] = self.cur_hold_vol
            self.dict_hold_amount[self.cur_bar] = self.cur_hold_amount
            self.update_cash(self.cur_cash)
            self.update_net()
        # 策略 
            self.log('run strategy')
            self.strategy()
        # 直接处理订单
            self.log('excute thisbar order')
            while not self.queue_order.empty():
                order = self.queue_order.get()
                self.excute(order)
            self.update_net()
            self.log('end bar')

