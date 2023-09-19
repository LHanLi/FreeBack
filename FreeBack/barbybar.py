import pandas as pd
import numpy as np
import re, queue 



# 订单类
class Order():
# 初始化  ('Buy', code, vol, price)   Buy Sell
    def __init__(self, type_, code, vol, price, order_id):
        # 交易方向（开仓 平仓）
        self.type = type_
        # 交易标的
        self.code = code
        # 买卖数量
        self.vol = vol
        # 交易价格  int：限价单   ’split'：平均价成交（最高价+最低价）/2 
        self.price = price
        # 订单编号
        self.order_id = order_id 
        # 执行优先级
        if type_ == 'Sell':
            self.priority = 0
        else:
            self.priority = 1
    def __lt__(self, other): 
        return self.priority < other.priority
# 交易员类
class Trader():
    # type : amount_trader 按照目标金额成交
    # target_amount  dict，key:code, value: amount, 标的目标市值，第二日开盘由交易员挂单
    target_amount = {}
    # type: batch_trader 按照目标标的等权持有
    weight = None
    # type: buyhold_trader
    code = None
    vol = None
    order_id = None
    # 默认开盘价执行
    def __init__(self, type, price='open'):
        self.type = type
        self.price = price


class World():
# 初始化  market(DataFrame)， 初始资金， 交易成本（万），
# 单根k线最大成交量限制， 
# 交易证券类型 'convertible' 'convertible_split' 'stock'
# 初始持仓和现金， index是代码，包括cash， value是张数（现金则是金额）
    def __init__(self, market, init_cash = 1000000, comm = 7, 
                 max_vol_perbar=999, tradetype=None, 
                 init_stat=None,
                 ):
        self.temp_log = ''
        self.error_log = ''
        self.warning_log = ''
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
        self.queue_order = queue.PriorityQueue()
    # 交易员列表，策略可以像列表中添加交易员，一个交易员对应一个交易员函数，如果交易执行完毕则
    # 交易员离开队列
        self.queue_trader = queue.Queue()
    # df_excute 订单执行记录（所有被执行订单会被记录到df_excute中）
        # columns:
        # 日期(订单执行），代码，买卖类型，执行价，发生数量，
        # 发生金额， 交易成本， 剩余现金， 执行状态， 
        # 订单报价类型（限价单、特定规则（平均价成交等））， 订单报价张数
        temp = ['date', 'code', 'BuyOrSell', 'price', 'occurance_vol', 
                'occurance_amount', 'comm', 'remain_vol', 'remain_amount',
                'remain_cash', 'stat', 'orderprice', 'ordervol']
        self.df_excute = pd.DataFrame(columns = temp)
        # 为df添加行、列名
        self.df_excute.columns.name = 'head'
        self.df_excute.index.name = 'unique'
        # index: 订单唯一id(每个订单被加入到df_excute之后+1)
        self.unique = 0
        
    # df_hold 持仓表 columns为market中所有标的代码，index为barline，value为持有张数
        all_codes = list(market.index.get_level_values(1).unique())
        df_hold = pd.DataFrame(columns = all_codes)
        df_hold.columns.name = 'code'
        df_hold.index.name = 'date'
        # 初始持仓为0
        df_hold.loc[self.barline[0]] = {}
        # 初始持仓不为0
        if type(init_stat) != type(None):
            for i in init_stat.index:
                if i in df_hold.columns:
                    df_hold[i] = init_stat[i]
                elif i == 'cash':
                    init_cash = init_stat[i]
                else:
                    df_hold[i] = init_stat[i]
                    print('no code %s in market'%i)
        self.df_hold = df_hold.fillna(0)
    # cur_hold 当前持仓 当前bar持仓不为0的代码， index为code   vol为持有张数  amount为持有金额
        cur_hold_vol = self.df_hold.loc[self.barline[0]]
        self.cur_hold_vol = cur_hold_vol[cur_hold_vol != 0]
        self.cur_hold_amount = (self.cur_hold_vol*self.cur_market['close']).dropna()  
        
    # series_cash 每bar持有现金   index为barline value为float
        self.series_cash = pd.Series(dtype = object)
    # cur_cash 当前现金
        self.cur_cash = init_cash
    # series_net, cur_net 当前净值（持仓（收盘价估算）+现金）
        self.series_net = pd.Series(dtype = object)
        self.cur_net = init_cash
    
    # log函数
    def log(self, notice):
        log_str = str(self.cur_bar) + ', ' + notice 
        # 每行显示150个字符
        log_str = re.sub(r'(.{150})', '\\1\n', log_str) 
        self.temp_log += (log_str + '\n')
    def log_warning(self, notice):
        log_str = str(self.cur_bar) + ', ' + notice 
        log_str = re.sub(r'(.{150})', '\\1\n', log_str) 
        self.warning_log += (log_str + '\n')
    def log_error(self, notice):
        log_str = str(self.cur_bar) + ', ' + notice + '\n'
        log_str = re.sub(r'(.{150})', '\\1\n', log_str)
        self.error_log += (log_str + '\n')
        # 错误需要直接跳出提示
        print(log_str)
    # 提交订单函数
    def sub_order(self, order):
        self.queue_order.put(order)
    # 提交trader函数
    def sub_trader(self, trader):
        self.queue_trader.put(trader)
    # excute调用
    # 更新订单执行记录函数
    def update_order(self, order_id, excute_log):
        self.df_excute.loc[order_id] = excute_log
    # 更新持仓函数  不更新hold_amount
    def update_hold(self, code, final_vol):
        self.df_hold.loc[self.cur_bar, code] = final_vol
        cur_hold = self.df_hold.loc[self.cur_bar]
        self.cur_hold_vol = cur_hold[cur_hold != 0]
    # 更新现金函数
    def update_cash(self, cash):
        self.cur_cash = cash
        self.series_cash.loc[self.cur_bar] = cash
    # 每个bar开始结束时调用
    # 更新净值函数 包含更新hold amount
    def update_net(self):
        self.cur_hold_amount = (self.cur_hold_vol * self.cur_market['close']).loc[self.cur_hold_vol.index].sort_values(ascending=False) 
        hold_amount = self.cur_hold_amount
        name_hold = hold_amount.index
        name_notnan = hold_amount[~np.isnan(hold_amount)].index
        name_delist = list(set(name_hold) - set(name_notnan))
        # 如果持有中有退市标的
        if name_delist != []:
            self.log_error('hold lost----delist list: %s'%name_delist)
        # 按照最后一个bar的收盘价计算价值
        # 将剔除持仓，lost_amount换为现金 
        for code in name_delist:
            price = self.market['close'].loc[:,code,:].iloc[-1]
            lost_amount = price * self.cur_hold_vol[code]
            self.update_hold(code, 0)
            self.update_cash(self.cur_cash+lost_amount)
        self.cur_net = hold_amount.sum() + self.cur_cash
        self.series_net.loc[self.cur_bar] = self.cur_net

    # 提交订单函数
    def buy(self, code, vol = None, price_type = 'open'):
        # 默认满仓买入
        if vol == None:
            vol = self.cur_net/self.cur_market['close'].loc[code]
            order = Order('Buy', code, vol, price_type, self.unique)
            self.unique += 1
            self.sub_order(order)
        else:
            # amount为正表示做多 为负表示做空
            order = Order('Buy', code, vol, price_type, self.unique)
            self.unique += 1
            self.sub_order(order)
    def sell(self, code = None, vol = None, price_type = 'open'):
        # 无参数时默认全部清仓
        if code == None:
            for code,vol in self.cur_hold_vol.items():
                order = Order('Sell', code, vol, price_type, self.unique)
                self.unique += 1
                self.sub_order(order)
        # 无委托量时默认清仓该code
        elif vol == None:
            self.cur_hold_vol
            order = Order('Sell', code, self.cur_hold_vol[code], price_type, self.unique)
            self.unique += 1
            self.sub_order(order)
        else:
            order = Order('Sell', code, vol, price_type, self.unique)
            self.unique += 1
            self.sub_order(order)

    # 交易员订单 由trader()在策略的次一根bar内执行
    # 买入卖出标的至目标市值 target_amount dict key:code, value:amount
    def trade_amount(self, target_amount, price='open'):
        trader = Trader('amount_trader', price)
        trader.target_amount = target_amount
        self.queue_trader.put(trader)
    def runtrade_amount(self, trader):
        target_amount = trader.target_amount
        # 交易计划 index：code  amount：目标仓位
        # 目标手数，按照最新盘口成交价、最小成交量、当前持仓确定目标手数
        buy_vol = {}
        sell_vol = {}
        for code in target_amount.keys():
            #self.log(code)
            # 保证order.code在当前可交易code中
            inmarket = True
            try:
                self.cur_market['vol'].loc[code] == 0
            #    # 停牌
            #    if self.cur_market['vol'].loc[code]==0:
            #        #self.log_warning('target amount sus----code: %s'%(code))
            #        inmarket = False
            except:
                # 没有找到code不发生交易
                #self.log_error('target amount 404----code: %s'%(code))
                inmarket = False
            if inmarket:
            # 如果无持仓 则买入
                if code not in self.cur_hold_vol.index:
                    #self.log('open '+ code + ' ' + str(target_amount[code]))
                    buy_vol[code] = target_amount[code]/self.cur_market[trader.price].loc[code]
                else:
                    deltaamount = target_amount[code] - self.cur_hold_amount[code]
                    # 需要买入
                    if deltaamount > 0:
                        #self.log('buy '+ code + ' ' + str(deltaamount))
                        buy_vol[code] = deltaamount/self.cur_market[trader.price].loc[code]
                    else:
                        # 全部卖出
                        if target_amount[code] == 0:
                            #self.log('close '+ code)
                            sell_vol[code] = self.cur_hold_vol[code]
                        else:
                            #self.log('sell '+ code + ' ' + str(deltaamount))
                            sell_vol[code] = -deltaamount/self.cur_market[trader.price].loc[code]
        for code in buy_vol.keys():
            self.buy(code, buy_vol[code], trader.price)
        for code in sell_vol.keys():
            self.sell(code, sell_vol[code], trader.price)
    # 等权/weight加权持有目标标的
    def trade_batch(self, weight, price='open'):
        trader = Trader('batch_trader', price)
        weight = weight/weight.sum()
        trader.weight = weight
        self.queue_trader.put(trader)
    def runtrade_batch(self, trader):
        # 按照执行价格计算的总资产
        net = (self.cur_hold_vol*self.cur_market[trader.price]).sum() + self.cur_cash
        ## 平均每标的的持有金额
        #amount = net/len(trader.code_list)
        # 提交订单
        # 不在code_list中的直接清仓
        for code in self.cur_hold_vol.index:
            if code not in trader.weight.index:
                self.sell(code, price_type=trader.price)
        # 在code_list中的补齐至amount
        for code, w in trader.weight.items():
            amount = w*net
            # 如果不在市场中，则卖出999
            try:
                delta = (amount - self.df_hold.loc[self.cur_bar][code]*self.cur_market.loc[code][trader.price])\
                        /self.cur_market.loc[code][trader.price]
            except:
                delta = -999 
            if delta > 0:
                self.buy(code, delta, trader.price)
            else:
                self.sell(code, -delta, trader.price)
    # 买入持有固定时间
    def trade_buyhold(self, code, vol, holdtime):
        trader = Trader('buyhold_trader')
        trader.code = code
        trader.vol = vol
        trader.holdtime =holdtime
        self.queue_trader.put(trader)
    def runtrade_buyhold(self, trader):
        # 如果还没有订单编号（未执行），则执行买单
        if  trader.order_id == None:
            trader.order_id = self.unique
            self.buy(trader.code, trader.vol)
            return True
        else:
            excutelog = self.df_excute.loc[trader.order_id]
            dealdate = excutelog['date']
            occurance_vol = excutelog['occurance_vol']
            if (self.cur_bar-dealdate)>=trader.holdtime:
                self.sell(trader.code, occurance_vol)
                return False
            else:
                return True
    def trade_buystop(self, code, amount):
        pass

    # 初始化（定义变量等）
    def init(self):
        pass
    # 交易员
    def runtrader(self):
        # 执行全部queue_trader中trader
        savetrader = []
        while not self.queue_trader.empty():
            trader = self.queue_trader.get()
            if trader.type == 'amount_trader':
                self.runtrade_amount(trader)
            elif trader.type == 'batch_trader':
                self.runtrade_batch(trader)
            elif trader.type == 'buyhold_trader':
                if self.runtrade_buyhold(trader):
                    savetrader.append(trader)
            else:
                pass
        for i in savetrader:
            self.queue_trader.put(i)

    # 执行订单部分
#    @staticmethod
    def rounding(self, vol):
        # 可转债最小交易单位为10张
        if self.tradetype == 'convertible':
            return vol - vol%10
        # 可转债拆单，最小成交为30张
        elif self.tradetype == 'convertible_split':
            if vol <= 30:
                return 0
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
            # 停牌
            if self.cur_market['vol'].loc[order.code]==0:
                self.log_warning('try excute sus----code: %s, unique:%s'%(order.code, self.unique+1))
                try:
                    remain_vol = self.cur_hold_vol.loc[order.code]
                except:
                    remain_vol = 0
                try:
                    remain_amount = self.cur_hold_amount.loc[order.code]
                except:
                    remain_amount = 0
                # 交割单
                excute_log = {'date':self.cur_bar, 'code':order.code, 'BuyOrSell':order.type,
                            'price':0, 'occurance_vol':0, 'occurance_amount':0, 
                            'comm':0, 'remain_vol':remain_vol, 'remain_amount':remain_amount,
                              'remain_cash': self.cur_cash, 'stat':'sus code',
                                'orderprice':order.price, 'ordervol':order.vol}
                # update
                self.update_order(order.order_id, excute_log)
                inmarket = False
        except:
            # 没有找到code不发生交易
            self.log_error('excute 404----code: %s, unique:%s'%(order.code, self.unique+1))
            try:
                remain_vol = self.cur_hold_vol.loc[order.code]
            except:
                remain_vol = 0
            try:
                remain_amount = self.cur_hold_amount.loc[order.code]
            except:
                remain_amount = 0
            # 交割单
            excute_log = {'date':self.cur_bar, 'code':order.code, 'BuyOrSell':order.type,
                            'price':0, 'occurance_vol':0, 'occurance_amount':0, 
                            'comm':0, 'remain_vol':remain_vol, 'remain_amount':remain_amount,
                            'remain_cash': self.cur_cash, 'stat':'404 code',
                                'orderprice':order.price, 'ordervol':order.vol}
            self.update_order(order.order_id, excute_log)
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
                # 持仓最大卖出量(可以做空则无此限制)
                try:
                    max_vol2 = self.cur_hold_vol[order.code]
                except:
                    max_vol2 = 0
                if self.cur_market.loc[order.code]['high'] >= price:
                    # 最大持仓数量限制
                    if max_vol2 <= max_vol1:
                        #if order.vol > max_vol2:
                        #    # 可以全部卖出
                        #    vol = max_vol2
                        #    stat = 'not enough hold'
                        #else:
                        #    vol = self.rounding(order.vol)
                        #    stat = 'normal'
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
                final_amount = final_vol * self.cur_market.loc[order.code]['close']
                comm_cost = vol*price*self.comm
                cur_cash_ = cur_cash_ - comm_cost
                # 订单执行记录
                excute_log = {'date':self.cur_bar, 'code':order.code, 'BuyOrSell':order.type,
                        'price':price, 'occurance_vol':vol, 'occurance_amount':vol*price, 
                            'comm':comm_cost, 'remain_vol':final_vol, 'remain_amount':final_amount,
                            'remain_cash': cur_cash_, 'stat':stat,
                            'orderprice':order.price, 'ordervol':order.vol}
            # order.type == ‘Sell'
            else:
                cur_cash_ = self.cur_cash + vol*price
                final_vol = self.df_hold.iloc[-1][order.code] - vol
                final_amount = final_vol * self.cur_market.loc[order.code]['close']
                comm_cost = vol*price*self.comm
                cur_cash_ = cur_cash_ - comm_cost
                excute_log = {'date':self.cur_bar, 'code':order.code, 'BuyOrSell':order.type,
                        'price':price, 'occurance_vol':vol, 'occurance_amount':vol*price, 
                            'comm':comm_cost, 'remain_vol':final_vol, 'remain_amount':final_amount,
                            'remain_cash': cur_cash_, 'stat':stat,
                                    'orderprice':order.price, 'ordervol':order.vol}
            # 更新World中信息
            self.update_order(order.order_id, excute_log)
            self.update_cash(cur_cash_)
            self.update_hold(order.code, final_vol)

    # XD 股息、债息以及送转股
    def dividend(self):
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
    # 转债专用 转债退市前2-4周会有公告，所以当出现公告时检查退市没有用到未来数据
    def convertible_delist(self, code, n=10):
        i = 1
        future_date = self.barline[self.bar_n]
        while future_date < self.barline[-1] and i<=n:
            future_date = self.barline[self.bar_n + i]
            i += 1
            try:
                # 非查无此合约
                vol = self.market['vol'].loc[future_date, code]
                # 有强赎、退市、兑付公告，则成交量为0代表退市
                if vol == 0:
                    if self.cur_market['announce'].loc[code] == 'Q1' or self.cur_market['announce'].loc[code] == 'F' or self.cur_market['announce'].loc[code] == 'T': 
                        return True
            except:
                # 当期未退市
                try:
                # 没有强赎、退市、兑付公告则使用未来函数
                    if self.cur_market['announce'].loc[code] != 'Q1' and self.cur_market['announce'].loc[code] != 'F' and self.cur_market['announce'].loc[code] != 'T': 
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
            # 当前bar 日期
            self.bar_n = bar_
            self.cur_bar = self.barline[bar_]
            self.log('new bar')
            self.cur_market = self.market.loc[self.cur_bar]
            # 更新账户状态 (默认与之前相同，防止没有订单情况)
            self.df_hold.loc[self.cur_bar] = self.df_hold.iloc[-1]
            self.update_cash(self.cur_cash)
            self.update_net()

            # 交易员下单
            self.runtrader()
        # broker处理订单（第一个bar不会处理，此时cur_hold和cur_cash为初始值）
            self.log('excute yesterbar order')
            while not self.queue_order.empty():
                # 接收订单
                order = self.queue_order.get()
                self.excute(order)
        # 处理分红、股息、送股
            #self.dividend()
        # 更新过hold之后再次更新净值
            self.update_net()
            
            self.log('run strategy')
        # 策略部分
            self.strategy()
            self.log('end bar')
            
        # log写入文件
        f = open('barbybar_log.txt', 'w')
        f.write('Regular log: \n\n\n')
        f.write(self.temp_log)
        f.write('\n\n\nError log: \n\n\n')
        f.write(self.error_log)
        f.write('\n\n\nWarning log: \n\n\n')
        f.write(self.warning_log)
        f.close()
        
    # 开启作弊 则用当前bar执行交易
    def cheat_run(self):
        self.init()
    # 遍历每个bar
        for bar_ in range(len(self.barline)):
        # regular
            self.bar_n = bar_
            self.cur_bar = self.barline[bar_]
            self.log('new bar')
            self.cur_market = self.market.loc[self.cur_bar]
            # 更新账户状态 (默认与之前相同，防止没有订单情况)
            self.df_hold.loc[self.cur_bar] = self.df_hold.iloc[-1]
            self.update_cash(self.cur_cash)
            self.update_net()
        # 策略
            self.log('run strategy')
            self.strategy()
        # 直接处理订单
            self.log('excute thisbar order')
            # 交易员下单
            self.runtrader()
        # broker处理订单（第一个bar不会处理，此时cur_hold和cur_cash为初始值）
            while not self.queue_order.empty():
                # 接收订单
                order = self.queue_order.get()
                self.excute(order)
            self.update_net()
            self.log('end bar')

        # log写入文件
        f = open('barbybar_log.txt', 'w')
        f.write('Regular log: \n\n\n')
        f.write(self.temp_log)
        f.write('\n\n\nError log: \n\n\n')
        f.write(self.error_log)
        f.write('\n\n\nWarning log: \n\n\n')
        f.write(self.warning_log)
        f.close()





