import pandas as pd
import numpy as np
from scipy import stats
from FreeBack.post import matplot
import datetime, copy

# 因子标准化函数

# 排序值均匀映射到(0,1]
def Rank(factor):
    # 因子排名
    rank = factor.groupby('date').rank()
  # normlize
    return rank/rank.groupby('date').max()

# 将任意因子转化为正态分布
# 转换为正态分布后默认产生3sigma内的样本，99.7% p=0.003
def Gauss(factor, p=0.003):
    rank = factor.groupby('date').rank()
    continuous = p/2+(1-p)*(rank-1)/(rank.groupby('date').max()-1)
    return continuous.map(lambda x: stats.norm.ppf(x))

# 标准化
def Norm(factor):
    return (factor - factor.groupby('date').mean())/factor.groupby('date').std()

# 因子降频（日频因子换月频）
def resample(factor):
    factor.name=0
    df = pd.DataFrame(factor)
    df['month'] = df.index.map(lambda x:x[0].month)
    df['yesterday_month'] = df['month'].groupby('code').shift()
    df = df.fillna(df.index[0][0].month)
    df['monthly'] = df.apply(lambda x: x[0] if x.month!=x.yesterday_month else np.nan, axis=1)
    df = df.groupby('code').fillna(method='ffill')
    df = df.groupby('code').fillna(method='bfill')
    return df['monthly']



# 为了使得并行回测尽可能与事件驱动框架结果接近：
# 1. 停牌。 当T日x停牌，因子需要对x调仓，事实上x的仓位需要一直等到x复牌才能发生调整。
# 得到factor(确定持仓), price（确定权重）, price_return（确定收益率）分别对应的market_factor, market_price, market_return
# 如果需要排除需要在market中添加一列“alpha-keep”为True为保留，False为排除
def get_market(market):
    if 'alpha-keep' not in market.columns:
        #return market, market, market[market['vol']!=0]
        #return market[market['vol']!=0], market
        return market.copy(), market.copy()
    else:
    ## market_price 为了防止未来函数保留alpha-keep为False的第一个记录
    #    select = market[['alpha-keep']]
    #    # 获取alpha-keep的滚动和，第一次出现时记为2
    #    # inde必须为 'code'和'date'，并且code内部的date排序
    #    select = select.reset_index()
    #    select = select.sort_values(by='code')
    #    select = select.set_index(['code','date'])
    #    select = select.sort_index(level=['code','date'])
    #    # 计算sum
    #    select['alpha-keep_RollingSum_2'] =  select.groupby('code', sort=False).rolling(2)['alpha-keep'].sum().values
    #    # 将index变回 date code
    #    select = select.reset_index()
    #    select = select.sort_values(by='date')
    #    select = select.set_index(['date','code'])
    #    select = select.sort_index(level=['date','code'])
    #    #select = my_pd.cal_RollingSum(select, 'alpha-keep', 2)
    #    # 第一次出现的为np.nan，alpha-keep为True时改为2，否则改为1 
    #    def replace(keep):
    #        if keep:
    #            return 2
    #        else:
    #            return 1
    #    select['alpha-keep_RollingSum_2'] = select.apply(lambda x: replace(x['alpha-keep']) if np.isnan(x['alpha-keep_RollingSum_2'])  else x['alpha-keep_RollingSum_2'], axis=1) 
    #    select_index =  select[select['alpha-keep'] | (select['alpha-keep_RollingSum_2']==1)].index
    #    market_price = market.loc[select_index] 
    #    # 确定持仓,排除的不要
    #    #market_factor = market[market['alpha-keep']]
        market_factor = market[market['alpha-keep']]
        #return market_factor, market_price, market_price[market_price['vol']!=0]
        return market_factor.copy(), market.copy()



# 因子投资组合
class Portfolio():
# 当日收益率为当日收盘价相对昨日收盘价收益率*前一日持仓权重
# 作弊模式    当日因子确定当日持仓 
# df_market.pivot_table('close','date','code')
# df_market.pivot_table('open','date','code').shift(-1)
# 不开启作弊  当日因子确定明日持仓 使用开盘价结算 
# df_market.pivot_table('open', 'date', 'code') 
# holdweight 持仓权重矩阵  例如流通市值
# comm 不影响结果，仅仅在result中给出多头费后年化收益率 
    def __init__(self, factor, price, holdweight=None, cheat = True, comm=0, norm=True, justdivide=False):
        self.comm = comm
        self.cheat = cheat
        self.norm = norm
        self.justdivide=justdivide
#        # 先按照截面排序归一化
        if norm:
            self.factor = Rank(pd.DataFrame(factor.rename('factor')))
        else:
            self.factor = pd.DataFrame(factor.rename('factor'))
        self.variable = factor
        # 一个确定持有张数（不去除停牌），一个确定收益率(去除停牌)
        self.price = price
        if type(holdweight) != type(None):
            # 退市后即为0
            holdweight = holdweight.fillna(0)
            self.holdweight = holdweight.apply(lambda x: x/x.sum(), axis=1)
        else:
            self.holdweight = None
        #if self.cheat:
            # 每日收益率(当日收盘相比上日收盘)
        returns = price/price.shift() - 1
        returns = returns.fillna(0)
        self.returns = returns
        #else:
        #    # 次日开盘相对当日开盘收益率
        #    returns = price_return.shift(-1)/price_return - 1
        #    returns = returns.fillna(0)
        #    self.returns = returns
        # 结果dataframe 行：时间周期  列：IC、ICIR（分组计算的IC、ICIR(非rank)）、 多空组合收益、多头收益、 等权收益、
        # 考虑换手率多空收益、 夏普、 多空平均换手率
        self.result = pd.DataFrame(columns=['group IC', 'group ICIR', 'L&S return', 'L return', 'market return', 
                'L&S sharpe', 'L sharpe', 'market sharpe', 'real return', 'real sharpe', 'turnover'])
        self.result.index.name = 'holding period'
# 全部区间 return
# divide
    def run(self, divide = (0, 0.2, 0.4, 0.6, 0.8, 1), periods=(1, 5, 20)):
        self.periods = periods
        # 最后一个区间为（0，1）表示等权配置收益指数
        # 如果是list则直接为a_b
        if type(divide) == type(list()):
            # 代理变量绝对值
            if self.norm == True:
                self.threshold = [self.variable.groupby('date').quantile(divide[0][0])] +\
                  [self.variable.groupby('date').quantile(i[1]) for i in divide]
            else:
                self.threshold = [ i for i in divide]
            self.a_b = divide
        # 选取factor区间[(0,0.2),(0.2,0.4)...]
        else:
            if self.norm==True:
                self.threshold = [self.variable.groupby('date').quantile(i) for i in divide]
            else:
                self.threshold = [i for i in divide]
            self.a_b = [(divide[i],divide[i+1]) for i in range(len(divide)-1)]
        # 如果justdivide = True不计算(0，1)
        if not self.justdivide:
            if self.norm == True:
                self.a_b = self.a_b + [(0,1)]
            else:
                self.a_b = self.a_b + [(self.factor.min().values[0], self.factor.max().values[0])]
    # 生成持仓表 -> 获得 df_contri(index date  columns code) -> 获得净值每日对数收益率 -> 获得换手率
    # 全部为矩阵操作
        self.matrix_hold()
        self.matrix_contri()
        self.matrix_lr()
        self.matrix_holdn()
        self.matrix_turnover()
        if not self.justdivide:
            self.get_result()
# plot
# 因子组合收益（单边做多，考虑交易成本（默认单边万7））
    def HoldReturn(self, i_period, dateleft=None, dateright=None, cost=0):
        if dateleft==None:
            dateleft = self.factor.index[0][0]
        if dateright==None:
            dateright = self.factor.index[-1][0]
        plt, fig, ax = matplot()
        ax2 = ax.twinx()
        # 画图曲线颜色和透明度区分
        # 不包含等权指数
        if self.justdivide:
            number = len(self.a_b)
        else:
            number = len(self.a_b)-1
        number0 = int(number/2)
        number1 = number - number0
        #前一半为绿色，后一半为红色 （做多因子数值高组，做空因子数值低组）
        color_list = ['C2']*number0 + ['C3']*number1
        # 颜色越靠近中心越浅
        alpha0 = (np.arange(number0)+1)[::-1]/number0
        alpha1 = (np.arange(number1)+1)/number1
        alpha_list = np.append(alpha0, alpha1)
        for i in range(number):
            returns = self.mat_returns[i_period][i].loc[dateleft:dateright]
#            ax.plot((1+returns).cumprod(), label=str(self.a_b[i]), alpha=0.3)
            turnover = self.mat_turnover[i_period][i].loc[dateleft:dateright]
            holdn = self.mat_holdn[i_period][i].loc[dateleft:dateright]
            # 真实净值变化
            returns = (1-turnover.shift().fillna(0)*cost/10000)*(1+returns)
            #ax.plot(returns.cumprod(), label=str(self.a_b[i])+' 换手率=%.1f'%(turnover.mean()*250))
            ax.plot(returns.cumprod(), c=color_list[i], alpha=alpha_list[i],\
                    label=str(self.a_b[i])+' 换手率=%.1f'%(turnover.mean()*250))
            # 持有数量
            ax2.plot(holdn, c=color_list[i], alpha=alpha_list[i], ls='--')
        if not self.justdivide:
            # 等权指数
            returns = self.mat_returns[i_period][-1].loc[dateleft:dateright]
            turnover = self.mat_turnover[i_period][-1].loc[dateleft:dateright]
            returns = (1-turnover.shift().fillna(0)*cost/10000)*(1+returns)
            ax.plot(returns.cumprod(), c='C1',\
                    label='等权指数 '+' 换手率=%.1f'%(turnover.mean()*250))
        ax.legend()
        ax.set_title('调整频率: %d 日'%self.periods[i_period])
        ax.set_ylabel('累计净值')
        ax.set_xlim(dateleft, dateright)
        ax.set_xlabel('日期')
        ax2.set_ylabel('持有数量')
        plt.gcf().autofmt_xdate()
        plt.savefig("HoldReturn.png")
        plt.show()
# 各组对数收益率-等权对数收益率
    def LogCompare(self, i_period):
        plt, fig, ax = matplot()
        benchmark = self.mat_lr[i_period][-1].cumsum()
        # 画图曲线颜色和透明度区分
        # 等权指数不画
        number = len(self.a_b)-1
        number0 = int(number/2)
        number1 = number - number0
        #前一半为绿色，后一半为红色 （做多因子数值高组，做空因子数值低组）
        color_list = ['C2']*number0 + ['C3']*number1
        # 颜色越靠近中心越浅
        alpha0 = (np.arange(number0)+1)[::-1]/number0
        alpha1 = (np.arange(number1)+1)/number1
        alpha_list = np.append(alpha0, alpha1)
        for i in range(number):
            log_return = self.mat_lr[i_period][i].cumsum()
            ax.plot(log_return - benchmark, label=str(self.a_b[i]) + '%.1f'%(self.mat_turnover[i_period][i].mean()*250),
                    c=color_list[i], alpha=alpha_list[i])
        # 因子收益
        LS = (self.mat_lr[i_period][-2] - self.mat_lr[i_period][0]).cumsum()
        factor_return =  100*(np.exp(LS.iloc[-1])**(365/(LS.index[-1]-LS.index[0]).days)-1) 
        ax.plot(LS, c='C0', label='L&S  anu.{r:.2f}%'.format(r=factor_return))
        ax.legend()
        ax.set_title('Period: %d bar(s)'%self.periods[i_period])
        ax.set_ylabel('Cumulative Log Return')
        ax.set_xlabel('Date')
        plt.gcf().autofmt_xdate()
        plt.savefig("LogCompare.png")
        plt.show()
# 各组分组因子值阈值和数量
    def FactorThreshold(self):
        plt, fig, ax = matplot()
        # 颜色与LogCompare中相同，最大值和最小值为橙色C1
        # 等权指数不画
        number = len(self.a_b)-1
        number0 = int(number/2)
        number1 = number - number0
        #前一半为绿色，后一半为红色 （做多因子数值高组，做空因子数值低组）
        color_list = ['C2']*number0 + ['C3']*number1
        # 颜色越靠近中心越浅
        alpha0 = (np.arange(number0)+1)[::-1]/number0
        alpha1 = (np.arange(number1)+1)/number1
        alpha_list = np.append(alpha0, alpha1)
        ax2 = ax.twinx()
        #ax.plot(self.threshold[0], label='low bound',
        #        c='C1')
        for i in range(number):
            # 按分位数分组则显示因子值，按因子值分组则显示分位数
            if self.norm==True:
                #ax.plot(self.threshold[i+1].rolling(20).mean(), label=str(self.a_b[i][1]),
                #    c=color_list[i], alpha=alpha_list[i])
                ax.plot(self.threshold[i+1], label=str(self.a_b[i][1]),
                    c=color_list[i], alpha=alpha_list[i])
            else:
                line = pd.Series(index=self.returns.index).copy()
                line.loc[:] = self.threshold[i+1]
                ax.plot(line, label=str(self.a_b[i][1]),
                    c=color_list[i], alpha=alpha_list[i])
            ax2.plot(self.group_number[i], ls='--', 
                    c=color_list[i], alpha=alpha_list[i])
        ax.legend()
        ax.set_ylabel('Factor thershold')
        # 避免显示异常值
        ax2.set_ylabel('Factor group stock number')
        ax.set_xlabel('Date')
        ax.set_xlim(self.returns.index[0], self.returns.index[-1])
        plt.gcf().autofmt_xdate()
        plt.savefig("FactorThreshold.png")
        plt.show()
# mat[period][factor range]  list[factor range]
# 获得每个持仓周期 每个因子区间的 hold （虚拟持仓 只保证比例关系正确, 和为1)  a_b factor range from a to b 区间内市值等权重
    def matrix_hold(self):
    # 每个bar按标的等权配置需要的持仓
        factor = self.factor.reset_index()
        # 选取因子值 满足a_b list中全部条件的 放置于list_hold (前开后闭，与Rank函数返回的(0，1]对应)
        bar_hold = [factor[(i[0]<factor['factor']) & (factor['factor']<=i[1])] for i in self.a_b]
        self.group_number = [i.groupby('date').count() for i in bar_hold]
        # 在date没有出现的code补np.nan
        # 每个bar持仓表，如果不开启作弊则统一向后移动一个bar
        if self.cheat == True:
            bar_hold = [i.pivot_table('factor', 'date', 'code') for i in bar_hold]
        else:
            bar_hold = [i.pivot_table('factor', 'date', 'code').shift() for i in bar_hold]
        # 非null的持仓数量为 1/price（持仓金额相等）
        bar_hold = [i.isnull() for i in bar_hold]
        bar_hold = [i.replace([True,False],[0,1])*(1/self.price) for i in bar_hold]
        bar_hold = [i.fillna(0) for i in bar_hold]
        # 当holdweight不为None时考虑此权重
        if type(self.holdweight) != type(None):
            bar_hold = [i*self.holdweight for i in bar_hold]
        # matrix hold
        mat_hold = []
        for period in self.periods:
            # 以period为周期 调整持仓的持仓表
            # 选取的index  period = 3  0,0,0,3,3,3,6...
            list_take_hold = [[hold.index[int(i/period)*period] for i in range(len(hold.index))]
                    for hold in bar_hold]
            list_hold = [bar_hold[i].loc[list_take_hold[i]]
                    for i in range(len(bar_hold))]
            # 复原index
            for hold in list_hold:
                hold.index = bar_hold[0].index
            mat_hold.append(list_hold)
        self.mat_hold = mat_hold
# 每个时间持仓标的数量
    def matrix_holdn(self):
        mat_holdn = []
        for period_list in self.mat_hold:
            list_holdn = [(i!=0).sum(axis=1) for i in period_list]
            mat_holdn.append(list_holdn)
        self.mat_holdn = mat_holdn 
# matrix contri     每日净值对数收益率： np.log(matrix_contri[i_period][i_a_b].sum(axis=1)+1)
    def matrix_contri(self):
        mat_weight = []
        mat_contri = []
        for list_hold in self.mat_hold: 
            # 合约市值权重  不是真实的市值 
            list_cap = [hold * self.price for hold in list_hold]
            list_weight = [cap.apply(lambda x: x/x.sum(), axis=1).fillna(0) for cap in list_cap]
            #if self.cheat:
            #    list_contri = [(weight*self.returns).fillna(0) for weight in list_weight]
            ## 当日收益(昨日weight)
            #else:
            list_contri = [(weight.shift()*self.returns).fillna(0) for weight in list_weight]
            # 或者乘以次日收益
#            list_contri = [(weight*self.returns.shift(-1)).fillna(0) for weight in list_weight]
            mat_contri.append(list_contri)
            mat_weight.append(list_weight)
        self.mat_contri = mat_contri
        self.mat_weight = mat_weight
# matrix logreturn  and returns
    def matrix_lr(self):
        mat_lr = []
        mat_returns = []
        for list_contri in self.mat_contri:
            list_returns = [contri.sum(axis=1) for contri in list_contri]
            list_lr = [np.log(returns+1) for returns in list_returns]
            mat_lr.append(list_lr)
            mat_returns.append(list_returns)
        self.mat_lr = mat_lr
        self.mat_returns = mat_returns
    def matrix_turnover(self):
        mat_turnover = []
        #for list_hold in self.mat_hold:
        #    # 持仓变化 初始期为0
        #    list_delta_hold = [hold - hold.shift().fillna(0) for hold in list_hold]
        #    # 成交量
        #    list_amount = [np.abs(delta_hold*self.price) for delta_hold in list_delta_hold]
        #    list_amount = [amount.sum(axis=1) for amount in list_amount]
        #    # 市值(等于持仓个数)
        #    #list_cap = [(hold*self.price).sum(axis=1) for hold in list_hold]
        #    list_cap = [(hold != 0).sum(axis=1) for hold in list_hold]
        #    # 换手率  如果清仓则会计算为np.inf 替换为1
        #    list_turnover = [list_amount[i]/list_cap[i] for i in range(len(list_hold))]
        #    list_turnover = [i.apply(lambda x: x if x!=np.inf else 1) for i in list_turnover]
        #    # 年化换手率
#       #     list_turnover = [i.mean()*250 for i in list_turnover]
        #    mat_turnover.append(list_turnover)
        # 假设当期没有换仓的权重与当期权重的差值即为换手率
        # 换仓周期
        for i in range(len(self.mat_weight)):
            list_weight = self.mat_weight[i]
            list_contri = self.mat_contri[i]
            list_NoAdjustWeight = [list_weight[j].shift().fillna(0) + list_contri[j] for j in range(len(list_weight))]
            list_NoAdjustWeight = [NoAdjustWeight.div(NoAdjustWeight.sum(axis=1), axis='rows') \
                                   for NoAdjustWeight in list_NoAdjustWeight]
            list_NoAdjustWeight = [NoAdjustWeight.fillna(0) for NoAdjustWeight in list_NoAdjustWeight]
            list_turnover = [np.abs((list_weight[j]-list_NoAdjustWeight[j])).sum(axis=1) for j in range(len(list_weight))]
            # 年化换手率
            #list_turnover = [i.mean()*250 for i in list_turnover]
            mat_turnover.append(list_turnover)
        self.mat_turnover = mat_turnover
    def get_result(self):
        # 每一个period
        for i in range(len(self.periods)):
            #  每一个分组 每个时刻的每个分组收益率对其分组序号做回归
            df_corr = pd.DataFrame()
            for j in range(len(self.a_b)-1):
                returns = pd.DataFrame(self.mat_returns[i][j])
                returns['factor']=j
                df_corr = pd.concat([df_corr, returns])
            df_corr = df_corr.groupby('date').corr(method='spearman')
            IC_series = df_corr.loc[(slice(None), 'factor'), 0]
            # 因子指标
            IC = IC_series.mean()
            ICIR = IC/IC_series.std()
            # 年化收益率和sharpe ratio
            duryears = (self.returns.index[-1] - self.returns.index[0]).days/365
            # 多空组合
            LS_returns = self.mat_lr[i][-2] - self.mat_lr[i][0]
            LS_std = LS_returns.std()*np.sqrt(250)
            LS_return_annual = np.exp(LS_returns.sum()/duryears) - 1
            LS_sharpe = (LS_return_annual-0.03)/LS_std
            # 多头组合
            L_returns = self.mat_lr[i][-2] 
            L_std = L_returns.std()*np.sqrt(250)
            L_return_annual = np.exp(L_returns.sum()/duryears) - 1
            L_sharpe = (L_return_annual-0.03)/L_std
            # 市场组合（等权）
            M_returns = self.mat_lr[i][-1]
            M_std = M_returns.std()*np.sqrt(250)
            M_return_annual = np.exp(M_returns.sum()/duryears) - 1
            M_sharpe = (M_return_annual-0.03)/M_std
            # 多头换手率
            turnover = self.mat_turnover[i][-2].mean()*250
            # 考虑换手率造成的交易成本后的多头收益率
            real_return = (L_return_annual+1)*(1-self.comm/10000)**(turnover) - 1
            real_sharpe = (real_return-0.03)/L_std
            record = {'group IC':IC, 'group ICIR':ICIR, 'L&S return': LS_return_annual, 
                      'L return':L_return_annual, 'market return':M_return_annual, 
                      'L&S sharpe':LS_sharpe, 'L sharpe':L_sharpe, 'market sharpe':M_sharpe, 
                        'real return':real_return, 'real sharpe':real_sharpe, 'turnover':turnover}
            self.result.loc[self.periods[i]] = record



def cal_CrossReg(df_, x_name, y_name, series=False):
    df = copy.copy(df_)
    name = y_name + '-' + x_name + '--alpha'
    beta = df.groupby('date').apply(lambda x: ((x[y_name]-x[y_name].mean())*(x[x_name]-x[x_name].mean())).sum()/((x[x_name]-x[x_name].mean())**2).sum())
    gamma = df.groupby('date').apply(lambda x: x[y_name].mean() - beta[x.index[0][0]]*x[x_name].mean())
    ##r = df.groupby('date').apply(lambda x: np.sqrt(((gamma[x.index[0][0]]+x[x_name]*beta[x.index[0][0]] - x[y_name].mean())**2).sum()/(((gamma[x.index[0][0]]+x[x_name]*beta[x.index[0][0]] - x[y_name].mean())**2).sum() + ((x[y_name]-(gamma[x.index[0][0]] + x[x_name]*beta[x.index[0][0]]))**2).sum()))) 
    #r = df[[x_name, y_name]].groupby('date').corr()
    #r = r.loc[(slice(None), x_name), y_name]
    #r = r.reset_index()[['date', y_name]].set_index('date')[y_name]
    r = df[[x_name, y_name]].groupby('date').corr().loc[(slice(None), x_name), y_name].reset_index()[['date', y_name]].set_index('date')[y_name]

    df[name] = df.groupby('date').apply(lambda x: x[y_name] - beta[x.index[0][0]]*x[x_name] - gamma[x.index[0][0]]).values

    if series:
        return df, beta, gamma, r
    else:
        return df
    
# 回归法
# 此处回归获得的因子收益率即为回归的斜率，对应的是等权做多/做空因子值（标准化后）
# 大于+/小于-0.302或者因子值最大/最小38.13%(61.87%)的组合收益
# 直接market_factor标准的market以及因子column名
class Reg():
    # factor_name为IC_series列名
    def __init__(self, factor, price, periods=(1, 3, 5, 10, 20), factor_name = 'alpha0'):
        self.price = price
        self.periods = periods
        factor = Gauss(factor)
        self.factor = factor
        factor = pd.DataFrame(factor.rename('factor'))
        # 输出结果 列：因子指标  行：时间周期
        result = pd.DataFrame(columns = ['absIC', 'IC', 'ICIR', 'annual returns', 'sharpe'])
        result.index.name='period'
        # 多周期IC\因子收益率序列
        IC_dict = {}
        fr_dict = {}
        # 每日回归截距
        gamma_dict = {}
        cross_dict = {}
        for period in self.periods:
            # 预测收益率  预测n期收益率
            returns = (price.shift(-period) - price)/price
            # 预测因子出现之后间隔n期的收益率
            #returns = ((price.shift(-1) - price)/price).shift(1-period)
            # 对数收益率
            #returns = np.log(price.shift(-period)/price)-1
            returns = returns.reset_index().melt(id_vars=['date']).sort_values(by='date').set_index(['date','code']).dropna()
            # 合并df
            df_corr = pd.concat([factor, returns], axis=1).dropna()
            cross_dict[period] = df_corr
            ## 计算IC序列
            #df_corr = df_corr.groupby('date').corr(method='spearman')
            #IC_series = df_corr.loc[(slice(None), 'factor'), 'value']
            #IC_dict[period] = IC_series
            df, beta, gamma, r = cal_CrossReg(df_corr, 'factor', 'value', True)
            gamma_dict[period] = gamma
            # 因子指标
            IC_dict[period] = r
            IC = r.mean()
            ICIR = IC/r.std()
            absIC = (abs(r)).mean()
            # 因子收益率(单位预测周期 1day)
            fr_dict[period] = beta/period
            fr = beta.mean()/period
            #fr_dict[period] = beta
            #fr = beta.mean()
            frIR = fr/beta.std()
            record = {'absIC':absIC, 'IC':IC, 'ICIR':ICIR, 'annual returns':250*fr, \
                      'sharpe':np.sqrt(250)*frIR}
            result.loc[period] = record
        self.IC_dict = IC_dict
        self.fr_dict = fr_dict
        self.cross_dict = cross_dict
        self.gamma_dict = gamma_dict
        self.result = result
        display(result)
        # 多周期平均IC序列
        #IC_series = IC_dict[periods[0]]
        #for period in periods[1:]:
        #    IC_series += IC_dict[period]
        #IC_series =  IC_series/len(periods)
        #self.IC_series = IC_series.reset_index()[['date', 'value']].set_index('date').rename(columns={'value':factor_name})
    # 因子QQ图
    def QQ(self, date=None):
        plt, fig, ax = matplot(w=6, d=4)
        if date==None:
            norm_dis = pd.Series(np.random.randn(len(self.factor))).sort_values()
            ax.scatter(norm_dis, self.factor.sort_values())
        else:
            norm_dis = pd.Series(np.random.randn(len(self.factor.loc[date]))).sort_values()
            ax.scatter(norm_dis, self.factor.loc[date].sort_values())
        ax.plot(norm_dis, norm_dis, c='C3', ls='--')
        ax.set_title('Q-Q plot')
        ax.set_aspect('equal')
        plt.show()
    # 因子收益率
    def factor_return(self, period=1, rolling_period=20):
        plt, fig, ax = matplot()
        ax.plot(250*self.fr_dict[period].cumsum(), label='累计因子收益率', c='C0')
        ax.legend(loc='lower left')
        ax2 = ax.twinx()
        ax2.plot(250*self.fr_dict[period].rolling(rolling_period).mean(), label='滚动因子收益率', c='C1')
        ax2.legend(loc='lower right')
        ax.set_xlim(self.factor.index[0][0], self.factor.index[-1][0])
        plt.show() 
    # 截面因子与收益率（散点图）
    def cross(self, date=None, period=1):
        plt, fig, ax = matplot()
        df_corr = self.cross_dict[period]
        beta = self.fr_dict[period]*period
        gamma = self.gamma_dict[period]
        r = self.IC_dict[period]
        if type(date)==type(None):
            ax.scatter(df_corr['factor'].values, df_corr['value'].values)
            ax.plot(np.linspace(-3,3,100), beta.mean()*np.linspace(-3,3,100) + gamma.mean(), c='C3')
        else:
            ax.scatter(df_corr.loc[date]['factor'], df_corr.loc[date]['value'])
            ax.plot(np.linspace(-3,3,100), beta.loc[date]*np.linspace(-3,3,100) + gamma.loc[date], c='C3')
            plt.title('r = %.2lf beta(万) = %.2lf gamma(万) = %.2lf'%(r.loc[date], beta.loc[date]*10000, gamma.loc[date]*10000))
        plt.show()
    # 因子换手率（因子自相关系数组合）
    def turnover(self):
        self.corr_dic = {}
        for period in self.periods:
            # 初始位置
            factor_original = self.factor.copy()
            factor_original.name = 'original'
            factor_latter = self.factor.groupby('code').shift(period).copy()
            factor_latter.name = 'latter'
            self.corr_dic[period] = pd.concat([factor_original, factor_latter], axis=1).groupby('date').corr(method='pearson').loc[:, 'original', :]['latter'].mean()

        # 直接从因子排序值获得换手率（或者类似指标）
        factor_rank = Rank(self.factor)
        hold_ratio = factor_rank/factor_rank.groupby('date').sum()
        turnover = abs(hold_ratio - hold_ratio.shift()).groupby('date').sum()

        plt, fig, ax = matplot()
        ax.plot(turnover*100)
        ax.set_ylabel('(%)')
        ax.set_title('因子换手率')
        ax.set_xlim(turnover.index[0], turnover.index[-1])
        plt.show()















'''# 单独对某因子策略分析
# 接受assess运行后的实例
class Post():
    rf = 0.03
# which对应mat_**[,]的因子策略，benchmark 如果不加入则默认为等权组合   交易成本 默认为万0  
    def __init__(self, assess,which=(0,0), benchmark=None, cost=0):
    # 传递数据
        self.cost = cost
        self.contri = assess.mat_contri[which[0]][which[1]]
        self.returns = assess.mat_returns[which[0]][which[1]]
        self.turnover = assess.mat_turnover[which[0]][which[1]]
        self.holdn = assess.mat_holdn[which[0]][which[1]]
#        # 持仓 Series
#        self.hold = assess.mat_hold[which[0]][which[1]]
#        s = pd.Series(dtype='float64')
#        for d in self.hold.index:
#            series = pd.Series({d:list(self.hold.columns[(self.hold != 0).loc[d]])})
#            s = pd.concat([s,series])
#        self.holdlist = s
        # 真实净值收益
        self.returns = (1-self.turnover.shift().fillna(0)*self.cost/10000)*(1+self.returns) 
        # 如果benchmark没有则默认取等权指数
        if type(benchmark) == type(None):
            self.benchmark = np.exp(assess.mat_lr[which[0]][-1].cumsum())
        else:
            self.benchmark = benchmark
    # 计算数据
    # 净值
        self.net = self.returns.cumprod()
    # 年化收益率
        years = (self.net.index[-1]-self.net.index[0]).days/365
        return_total = self.net[-1]/self.net[0]
        self.return_annual = return_total**(1/years)-1
    # 年化波动率 shrpe
        self.std_annual = np.exp(np.std(np.log(self.returns))*np.sqrt(250)) - 1
        self.sharpe = (self.return_annual - self.rf)/self.std_annual
    # 回撤
        a = np.maximum.accumulate(self.net)
        self.drawdown = (a-self.net)/a
    # 持仓贡献分析
        # 各合约对组合贡献（先将contri收益率转化为对数收益率，然后按合约各自求和）
        self.contribution = (self.contri+1).prod() -1
        # 各合约持仓bars数量
        self.hold_bars = (self.contri!=0).sum()

# 净值曲线
    def pnl(self, timerange=None, filename=None):
        plt, fig, ax = matplot()
        # 只画一段时间内净值（用于展示局部信息,只列出sharpe）
        if type(timerange) != type(None):
            # 时间段内净值与基准
            net = self.net.loc[timerange[0]:timerange[1]]
            returns = self.returns.loc[timerange[0]:timerange[1]]
            benchmark = self.benchmark.loc[timerange[0]:timerange[1]]
            # 计算夏普
            years = (timerange[1]-timerange[0]).days/365
            return_annual = (net[-1]/net[0])**(1/years)-1
            std_annual = np.exp(np.std(np.log(returns))*np.sqrt(250)) - 1 
            sharpe = (return_annual - self.rf)/std_annual
            ax.text(0.7,0.05,'Sharpe:  {}'.format(round(sharpe,2)), transform=ax.transAxes)
            ax.plot(net/net[0], c='C0', label='p&l')
        # colors of benchmark
            colors_list = ['C4','C5','C6','C7']
            for i in range(len(benchmark.columns)):
                ax.plot((benchmark[benchmark.columns[i]]+1).cumprod(), 
                        c=colors_list[i], label=benchmark.columns[i])
        else: 
    #评价指标
            ax.text(0.7,0.05,'年化收益率: {}%\n夏普比率:   {}\n最大回撤:   {}%'.format(
            round(100*self.return_annual,2), round(self.sharpe,2), round(100*max(self.drawdown),2)), transform=ax.transAxes)
        # 净值与基准
            ax.plot(self.net, c='C0', label='策略')
        # benchmark 匹配回测时间段
            benchmark = self.benchmark.loc[self.net.index[0]:]
        # colors of benchmark
            colors_list = ['C4','C5','C6','C7']
            for i in range(len(benchmark.columns)):
                ax.plot((benchmark[benchmark.columns[i]]+1).cumprod(),  c=colors_list[i], label=benchmark.columns[i])
            plt.legend()
            # 回撤
            ax2 = ax.twinx()
            ax2.fill_between(self.drawdown.index,-100*self.drawdown, 0, color='C1', alpha=0.1)
            ax.set_ylabel('累计净值')
            ax.set_xlabel('日期')
            ax2.set_ylabel('回撤 (%)')
        if type(filename) == type(None):
            plt.savefig('pnl.png')
        else:
            plt.savefig(filename)
        plt.show()

# 收益分解 持有时间
    def disassemble(self, timerange=None):
        # 整个时间还是一段时间内
        if type(timerange) == type(None):
            contribution = self.contribution
        else:
            contri = self.contri[timerange[0]:timerange[1]]
            contribution = (contri+1).prod() -1
        # 去掉从贡献为0的合约 从小到大（负到正）
        contribution = contribution[contribution!=0]
        contribution = contribution.sort_values() 
        # key: code    value: DateIndex
        dict_dateindex = {}
        for code in contribution.index:
            dict_dateindex[code] = self.contri.index[self.contri[code] != 0]
        return dict_dateindex

# 持有时股价表现  输入市场df,合约代码和持有日期，从dict_dateindex获取
    def hold_plot(self, market, code, dateindex):
        plt, fig, ax = matplot()
        # 标的收盘价
        ax.plot(market.loc[:,code,:]['close'], c='C3', label='收盘价')
        ax.plot(market.loc[:,code,:]['Pc'], c='C2', label='转股价值')
        ax.set_ylabel('价格（元）')
        ax.set_xlabel('日期')
        y_low = min(market.loc[:,code,:]['close'].min(), market.loc[:,code,:]['Pc'].min())
        y_high = max(market.loc[:,code,:]['close'].max(), market.loc[:,code,:]['Pc'].max()) 
        ax.set_ylim(y_low*0.95,y_high*1.05)
        # 持有期间绘制阴影
        tradedate = market.reset_index()['date'].unique()
        i = 1
        start = 0
        while i<len(dateindex):
            # 连续交易日（未清仓）
            if np.where(tradedate == dateindex[i])[0][0] == np.where(tradedate == dateindex[i-1])[0][0] + 1:
                i += 1
                continue
            # 交易日不连续（换仓） 
            else:
                ax.fill_between(dateindex[start:i], 0, 9999, facecolor='C1', alpha=0.1)
                start = i
                i += 1
        # 如果不是以交易日不连续结束结束
        i = i-1
        if np.where(tradedate == dateindex[i])[0][0] == np.where(tradedate == dateindex[i-1])[0][0] + 1:
            ax.fill_between(dateindex[start:i], 0, 9999, facecolor='C1', alpha=0.1)
        # 其他数据
        ax2 = ax.twinx()
        ax2.plot(market.loc[:,code,:]['premium']*100, label='转股溢价率(%)')
        ax2.legend(loc='upper right')
        ax.legend(loc='upper left')
        
        plt.savefig('hold_perform.png')

# 收益分解 画图
    def disassemble_plot(self):
        # 去掉0，排序
        contribution = self.contribution
        n_total = len(contribution)
        contribution = contribution[contribution!=0]
        n = len(contribution)
        contribution = contribution.sort_values()

        plt, fig, ax = matplot()
        ax.plot(contribution.values*100, c='C1')
        ax.set_ylabel('净值贡献 (%)')
        ax.text(0.05,0.9, '从%d只标的选取了%d只'%(n_total,n), transform=ax.transAxes)
        ax2 = ax.twinx()
        ax2.plot(self.hold_bars[contribution.index].values, alpha=0.3)
        ax2.set_ylabel('持有天数') 
        plt.savefig('disassemble.png')
        plt.show()'''
