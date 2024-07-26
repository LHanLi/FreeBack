import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import FreeBack as FB
import os

# 结果文件保存于output中
def check_output():
    if 'output' in os.listdir():
        pass
    else:
        os.mkdir('output')

# 该模块包含：
# 因子计算常用函数
# 单因子与多因子检验模块（组合法、回归法）


#######################################################################################################
####################################### 因子计算常用函数 #################################################
#######################################################################################################
# 无特殊说明下 factor,price的格式为pd.Series,其中index的格式为multiindex (date code)
# Rank      
# 每日全部index的因子值从小到大排序,均匀映射到(0,1)
# Norm      
# 将每日因子值在截面上标准化到 \mu = 0 \sigma = 1 分布
# scale     
# 使得截面上因子值满足sum(abs(x)) = 1(a) 
# Gauss     
# 将每日的因子值转化为正态分布
# resample_fill/select  
# 因子降频（日频因子换月频/周频）
# QQ        绘制因子分布的QQ图，观察是否符合正态分布


def Rank(factor, norm=False):
    # 因子排名
    rank = factor.groupby('date').rank()
    if norm:
        return 2*3**0.5*(rank/(rank.groupby('date').max()+1)-0.5)
    return rank/(rank.groupby('date').max()+1)
def Norm(factor):
    return (factor - factor.groupby('date').mean())/factor.groupby('date').std()

def scale(factor, a=1):
    return a*factor/factor.groupby('date').apply(lambda x:abs(x).sum())
# 通过正态分布累计概率函数的逆函数将[p,1-p]的均匀分布转化为正态分布，转换为正态分布后默认产生3sigma内的样本，99.7% p=0.003
def Gauss(factor, p=0.003, slice=False):
    # 开启slice选项时为仅转化一个截面启代表仅有一个截面数据
    if not slice:
        rank = factor.groupby('date').rank()
        continuous = p/2+(1-p)*(rank-1)/(rank.groupby('date').max()-1)
        def func(ser):
            return ser.map(lambda x: stats.norm.ppf(x))
        result = FB.my_pd.parallel(continuous, func)
        # 如果所有值相同则替换为0
        if_same = result.groupby('date').apply(lambda x: (~x.duplicated()).sum())
        result.loc[if_same[if_same==1].index] = 0
        return result
    else:
        rank = factor.rank()
        continuous = p/2+(1-p)*(rank-1)/(rank.max()-1)
        return continuous.map(lambda x: stats.norm.ppf(x))
# 每月、每周内的因子值替换为月初、周初因子值
def resample_fill(factor, freq='month'):
    factor.name=0
    df = pd.DataFrame(factor)
    df['th'] = df.index.map(lambda x:getattr(x[0], freq))
    df['yesterday_th'] = df['th'].groupby('code').shift()
    df = df.fillna(getattr(df.index[0][0], freq))
    df['after'] = df.apply(lambda x: x[0] if x.th!=x.yesterday_th else np.nan, axis=1)
    df = df.groupby('code').fillna(method='ffill')
    df = df.groupby('code').fillna(method='bfill')
    return df['after']
# 直接只选用原数据的周初、月初值
def resample_select(market, freq='month'):
    if freq=='month':
        # 前一天月份和今天不同，后一天月份和今天相同
        market['th'] = market.index.get_level_values(0).map(lambda x: x.month)
    elif freq=='week':
        market['th'] = market.index.get_level_values(0).map(lambda x: x.week)
    return market[(~market['th'].groupby('code').shift().isna())&\
                  (market['th'].groupby('code').shift()!=market['th'])&\
                  (market['th'].groupby('code').shift(-1)==market['th'])].drop(columns='th')
def QQ(factor, date=None):
    plt, fig, ax = FB.display.matplot(w=6, d=4)
    if date==None:
        norm_dis = pd.Series(np.random.randn(len(factor))).sort_values()
        ax.scatter(norm_dis, factor.sort_values())
    else:
        norm_dis = pd.Series(np.random.randn(len(factor.loc[date]))).sort_values()
        ax.scatter(norm_dis, factor.loc[date].sort_values())
    ax.plot(norm_dis, norm_dis, c='C3', ls='--')
    ax.set_title('Q-Q plot')
    ax.set_aspect('equal')
    plt.show()



#######################################################################################################
######################################### 单因子检测 ####################################################
#######################################################################################################
# 1. 组合法， 计算各因子分层组合的收益及其他特征。
    # Porfolio, 因子投资组合表现
    # factorgroup
# 2. 回归法， 假设截面上因子与未来收益率线性相关，计算该线性关系的一系列统计量。
    # Reg

########################################## 组合法 #######################################################

class Portfolio():
# factor 数据类型pd.Series, multiindex(date,code)   T日因子值(可以在此排除涨停板)
# price 数据类型pd.Series, multiindex(date,code) T日交易价格(最宽松情况可以使用当日收盘价，\
# 使用后一天开盘价或者VWAP等是接近实际情况的(全市场数据即可) 
# norm 是否按照因子值的排序来对投资组合分组,默认开启。否则使用原始因子值进行分组 
# divide 
# 数据类型为tuple时，给出全部阈值确定连续分组；
# 数据类型为list， 给出每个分组的前后阈值,例：[(0,0.1),(0.9,1)]表示选取最小10%和最大10%构建组合，前开后闭。
# norm开启时，元素<=1时表示百分比，大于1时表示绝对数量，如(0,0.2)表示持有钱20%，(0，10)表示持有前十只。
# 否则表示因子值的阈值。
# justdivide 是否仅计算给出分组的收益而不计算全市场组合收益，默认计算全市场组合收益(权重由holdweight确定)
# periods (1, 5, 20)同时计算1天5天20天轮动的结果
# holdweight pd.Series, multiindex(data,code)  T日的投资组合权重，默认等权
# comm 每次单边换手的交易成本，默认为0
## 当日收益率为当日收盘价相对昨日收盘价收益率*前一日持仓权重
## comm 不影响结果，仅仅在result中给出多头费后年化收益率 
    def __init__(self, factor, price, norm=True, divide=(0, 0.2, 0.4, 0.6, 0.8, 1),\
                 justdivide=False, periods=(1, ), \
                  holdweight=None, comm=0, returns=None):
        self.comm = comm
        self.norm = norm
        self.justdivide=justdivide
        # 一个确定持有张数（不去除停牌），一个确定收益率(去除停牌)
        self.price = pd.DataFrame(price.rename('price')).pivot_table('price', 'date' ,'code')
        # 每日收益率(当日收盘相比上日收盘)
        if returns is None:
            returns = self.price/self.price.shift() - 1
        self.returns = returns.fillna(0)
#        # 先按照截面排序归一化
        if norm:
            self.factor = Rank(pd.DataFrame(factor.rename('factor')))
        else:
            self.factor = pd.DataFrame(factor.rename('factor'))
        self.variable = factor
        # 组合权重 
        if type(holdweight) != type(None):
            # 没有权重则按0计
            holdweight = holdweight.fillna(0)
            holdweight = pd.DataFrame(holdweight.rename('weight')).pivot_table('weight', 'date' ,'code')
            self.holdweight = holdweight.apply(lambda x: x/x.sum(), axis=1)
        else:
            self.holdweight = None
        # 结果dataframe 行：时间周期  列：IC、ICIR（分组计算的IC、ICIR(非rank)）、 多空组合收益、多头收益、 等权收益、
        # 考虑换手率多空收益、 夏普、 多空平均换手率
        self.result = pd.DataFrame(columns=['group IC', 'group ICIR', 'L&S return', 'L return',\
                 'market return', 'L&S sharpe', 'L sharpe', 'market sharpe', 'real return', \
                    'real sharpe', 'turnover'])
        self.result.index.name = 'holding period'
        # 运行
        self.run(divide, periods)
    def run(self, divide, periods):
        self.divide = divide
        self.periods = periods
        # self.a_b表示对因子分隔的阈值，如果是list则直接为a_b
        if type(self.divide) == type(list()):
            self.a_b = self.divide
        # 如果是tuple则转化成list
        else:
            self.a_b = [(self.divide[i],self.divide[i+1]) for i in range(len(self.divide)-1)]
        # 如果justdivide为False,则增加一个可以选中全部标的的组合
        if not self.justdivide:
            if self.norm == True:
                # 按百分比划分
                if self.a_b[-1][1]<=1:
                    self.a_b = self.a_b + [(0,1)]
                else:
                    self.a_b = self.a_b + [(0, 99999)]
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
# mat[period][factor range]  list[factor range]
# 获得每个持仓周期 每个因子区间的虚拟持仓(只保证比例关系正确) 
# a_b factor range from a to b 区间内市值等权重
    def matrix_hold(self):
    # 每个bar按标的等权配置需要的持仓
        if self.norm:
            if self.a_b[-1][1]<=1:
                look_factor = self.factor.groupby('date').rank(pct=True)
            else:
                print('整数排序')
                look_factor = self.factor.groupby('date').rank()
        else:
            look_factor = self.factor
        look_factor = look_factor.reset_index()
        # 选取因子值 满足a_b list中全部条件的 放置于list_hold (前开后闭，与Rank函数返回的(0，1]对应)
        bar_hold = [look_factor[(i[0]<look_factor['factor']) &\
                                 (look_factor['factor']<=i[1])] for i in self.a_b]
        # 每一组合选中的标的个数
        self.group_number = [i.groupby('date').count() for i in bar_hold]
        # 在date没有出现的code补np.nan
        bar_hold = [pd.DataFrame(i.pivot_table('factor', 'date', 'code'),\
                                 index=self.factor.index.get_level_values(0).unique())\
                                     for i in bar_hold]
        # 等权情况下持有标的的持仓数量为 1/price（持仓金额相等）
        #bar_hold = [(i.isnull().replace([True,False],[0,1])*\
        #             (1/self.price)).fillna(0) for i in bar_hold]
        bar_hold = [((~i.isnull()).astype(int)*\
                     (1/self.price)).fillna(0) for i in bar_hold]
        # 当holdweight不为None时考虑此权重
        if type(self.holdweight) != type(None):
            bar_hold = [i*self.holdweight for i in bar_hold]
        # 当period不等于1时需要按照period调整持仓
        mat_hold = []
        for period in self.periods:
            # 以period为周期 调整持仓的持仓表
            # 选取的index  period = 3  0,0,0,3,3,3,6...
            list_take_hold = [[hold.index[int(i/period)*period]\
                                for i in range(len(hold.index))]
                    for hold in bar_hold]
            list_hold = [bar_hold[i].loc[list_take_hold[i]]
                    for i in range(len(bar_hold))]
            # 提取的index非连续，复原到原来的连续交易日index
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
            # 当日收益(return*昨日weight)
            list_contri = [(weight.shift()*self.returns).fillna(0) for weight in list_weight]
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
        # 假设当期没有换仓的权重与当期权重的差值即为换手率
        # 换仓周期
        for i in range(len(self.mat_weight)):
            list_weight = self.mat_weight[i]
            list_contri = self.mat_contri[i]
            list_NoAdjustWeight = \
                [list_weight[j].shift().fillna(0) + list_contri[j] for j in range(len(list_weight))]
            list_NoAdjustWeight = [NoAdjustWeight.div(NoAdjustWeight.sum(axis=1), axis='rows') \
                                   for NoAdjustWeight in list_NoAdjustWeight]
            list_NoAdjustWeight = [NoAdjustWeight.fillna(0) for NoAdjustWeight in list_NoAdjustWeight]
            list_turnover = \
                [np.abs((list_weight[j]-list_NoAdjustWeight[j])).sum(axis=1) for j in range(len(list_weight))]
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
            # 后续改成简单收益率相减，类似的还有不少-zzq
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

# plot
# 因子组合收益（单边做多，考虑交易成本（默认单边万7））
    def HoldReturn(self, i_period=0, dateleft=None, dateright=None, cost=0):
        if dateleft==None:
            dateleft = self.factor.index[0][0]
        if dateright==None:
            dateright = self.factor.index[-1][0]
        plt, fig, ax = FB.display.matplot()
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
                    label=str(self.a_b[i])+' 换手=%.1f'%(turnover.mean()*250))
            # 持有数量
            #ax2.plot(holdn, c=color_list[i], alpha=alpha_list[i], ls='--')
            ax2.plot(holdn, c=color_list[i], alpha=0.2, ls='--')
        if not self.justdivide:
            # 等权指数
            returns = self.mat_returns[i_period][-1].loc[dateleft:dateright]
            turnover = self.mat_turnover[i_period][-1].loc[dateleft:dateright]
            returns = (1-turnover.shift().fillna(0)*cost/10000)*(1+returns)
            ax.plot(returns.cumprod(), c='C1',\
                    label='等权指数 '+' 换手=%.1f'%(turnover.mean()*250))
        #ax.legend()
        if number<8:
            ax.legend(bbox_to_anchor=(0.5, -0.55), loc=8, ncol=2)
        elif number<10:
            ax.legend(bbox_to_anchor=(0.5, -0.65), loc=8, ncol=2)
        else:
            ax.legend(bbox_to_anchor=(0.5, -0.7), loc=8, ncol=2)
        ax.set_title('调整频率: %d 日'%self.periods[i_period])
        ax.set_ylabel('累计净值')
        ax.set_xlim(dateleft, dateright)
        #ax.set_xlabel('日期')
        ax2.set_ylabel('持有数量')
        plt.gcf().autofmt_xdate()
        check_output()
        plt.savefig('./output/alpha-Portfolio-HoldReturn.png',\
                     bbox_inches='tight')
        plt.show()
# 各组对数收益率-等权对数收益率
    def LogCompare(self, i_period=0, dateleft=None, dateright=None, ifbench=True):
        if dateleft==None:
            dateleft = self.factor.index[0][0]
        if dateright==None:
            dateright = self.factor.index[-1][0]
        plt, fig, ax = FB.display.matplot()
        if ifbench:
            benchmark = self.mat_lr[i_period][-1].cumsum()
        else:
            benchmark = 0
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
            ax.plot(log_return - benchmark, \
                    label=str(self.a_b[i]) + '%.1f'%(self.mat_turnover[i_period][i].mean()*250),
                    c=color_list[i], alpha=alpha_list[i])
        # 因子收益
        LS = (self.mat_lr[i_period][-2] - self.mat_lr[i_period][0]).cumsum()
        factor_return =  100*(np.exp(LS.iloc[-1])**(365/(LS.index[-1]-LS.index[0]).days)-1) 
        ax.plot(LS, c='C0', label='L&S  anu.{r:.2f}%'.format(r=factor_return))
        ax.legend()
        ax.set_title('Period: %d bar(s)'%self.periods[i_period])
        ax.set_ylabel('Cumulative Log Return')
        ax.set_xlabel('Date')
        ax.set_xlim(dateleft, dateright)
        plt.gcf().autofmt_xdate()
        check_output()
        plt.savefig('./output/alpha-Portfolio-LogCompare.png',\
                     bbox_inches='tight')
        plt.show()
# 柱状图
    def Bar(self, i_period=0):
        # 按年度划分收益率
        df_returns = pd.concat(self.mat_lr[i_period], axis=1)\
            .rename(columns=dict(zip(range(len(self.a_b)), self.a_b)))
        df_returns['year'] = df_returns.index.year
        df_returns = 100*(np.exp(df_returns.groupby('year').sum())-1)
        # 作图
        plt, fig, ax = FB.display.matplot()
        x = np.arange(len(df_returns))
        width = 0.08
        # 偏移量
        move = np.arange(-len(self.a_b)*width/2,len(self.a_b)*width/2,width)
        for n in range(len(self.a_b)):
            ax.bar(np.arange(len(df_returns))+move[n], df_returns[self.a_b[n]].values,\
                    width=width, label='%s'%(str(self.a_b[n])))
        ax.legend(bbox_to_anchor=(0.5,-0.3), loc=10, ncol=3)
        plt.xticks(x, list(df_returns.index), fontsize=20)
        ax.set_ylabel("(%)")
        check_output()
        plt.savefig('./output/alpha-Portfolio-Bar.png',\
                     bbox_inches='tight')
        plt.show()
# 各组分组因子值阈值和数量
    def FactorThreshold(self):
        plt, fig, ax = FB.display.matplot()
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
        check_output()
        plt.savefig('./output/alpha-Portfolio-FactorThreshold.png',\
                     bbox_inches='tight')
        plt.show()



# 因子分组统计(可以统计收益率，也可以统计其他变量，例如未来20日的波动率等)
# market, 分组变量1， 分组变量2， 分组数，统计变量，延迟周期
def FactorGroup(market_factor, group_value0, group_value1=None,\
        group_value0_num=5, group_value1_num=3, returns_key='f-returns', delay=0,\
            level0s=None, level1s=None):
    #market_factor = market.copy()
    # 获得分组变量
    # 对于连续因子由小到大等分,滞后delay期。
    if market_factor[group_value0].dtype in [float, np.int64]:
        group0 = 'group_' + group_value0
        market_factor[group0] = Rank(market_factor[group_value0]).groupby('code'\
                                ).shift(delay).dropna().map(lambda x: int(x*group_value0_num))
    # 对于离散因子按原因子分组,滞后delay期。
    else:
        group0 = group_value0
        market_factor[group0] = market_factor[group0].groupby('code').shift(delay).dropna()
    # 单因子分组情形
    if group_value1==None:
        group = market_factor.groupby([group0, 'date'])
        result_returns = group[returns_key].mean()

        plt, fig, ax = FB.display.matplot()
        ax1 = ax.twinx()
        for i in result_returns.index.get_level_values(0).unique():
            ax.plot(result_returns.loc[i].cumsum(), label='group %s, %.2lf'%(i, \
                        100*np.exp(250*result_returns.loc[i].mean())-100))
            ax1.plot(group['close'].count().loc[i], alpha=0.3)
        ax.legend()
        ax.set_ylabel('累计收益率（单利）')
        ax1.set_ylabel('分组个数')
        ax.set_xlim(market_factor.index[0][0], market_factor.index[-1][0])
        check_output()
        plt.savefig('output/FactorGroup.png')
        return
    # 计算指标的分组标签
    if market_factor[group_value1].dtype in [float, np.int64]:
        group1 = 'group_' + group_value1
        market_factor[group1] = Rank(market_factor[group_value1]).groupby('code'\
                                ).shift(delay).dropna().map(lambda x: int(x*group_value1_num))
    else:
        group1 = group_value1
        market_factor[group1] = market_factor[group1].groupby('code').shift(delay).dropna()
    # 分组
    tradeday = market_factor.index.get_level_values(0).unique()
    group = market_factor.groupby([group0, group1, 'date'])
    result_returns = group[returns_key].mean()
    result_num = group['close'].count().loc[:, :, tradeday[-20]:tradeday[-1]]
    if level0s==None:
        level0s = result_returns.index.get_level_values(0).unique()
    if level1s==None:
        level1s = result_returns.index.get_level_values(1).unique()
    # 结果矩阵（收益率、组内个数）
    dict_returns = {}
    dict_num = {}
    for level0 in level0s:
        for level1 in level1s:
            try:
                dict_returns[(level0, level1)] = \
                    100*np.exp(250*result_returns.loc[level0, level1].mean())-100
                dict_num[(level0, level1)] = result_num.loc[level0, level1].mean()
            except:
                dict_returns[(level0, level1)] = 0
                dict_num[(level0, level1)] = 0
    # 将绝对值转化为颜色
    def color_map(x, min_r, max_r):
        if x>0:
            return [1,1-(x-min_r)/max_r,1-(x-min_r)/max_r]
        elif x == 0:
            return [1,1,1]
        else:
            return [1+(x-min_r)/max_r,1,1+(x-min_r)/max_r]
    # 
    plot = np.ones((len(level0s),len(level1s),3))
    plt, fig, ax = FB.display.matplot()
    for level0 in range(len(level0s)):
        for level1 in range(len(level1s)):
            # 先列再行
            plot[level0][level1] = color_map(dict_returns[(level0s[level0], level1s[level1])], \
                                    0.9*min(dict_returns.values()), 1.1*max(dict_returns.values()))
            # 先行再列
            ax.text(level1, level0, 
        round(dict_returns[(level0s[level0], level1s[level1])], 1),
                ha='center', va='center')
            ax.text(level1, level0, 
        '    ' + str(int(dict_num[(level0s[level0], level1s[level1])])),
                ha='left', va='top', fontsize=10, color='C0')
    ax.imshow(plot, aspect='auto')
    ax.set(xticks=list(range(len(level1s))))
    ax.set_xticklabels([i for i in level1s])
    ax.set_xlabel(group1)
    ax.set(yticks=list(range(len(level0s))))
    ax.set_yticklabels([i for i in level0s])
    ax.set_ylabel(group0)
    ax.set_title('双因子分组')
    ax.grid(False)
    plt.savefig('MutiFactorGroup.png')



######################################################### 回归法 ####################################################################



# 截面一元线性回归
def cal_CrossReg(df, x_name, y_name, series=False):
    name = y_name + '-' + x_name + '--alpha'

    # 使用sm模块
    result = df.groupby('date', sort=False).apply(lambda d: sm.OLS(d[y_name], sm.add_constant(d[x_name])).fit())
    
    # 如果d[x_name]中所有数相同为C且不为零，这时params中没有const，x_name为d[y_name].mean()/C
    # rsquared为0
    # 当d[x_name]全为0时，params['const']为0，params[x_name]为d[y_name].mean()
    # rsquared可能为极小的负数
    def func(x, name):
        try:
            return x.params[name]
        except:
            print('sm reg warning')
            return 0
    gamma = result.map(lambda x: func(x, 'const'))
    beta = result.map(lambda x: func(x,x_name))
    r = result.map(lambda x: np.sign(func(x, x_name))*np.sqrt(abs(x.rsquared)))

    if series:
        return beta, gamma, r
    else:
        df[name] = df.groupby('date').apply(lambda x: x[y_name] - beta[x.index[0][0]]*x[x_name] - gamma[x.index[0][0]]).values
        return df
    
# 回归法
# 此处回归获得的因子收益率即为回归的斜率，对应的是等权做多/做空因子值（标准化后）
# 大于+/小于-0.302或者因子值最大/最小38.13%(61.87%)的组合收益
# 直接market_factor标准的market以及因子column名
class Reg():
    # factor_name为IC_series列名
    def __init__(self, factor, price, periods=(1, 5, 20), factor_name = 'alpha0', \
                 gauss=False, point=True):
        #import time
        #start = time.time()
        self.price = pd.DataFrame(price.rename('price')).pivot_table('price', 'date' ,'code')
        self.periods = periods
        self.point = point
        if gauss:
            factor = Gauss(factor)
        else:
            factor = Rank(factor, norm=True)
        self.factor = factor
        self.factor.name = factor_name
        factor = pd.DataFrame(factor.rename('factor'))
        # 输出结果 列：IC绝对值均值， IC均值， ICIR， 年化因子收益率， 年化夏普， 年化换手， 
        # 交易成本万3\10\30
        #   行：时间周期
        result = pd.DataFrame(columns = ['absIC', 'IC', 'ICIR', 'annual return',\
                     'sharpe', 'turnover',\
                'comm3_r', 'comm3_s', 'comm10_r', 'comm10_s'])
        result.index.name='period'
        # 多周期IC\因子收益率序列
        IC_dict = {}
        #rankIC_dict = {}
        fr_dict = {}
        # 每日回归截距
        gamma_dict = {}
        cross_dict = {}
        # 多空单位因子收益率组合平均换手率
        turnover_dict = {}
        #print(time.time()-start)
        #start = time.time()
        for period in self.periods:
            if point: # 预测因子出现之后间隔n期的收益率
                returns = ((self.price.shift(-1) - self.price)/self.price).shift(1-period)
            else: # 预测收益率  预测n期内收益率
                returns = (self.price.shift(-period) - self.price)/self.price
            returns = returns.reset_index().melt(id_vars=['date']).\
                sort_values(by='date').set_index(['date','code']).dropna()
            # 合并df
            df_corr = pd.concat([factor, returns], axis=1).dropna()
            cross_dict[period] = df_corr
            #print(time.time()-start)
            #start = time.time()
            ## 计算IC序列
            beta, gamma, r = cal_CrossReg(df_corr, 'factor', 'value', True)
            gamma_dict[period] = gamma
            #print(time.time()-start)
            #start = time.time()
            # 因子指标
            #rankIC_dict[period] = df_corr.groupby('date').corr(method='spearman')['factor'].loc[:, 'value']
            #rankIC = rankIC_dict[period].mean()
            IC_dict[period] = r
            IC = r.mean()
            ICIR = IC/r.std()
            absIC = (abs(r)).mean()
            # 因子收益率(单位预测周期 1day)
            if point:
                fr_dict[period] = beta
                fr = beta.mean()
            else:
                fr_dict[period] = beta/period
                fr = beta.mean()/period
            frIR = np.sqrt(period)*fr/beta.std()
            # 换手率
            # 多头组合成分
            factor_L = self.factor[self.factor>0.302].copy()
            name = factor_L.name
            factor_L = factor_L.reset_index()
            factor_L[name] = 1
            # 组合权重
            weight_L = factor_L.pivot_table(name, 'date', 'code')
            weight_L = weight_L.div(weight_L.sum(axis=1), axis='rows').fillna(0)
            # 如果未调整period日后的组合权重
            noadjust_weight = (weight_L.shift(period)*(self.price/self.price.shift(period)))[weight_L.columns]
            noadjust_weight = noadjust_weight.div(noadjust_weight.sum(axis=1), axis='rows').fillna(0)
            turnover_L = (abs(weight_L-noadjust_weight).sum(axis=1)).mean()/period
            # 空头组合成分
            factor_S = self.factor[self.factor<-0.302].copy()
            name = factor_S.name
            factor_S = factor_S.reset_index()
            factor_S[name] = 1
            # 组合权重
            weight_S = factor_S.pivot_table(name, 'date', 'code')
            weight_S = weight_S.div(weight_S.sum(axis=1), axis='rows').fillna(0)
            # 如果未调整period日后的组合权重
            noadjust_weight = (weight_S.shift(period)*(self.price/self.price.shift(period)))[weight_S.columns]
            noadjust_weight = noadjust_weight.div(noadjust_weight.sum(axis=1), axis='rows').fillna(0)
            turnover_S = (abs(weight_S-noadjust_weight).sum(axis=1)).mean()/period
            turnover = ((turnover_S+turnover_L)/2).mean()*250
            turnover_dict[period] = turnover
            # 费后收益及夏普
            comm3_return = ((1+250*fr)*(1-3/1e4)**turnover-1)
            comm3_sharpe = comm3_return/(np.sqrt(250)*beta.std()) 
            comm10_return = ((1+250*fr)*(1-10/1e4)**turnover-1)
            comm10_sharpe = comm10_return/(np.sqrt(250)*beta.std()) 
            #record = {'absIC':round(absIC*100,1), 'IC':round(IC*100,1), 'rankIC':round(100*rankIC,1),\
            record = {'absIC':round(absIC*100,1), 'IC':round(IC*100,1),\
                      'ICIR':round(10*ICIR,1), \
                      'annual return':round(250*fr*100,1), \
                      'sharpe':round(np.sqrt(250)*frIR,1), 'turnover':round(turnover,1),\
                'comm3_r':round(100*comm3_return,1), 'comm3_s':round(comm3_sharpe,1),\
                    'comm10_r':round(100*comm10_return,1), 'comm10_s':round(comm10_sharpe,1)}
            result.loc[period] = record
            #print(time.time()-start)
            #start = time.time()
        self.IC_dict = IC_dict
        self.fr_dict = fr_dict
        self.cross_dict = cross_dict
        self.gamma_dict = gamma_dict
        self.result = result
        FB.display(result)
    # 因子收益率
    def factor_return(self, period=1, rolling_period=20):
        plt, fig, ax = FB.display.matplot()
        cumsum_fr = 250*self.fr_dict[period].cumsum()
        ax.plot(cumsum_fr, label='累计因子收益率', c='C0')
        ax.plot(cumsum_fr.rolling(20).min(),\
                 alpha=0.5, c='C2')
        ax.plot(cumsum_fr.rolling(20).max(),\
                  alpha=0.5, c='C3')
        ax.legend(loc='lower left')
        ax.legend(bbox_to_anchor=(0.17, 1.06), loc=10, ncol=1)
        ax2 = ax.twinx()
        ax2.plot(250*self.fr_dict[period].rolling(rolling_period).mean(), label='滚动因子收益率（右）', c='C1')
        #ax2.legend(loc='lower right')
        ax2.legend(bbox_to_anchor=(0.78, 1.06), loc=10, ncol=1)
        ax.set_xlim(self.factor.index[0][0], self.factor.index[-1][0])
        plt.show() 
    # 截面因子与收益率（散点图） n为分级靠档组数
    def cross(self, date=None, period=1, n=100):
        plt, fig, ax = FB.display.matplot()
        df_corr = self.cross_dict[period].copy()
        if self.point:
            beta = self.fr_dict[period]
        else:
            beta = self.fr_dict[period]*period
        gamma = self.gamma_dict[period]
        r = self.IC_dict[period]
        if type(date)==type(None):
            # 如果因子值少于n个（因子值重复过多）则不需要分级靠档
            # 因子值按分位数分级靠档为n组
            if len(df_corr['factor'].unique())<n:
                factor_group = df_corr.groupby('factor')['value'].mean()
            else:
                threshold = [df_corr['factor'].quantile(i/n) for i in range(n+1)]
                label = [(i+j)/2 for i,j in zip(threshold[:-1],threshold[1:])]
                df_corr['factor_group'] = pd.cut(df_corr['factor'], bins=threshold, labels=label)
                factor_group = df_corr.groupby('factor_group')['value'].mean()
            ax.scatter(factor_group.index, factor_group.values, s=4)
            ax.plot(np.linspace(-3,3,100), beta.mean()*np.linspace(-3,3,100) + gamma.mean(), c='C3')
            plt.title('r = %.2lf beta(万) = %.2lf gamma(万) = %.2lf'%(r.mean(), beta.mean()*10000, gamma.mean()*10000))
        else:
            ax.scatter(df_corr.loc[date]['factor'], df_corr.loc[date]['value'])
            ax.plot(np.linspace(-3,3,100), beta.loc[date]*np.linspace(-3,3,100) + gamma.loc[date], c='C3')
            plt.title('r = %.2lf beta(万) = %.2lf gamma(万) = %.2lf'%(r.loc[date], beta.loc[date]*10000, gamma.loc[date]*10000))
        plt.show()
    # 因子自相关系数组合
    def autocorr(self):
        self.corr_dic = {}
        for period in self.periods:
            # 初始位置
            factor_original = self.factor.copy()
            factor_original.name = 'original'
            factor_latter = self.factor.groupby('code').shift(period).copy()
            factor_latter.name = 'latter'
            self.corr_dic[period] = pd.concat([factor_original, factor_latter], axis=1).groupby('date').corr(method='pearson').loc[:, 'original', :]['latter'].mean()



'''
因子中性化函数
market: index格式为date,code
col1: str 被中心化列名
col2: str 中心化参考列名
group_num: int 分组数量;
返回: seires, category
'''
def Factor_Neutralization(market, col1, col2, col1_group_num=5, col2_group_num=10):
    def fun(x):
        label1 = [i for i in range(col1_group_num)]
        label2 = [i for i in range(col2_group_num)]
        s = x.groupby(pd.qcut(x[col2], col2_group_num, labels=label2))\
            .apply(lambda x: pd.qcut(x[col1], col1_group_num, labels=label1)).droplevel(0)
        return s

    df = market.copy(deep=True)
    s = df.groupby(level='date').apply(lambda x: fun(x.droplevel(0)))
    s.name = col1 + '_de' + col2
    return s
    











