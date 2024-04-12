import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import statsmodels.api as sm
from plottable import ColumnDefinition, ColDef, Table
from matplotlib.colors import LinearSegmentedColormap
from FreeBack.display import *
import os
  

'''
# matplot绘图
def matplot(r=1, c=1, sharex=False, sharey=False, w=8, d=5):
    # don't use sns style
    sns.reset_orig()
    #plot
    #run configuration 
    plt.rcParams['font.size']=14
    plt.rcParams['font.family'] = 'KaiTi'
    #plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif']=['Microsoft YaHei']
    plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
    plt.rcParams['axes.linewidth']=1
    plt.rcParams['axes.grid']=True
    plt.rcParams['grid.linestyle']='--'
    plt.rcParams['grid.linewidth']=0.2
    plt.rcParams["savefig.transparent"]='True'
    plt.rcParams['lines.linewidth']=0.8
    plt.rcParams['lines.markersize'] = 1
    
    #保证图片完全展示
    plt.tight_layout()
        
    #subplot
    fig,ax = plt.subplots(r,c,sharex=sharex, sharey=sharey,figsize=(w,d))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace = None, wspace=0.5)
        
    plt.gcf().autofmt_xdate()

    return plt, fig, ax

# 月度数据热力图  
# period_value格式为 index month ‘2023-7-1’  value  0: ***
# color_threshold 为红绿色分界点
def month_thermal(period_value, color_threshold=0):   
    # 数值>0红色，<0为绿色。转化为颜色[R,G,B]
    def color_map(x, max_r):
        if x>0:
            return [1,1-x/max_r,1-x/max_r]
        elif x == 0:
            return [1,1,1]
        else:
            return [1+x/max_r,1,1+x/max_r]
    # i 年份序号（纵坐标）  j 月份序号（横坐标） calendar是值， plot是颜色
    def calendar_array(dates, data):
    #    i, j = zip(*[d.isocalendar()[1:] for d in dates])
    # 年份  月份 array
        i, j = zip(*[(d.year,d.month) for d in dates])
        i = np.array(i) - min(i)
        j = np.array(j) - 1
    # 总年份
        ni = max(i) + 1
    # 12个月
    # 值 矩阵
        calendar = np.nan * np.zeros((ni, 12))
    # 颜色值 矩阵 默认白色[1,1,1]
        plot =   np.ones((ni, 12, 3))
        calendar[i, j] = data
    # 绝对值最大为纯红（绿）
        max_r = np.abs(data).max().max()
        mat_color = [color_map(i-color_threshold,max_r) for i in data]
        plot[i,j] = np.array(mat_color)
        return i, j, plot, calendar

    # 纵坐标为年份，横坐标为月份, 填充值
    dates, data = list(period_value.index), period_value.iloc[:,0].values
    i, j, plot, calendar = calendar_array(dates, data)
    # 绘制热力图
    plt, fig, ax = matplot()
    ax.imshow(plot, aspect='auto')
    # 设置纵坐标 年份
    i = np.array(list(set(i)))
    i.sort()
    ax.set(yticks=i)
    # 年份
    years = list(set([i.year for i in period_value.index]))
    ax.set_yticklabels(years)
    # 设置横坐标 月份
    j = np.array(list(set(j)))
    j.sort()
    ax.set(xticks=j)
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
                                'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    # 关闭网格
    ax.grid(False)
    # 显示数值
    for i_ in i:
        for j_ in j:
            if not np.isnan(calendar[i_][j_]):
                ax.text(j_, i_, calendar[i_][j_].round(2), ha='center', va='center')
    return plt, fig, ax

# 月度收益热力图    输入对数收益率的series 
def plot_thermal(df_returns):
    # 先转化为对数收益率
    #df_lr = df_returns.apply(lambda x: np.log(x+1))
    df_lr = df_returns.reset_index()
    # 筛出同月数据
    df_lr['month'] = df_lr['date'].apply(lambda x: x - datetime.timedelta(x.day-1))
    df_lr = df_lr[['month', df_returns.name]]
    df_lr = df_lr.set_index('month')
    # 月度收益 %
    period_return = (np.exp(df_lr.groupby('month').sum()) - 1)*100
    # 收益率转化为颜色[R,G,B]
    def color_map(x,max_r):
        if x>0:
            return [1,1-x/max_r,1-x/max_r]
        elif x == 0:
            return [1,1,1]
        else:
            return [1+x/max_r,1,1+x/max_r]
    # i 年份序号（纵坐标）  j 月份序号（横坐标） calendar是值， plot是颜色
    def calendar_array(dates, data):
    #    i, j = zip(*[d.isocalendar()[1:] for d in dates])
    # 年份  月份 array
        i, j = zip(*[(d.year,d.month) for d in dates])
        i = np.array(i) - min(i)
        j = np.array(j) - 1
    # 总年份
        ni = max(i) + 1
    # 12个月
    # 收益率 矩阵
        calendar = np.nan * np.zeros((ni, 12))
    # 颜色值 矩阵 默认白色[1,1,1]
        plot =   np.ones((ni, 12, 3))
        calendar[i, j] = data
    # 正最大收益为纯红 负最大收益为纯绿
        max_r = np.abs(data).max().max()
        mat_color = [color_map(i,max_r) for i in data]
        plot[i,j] = np.array(mat_color)
        return i, j, plot, calendar

    # 纵坐标为年份，横坐标为月份, 填充值
    dates, data = list(period_return.index), period_return.iloc[:,0].values
    i, j, plot, calendar = calendar_array(dates, data)
    # 绘制热力图
    plt, fig, ax = matplot()
    ax.imshow(plot, aspect='auto')
    # 设置纵坐标 年份
    i = np.array(list(set(i)))
    i.sort()
    ax.set(yticks=i)
    # 年份
    years = list(set([i.year for i in df_returns.index]))
    ax.set_yticklabels(years)
    # 设置横坐标 月份
    j = np.array(list(set(j)))
    j.sort()
    ax.set(xticks=j)
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
                                'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    # 关闭网格
    ax.grid(False)
    # 显示数值
    for i_ in i:
        for j_ in j:
            if not np.isnan(calendar[i_][j_]):
                ax.text(j_, i_, calendar[i_][j_].round(2), ha='center', va='center')

    return plt, fig, ax
'''

# 仅处理收益率序列（简单收益率，非对数收益率）结果
class SeriesPost():
    def __init__(self, returns, benchmark=None, stratname='策略'):
        self.stratname = stratname
        self.returns = returns
        self.net = (1+returns).cumprod()
        # 无风险利率
        self.rf = 0.03
        # 基准
        if type(benchmark) == type(None):
            benchmark = pd.DataFrame(index = returns.index)
            benchmark['zero'] = 0
            self.benchmark = benchmark
        self.benchmark = benchmark.loc[returns.index].fillna(0)
        # 基准波动率
        self.sigma_benchmark = np.exp(np.log(self.benchmark[\
            self.benchmark.columns[0]]+1).std())-1
        # 计算复杂指标
        # 净值曲线
        self.lr = np.log(self.returns + 1)
        # 超额收益 默认第一个benchmark
        self.excess_lr = self.lr-np.log(self.benchmark[self.benchmark.columns[0]]+1)
        self.returns.index.name = 'date'
        
        # 评价指标
        # 年化收益率（+1）
        self.years = (returns.index[-1]-returns.index[0]).days/365  
        self.return_total = (returns+1).prod()                             
        self.return_annual = self.return_total**(1/self.years)-1    
        excess_total = np.exp(self.excess_lr.sum())                  # 超额收益
        self.excess_return_annual = excess_total**(1/self.years)-1
        # 年化波动率 shrpe
        self.sigma = np.exp(self.lr.std())-1
        self.sharpe = (self.return_annual - self.rf)/(self.sigma*np.sqrt(250))
        # 超额年化波动率 shrpe
        self.excess_sigma = np.exp(self.excess_lr.std())-1
        self.excess_sharpe = (self.excess_return_annual - self.rf)/(self.excess_sigma*np.sqrt(250))
        # 回撤
        a = np.maximum.accumulate(self.net)
        self.drawdown = (a-self.net)/a
    def cal_complex(self):
        # 复杂指标 
        win = self.lr[self.lr>0]
        loss = self.lr[self.lr<0]
        excesswin = self.excess_lr[self.excess_lr>0]
        excessloss = self.excess_lr[self.excess_lr<0]
        # 下行波动率
        self.sigma_down = np.exp((self.lr-\
                            self.lr.mean()).apply(lambda x: min(x,0)).std())-1
        # 跟踪误差
        self.sigma_alpha = np.exp(np.std(self.lr-\
                             np.log(self.benchmark[self.benchmark.columns[0]]+1)))-1
        # 超额回撤
        excess_net = np.exp(self.excess_lr.fillna(0).cumsum())
        a = np.maximum.accumulate(excess_net)
        self.excess_drawdown = (a-excess_net)/a
        # CAPM (无风险收益为0)
        y = self.returns.fillna(0)
        x = self.benchmark[self.benchmark.columns[0]].fillna(0)
        x = sm.add_constant(x)
        model = sm.OLS(y, x).fit()
        # T-M 模型(无风险收益为0)
        y = self.returns.fillna(0)
        x = self.benchmark[self.benchmark.columns[0]].fillna(0)
        x = sm.add_constant(x)
        model = sm.OLS(y, x).fit()
        # 市场风险暴露
        self.beta = model.params[self.benchmark.columns[0]]
        # 市场波动无法解释的截距项
        self.alpha = model.params['const'] 
        #model.summary()
        ## 索提诺比率   单位下行风险的超额收益
        #self.sortino = (self.return_annual - self.rf)/(self.sigma_down*np.sqrt(250))
        # 特雷诺指数  单位beta的超额收益
        self.treynor  = (self.return_annual - self.rf)/self.beta 
# 净值曲线
# 时间起止（默认全部），是否显示细节,是否自定义输出图片名称，是否显示对数，是否显示超额
    def pnl(self, timerange=None, detail=True, filename=None, log=False, excess=False):
        plt, fig, ax = matplot(w=10)
        # 只画一段时间内净值（用于展示局部信息,只列出sharpe）
        if type(timerange) != type(None):
            # 时间段内净值与基准
            net = self.net.loc[timerange[0]:timerange[1]]
            returns = self.returns.loc[timerange[0]:timerange[1]]
            # 计算夏普
            years = (pd.to_datetime(timerange[1])-pd.to_datetime(timerange[0])).days/365
            return_annual = (net[-1]/net[0])**(1/years)-1
            std_annual = returns.std()*np.sqrt(250)
            sharpe = (return_annual - self.rf)/std_annual
            if detail:
                # 回撤
                a = np.maximum.accumulate(net)
                drawdown = (a-net)/a 
                ax.text(0.7,0.05,'年化收益率: {}%\n夏普比率:   {}\n最大回撤:   {}%\n'.format(
                    round(100*return_annual,2), round(sharpe,2), 
                        round(100*max(drawdown),2)), transform=ax.transAxes)
                # 回撤
                ax2 = ax.twinx()
                ax2.fill_between(drawdown.index,-100*drawdown, 0, color='C1', alpha=0.1)
                if excess:
                    ax.plot(np.exp(self.excess_lr.loc[timerange[0]:timerange[1]].cumsum()), 
                            c='C3', label='超额收益')
            else:
                ax.text(0.7,0.05,'Sharpe:  {}'.format(round(sharpe,2)), transform=ax.transAxes)
            ax.plot(net/net[0], c='C0', label='p&l')
            if type(self.benchmark) != type(None):
                benchmark = self.benchmark.loc[timerange[0]:timerange[1]].copy()
                benchmark.iloc[0] = 0
        # colors of benchmark
                colors_list = ['C4','C5','C6','C7']
                for i in range(len(benchmark.columns)):
                    ax.plot((benchmark[benchmark.columns[i]]+1).cumprod(), 
                            c=colors_list[i], label=benchmark.columns[i])
            if log:
                # 对数坐标显示
                ax.set_yscale("log")
            ax.set_xlim(returns.index[0], returns.index[-1])
            plt.gcf().autofmt_xdate()
        else:
    #评价指标
            ax.text(0.7,0.05,'年化收益率: {}%\n夏普比率:   {}\n最大回撤:   {}%\n'.format(
            round(100*self.return_annual,2), round(self.sharpe,2), 
            round(100*max(self.drawdown),2)), transform=ax.transAxes)
        # 净值与基准
            ax.plot(self.net, c='C0', label=self.stratname)
            if not (self.benchmark==0).any().values[0]:
                # benchmark 匹配回测时间段, 基准从0开始
                # 如果基准是0就不绘制了
                benchmark = self.benchmark.loc[self.net.index[0]:self.net.index[-1]].copy()
                benchmark.loc[self.net.index[0]] = 0
                # colors of benchmark
                colors_list = ['C4','C5','C6','C7']
                for i in range(len(benchmark.columns)):
                    ax.plot((benchmark[benchmark.columns[i]]+1).cumprod(), \
                            c=colors_list[i], label=benchmark.columns[i])
                if excess:
                    ax.plot(np.exp(self.excess_lr.cumsum()), c='C3', label='超额收益')
                plt.legend(loc='upper left')
            if log:
                # 对数坐标显示
                ax.set_yscale("log")
            # 回撤
            ax2 = ax.twinx()
            ax2.fill_between(self.drawdown.index,-100*self.drawdown, 0, color='C1', alpha=0.1)
            ax.set_ylabel('累计净值')
            ax.set_xlabel('日期')
            ax2.set_ylabel('回撤 (%)')
            ax.set_xlim(self.net.index[0], self.net.index[-1])
            plt.gcf().autofmt_xdate()
        if 'output-post' in os.listdir():
            pass
        else:
            os.mkdir('output-post')
        if type(filename) == type(None):
            plt.savefig('./output-post/pnl.png')
        else:
            plt.savefig('./output-post/'+filename)
        plt.show()



# 传入barbybar运行完毕的world对象
class Post():
    # benchmark为收益率（非对数收益率）
    def __init__(self, world, benchmark=None, stratname='策略'):
        # world
        self.world = world
        # 策略名
        self.stratname = stratname
        # 无风险利率
        self.rf = 0.03
        # 基准
        if type(benchmark) == type(None):
            benchmark = pd.DataFrame(index = world.series_net.index)
            benchmark['zero'] = 0
            self.benchmark = benchmark
        self.benchmark = benchmark.loc[world.series_net.index].fillna(0)
        self.sigma_benchmark = np.exp(np.log(self.benchmark[\
            self.benchmark.columns[0]]+1).std())-1
        self.details()

# 策略详细评价指标
    def details(self):
        # 净值曲线
        self.abs_net = self.world.series_net
        self.net = self.world.series_net/self.world.series_net.iloc[0]
        self.returns = self.net/self.net.shift() - 1
        self.lr = np.log(self.returns + 1)
        # 超额收益 默认第一个benchmark
        self.excess_lr = self.lr  - np.log(self.benchmark[self.benchmark.columns[0]]+1)
        self.returns.index.name = 'date'
        # 评价指标
        # 年化收益率
        self.years = (self.net.index[-1]-self.net.index[0]).days/365
        self.return_total = self.net[-1]/self.net[0]
        self.return_annual = self.return_total**(1/self.years)-1
        excess_total = np.exp(self.excess_lr.sum())
        self.excess_return_annual = excess_total**(1/self.years)-1
        # 年化波动率 shrpe
        self.sigma = np.exp(self.lr.std())-1
        self.sharpe = (self.return_annual - self.rf)/(self.sigma*np.sqrt(250))
        # 超额年化波动率 shrpe
        self.excess_sigma = np.exp(self.excess_lr.std())-1
        self.excess_sharpe = (self.excess_return_annual - self.rf)/(self.excess_sigma*np.sqrt(250))
        # 回撤
        a = np.maximum.accumulate(self.net)
        self.drawdown = (a-self.net)/a
        # 持仓标的
        self.df_hold = self.world.df_hold
        held = self.world.df_hold.reset_index().melt(id_vars=['date']).set_index(['date', 'code'])
        held = held['value'] * self.world.market['close']
        self.held = held[(held != 0)&~np.isnan(held)]
        # 现金
        self.cash = self.world.series_cash
        # 交割单
        self.excute = self.world.df_excute.reset_index().set_index(['date', 'code', 'unique'])
        # 换手率
        occurance_amount = abs(self.excute['occurance_amount']).groupby('date').sum()
        #turnover = (occurance_amount/(self.net*self.cash.iloc[0])).fillna(0)
        turnover = (occurance_amount/self.world.series_net).fillna(0)
        self.turnover = turnover.mean()*250

        # 逐笔交易信息
        self.trade()

        # details 表格
        win = self.lr[self.lr>0]
        loss = self.lr[self.lr<0]
        excesswin = self.excess_lr[self.excess_lr>0]
        excessloss = self.excess_lr[self.excess_lr<0]
        # 下行波动率
        self.sigma_down = np.exp((self.lr-self.lr.mean()).apply(lambda x: min(x,0)).std())-1
        # 跟踪误差
        sigma_alpha = np.exp(np.std(self.lr-np.log(self.benchmark[self.benchmark.columns[0]]+1)))-1
        # 超额回撤
        excess_net = np.exp(self.excess_lr.fillna(0).cumsum())
        a = np.maximum.accumulate(excess_net)
        self.excess_drawdown = (a-excess_net)/a
        # CAPM (无风险收益为0)
        y = self.returns.fillna(0)
        x = self.benchmark[self.benchmark.columns[0]].fillna(0)
        x = sm.add_constant(x)
        model = sm.OLS(y, x).fit()
        # T-M 模型(无风险收益为0)
        y = self.returns.fillna(0)
        x = self.benchmark[self.benchmark.columns[0]].fillna(0)
        x = sm.add_constant(x)
        model = sm.OLS(y, x).fit()
        # 市场风险暴露
        self.beta = model.params[self.benchmark.columns[0]]
        # 市场波动无法解释的截距项
        self.alpha = model.params['const'] 
        #model.summary()
        ## 索提诺比率   单位下行风险的超额收益
        #self.sortino = (self.return_annual - self.rf)/(self.sigma_down*np.sqrt(250))
        # 特雷诺指数  单位beta的超额收益
        self.treynor  = (self.return_annual - self.rf)/self.beta

        col0 = pd.DataFrame(columns=['col0'])
        col0.loc[0] = '运行时间（年, 日）'
        col0.loc[1] = '%s, %s'%(round(self.years,1), len(self.net))
        col0.loc[2] = '盈利次数（日）'
        col0.loc[3] = len(win)
        col0.loc[4] = '超额次数（日）'
        col0.loc[5] = len(excesswin)
        col0.loc[6] = '空仓次数（日）' 
        col0.loc[7] = len(self.net)-len(win)-len(loss) 
        col1 = pd.DataFrame(columns=['col1'])
        col1.loc[0] = '年化收益率（%）'
        col1.loc[1] = round(self.return_annual*100,1)
        col1.loc[2] = '年化超额收益率（%）'
        col1.loc[3] = round(self.excess_return_annual*100,1)
        col1.loc[4] = '累计收益（%）'
        col1.loc[5] = round((self.return_total-1)*100,1)
        col1.loc[6] = '累计超额收益（%）'
        col1.loc[7] = round((excess_total-1)*100,1)
        col2 = pd.DataFrame(columns=['col2'])
        col2.loc[0] = '日胜率（%）'
        col2.loc[1] = round(100*len(win)/(len(win)+len(loss)),1) 
        col2.loc[2] = '超额日胜率（%）'
        col2.loc[3] = round(100*len(excesswin)/(len(excesswin)+\
                                len(excessloss)),1) 
        col2.loc[4] = '日盈亏比'
        col2.loc[5] = round(win.mean()/abs(loss.mean()),2) 
        col2.loc[6] = '亏损日平均亏损（万）'
        col2.loc[7] = round(abs(loss.mean())*10000,1) 
        col3 = pd.DataFrame(columns=['col3'])
        col3.loc[0] = '最大回撤（%）'
        col3.loc[1] = round(max(self.drawdown)*100, 1)
        col3.loc[2] = '超额最大回撤（%）'
        col3.loc[3] = round(max(self.excess_drawdown)*100, 1)
        col3.loc[4] = '波动率（%）'
        col3.loc[5] = round(self.sigma*np.sqrt(250)*100, 1)
        col3.loc[6] = '基准波动率（%）'
        col3.loc[7] = round(self.sigma_benchmark*np.sqrt(250)*100, 1)
        col4 = pd.DataFrame(columns=['col4'])
        col4.loc[0] = '下行波动率（%）'
        col4.loc[1] = round(self.sigma_down*np.sqrt(250)*100, 1)
        col4.loc[2] = 'beta系数'
        col4.loc[3] = round(self.beta,2) 
        col4.loc[4] = '跟踪误差（%）'
        col4.loc[5] = round(sigma_alpha*np.sqrt(250)*100, 1)
        col4.loc[6] = '年化换手' 
        col4.loc[7] = round(self.turnover,1)
        col5 = pd.DataFrame(columns=['col5'])
        col5.loc[0] = '夏普比率'
        col5.loc[1] = round(self.sharpe,2)
        col5.loc[2] = '超额夏普' 
        col5.loc[3] = round(self.excess_sharpe,2)
        col5.loc[4] = '卡玛比率'
        col5.loc[5] = round(self.return_annual/max(self.drawdown),2)
        col5.loc[6] = '无风险收益率（%）' 
        col5.loc[7] = self.rf*100 
        col6 = pd.DataFrame(columns=['col6'])
        col6.loc[0] = '詹森指数（alpha）'
        #col6.loc[0] = '詹森指数'
        col6.loc[1] = round(self.alpha*250*100,1) 
        col6.loc[2] = '特雷诺指数'  # 单位beta的超额收益 
        col6.loc[3] =  round(self.treynor, 2)
        col6.loc[4] = '信息比率'   # 超额收益的夏普
        col6.loc[5] = round(self.alpha*np.sqrt(250)/sigma_alpha,2) 
        col6.loc[6] = '' 
        col6.loc[7] = '' 
        col7 = pd.DataFrame(columns=['col7'])
        col7.loc[0] = 'Hurst指数' 
        col7.loc[1] = '' 
        col7.loc[2] = 'T-M模型' 
        col7.loc[3] = '' 
        col7.loc[4] = '' 
        col7.loc[5] = '' 
        col7.loc[6] = '' 
        col7.loc[7] = '' 
        df_details = pd.concat([col0, col1, col2, col3, \
                col4, col5, col6, col7], axis=1)

        plt, fig, ax = matplot(w=22)
        column_definitions = [ColumnDefinition(name='col0', group="收益能力"), \
                              ColumnDefinition(name='col1', group="收益能力"), \
                            ColumnDefinition(name='col2', group='收益能力'), \
                            ColumnDefinition(name='col3', group='风险水平'), \
                            ColumnDefinition(name="col4", group='风险水平'), \
                            ColumnDefinition(name="col5", group='风险调整'), \
                            ColumnDefinition(name="col6", group='风险调整'),
                            ColumnDefinition(name="col7", group='业绩持续性分析')] +\
                            [ColDef("index", title="", width=1.5, textprops={"ha":"right"})]
        tab = Table(df_details, row_dividers=False, col_label_divider=False, 
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
        plt.savefig('details.png')
        plt.show()
# 净值曲线
    def pnl(self, timerange=None, detail=False, filename=None, log=False, excess=True):
        plt, fig, ax = matplot(w=10)
        # 只画一段时间内净值（用于展示局部信息,只列出sharpe）
        if type(timerange) != type(None):
            # 时间段内净值与基准
            net = self.net.loc[timerange[0]:timerange[1]]
            returns = self.returns.loc[timerange[0]:timerange[1]]
            # 计算夏普
            years = (pd.to_datetime(timerange[1])-pd.to_datetime(timerange[0])).days/365
            return_annual = (net[-1]/net[0])**(1/years)-1
            std_annual = returns.std()*np.sqrt(250)
            sharpe = (return_annual - self.rf)/std_annual
            if detail:
                # 回撤
                a = np.maximum.accumulate(net)
                drawdown = (a-net)/a 
                ax.text(0.7,0.05,'年化收益率: {}%\n夏普比率:   {}\n最大回撤:   {}%\n'.format(
                    round(100*return_annual,2), round(sharpe,2), 
                        round(100*max(drawdown),2)), transform=ax.transAxes)
                # 回撤
                ax2 = ax.twinx()
                ax2.fill_between(drawdown.index,-100*drawdown, 0, color='C1', alpha=0.1)
                if excess:
                    ax.plot(np.exp(self.excess_lr.loc[timerange[0]:timerange[1]].cumsum()), 
                            c='C3', label='超额收益')
            else:
                ax.text(0.7,0.05,'Sharpe:  {}'.format(round(sharpe,2)), transform=ax.transAxes)
            ax.plot(net/net[0], c='C0', label='p&l')
            if type(self.benchmark) != type(None):
                benchmark = self.benchmark.loc[timerange[0]:timerange[1]].copy()
                benchmark.iloc[0] = 0
        # colors of benchmark
                colors_list = ['C4','C5','C6','C7']
                for i in range(len(benchmark.columns)):
                    ax.plot((benchmark[benchmark.columns[i]]+1).cumprod(), 
                            c=colors_list[i], label=benchmark.columns[i])
            if log:
                # 对数坐标显示
                ax.set_yscale("log")
            ax.set_xlim(returns.index[0], returns.index[-1])
            plt.gcf().autofmt_xdate()
        else: 
    #评价指标
            ax.text(0.7,0.05,'年化收益率: {}%\n夏普比率:   {}\n最大回撤:   {}%\n'.format(
            round(100*self.return_annual,2), round(self.sharpe,2), 
            round(100*max(self.drawdown),2)), transform=ax.transAxes)
        # 净值与基准
            ax.plot(self.net, c='C0', label=self.stratname)
            if type(self.benchmark) != type(None):
                # benchmark 匹配回测时间段, 基准从0开始
                benchmark = self.benchmark.loc[self.net.index[0]:self.net.index[-1]].copy()
                benchmark.loc[self.net.index[0]] = 0
                # colors of benchmark
                colors_list = ['C4','C5','C6','C7']
                for i in range(len(benchmark.columns)):
                    ax.plot((benchmark[benchmark.columns[i]]+1).cumprod(),  c=colors_list[i], label=benchmark.columns[i])
                if excess:
                    ax.plot(np.exp(self.excess_lr.cumsum()), c='C3', label='超额收益')
                plt.legend(loc='upper left')
            if log:
                # 对数坐标显示
                ax.set_yscale("log")
            # 回撤
            ax2 = ax.twinx()
            ax2.fill_between(self.drawdown.index,-100*self.drawdown, 0, color='C1', alpha=0.1)
            ax.set_ylabel('累计净值')
            ax.set_xlabel('日期')
            ax2.set_ylabel('回撤 (%)')
            ax.set_xlim(self.net.index[0], self.net.index[-1])
            plt.gcf().autofmt_xdate()
        if type(filename) == type(None):
            plt.savefig('pnl.png')
        else:
            plt.savefig(filename)
        plt.show()
# 滚动收益
    def rolling_return(self):
        plt, fig, ax = matplot()
        ax.plot((self.net/self.net.shift(120)-1)*100, c='C0', label='滚动半年收益')
        #ax.plot((post0.net/post0.net.shift(60)-1)*4, c='C1', label='quarter return')
        #ax.plot((post0.net/post0.net.shift(120)-1)*2, c='C3', label='half year return')
        ax2 = ax.twinx()
        ax2.plot((self.net/self.net.shift(250)-1)*100, c='C3', label='滚动年度收益')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.set_ylabel(' (%)')
        ax2.set_ylabel('(%)')
        ax.set_xlim(self.returns.index[0], self.returns.index[-1])
        plt.gcf().autofmt_xdate()
        plt.savefig('rolling_return.png')
        plt.show()
# 滚动sharpe
    def rolling_sharpe(self):
        plt, fig, ax = matplot()
        halfyearly_sharpe = (np.exp(self.lr.rolling(120).mean()*250)-1)/\
            ((np.exp(self.lr.rolling(120).std())-1)*np.sqrt(250))
        yearly_sharpe = (np.exp(self.lr.rolling(250).mean()*250)-1)/\
            ((np.exp(self.lr.rolling(250).std())-1)*np.sqrt(250))
        ax.plot(halfyearly_sharpe, c='C0', label='滚动半年sharpe')
        ax2 = ax.twinx()
        ax2.plot(yearly_sharpe, c='C3', label='滚动年度sharpe')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.set_ylim(-3, 10)
        ax.set_xlim(self.returns.index[0], self.returns.index[-1])
        plt.gcf().autofmt_xdate()
        plt.savefig('rolling_sharpe.png')
# 月度收益
    def pnl_monthly(self):
        plt,fig,ax = plot_thermal(self.lr)
        plt.savefig('pnl_monthly.png')
        plt.show()
    def pnl_yearly(self):
        lr = self.lr
        lr.name = 'lr'
        bench = np.log(self.benchmark[self.benchmark.columns[0]].fillna(0)+1)
        bench.name = 'bench'
        year = pd.Series(dict(zip(self.returns.index, self.returns.index.map(lambda x: x.year))))
        year.name = 'year'
        yearly_returns = pd.concat([year, lr, bench], axis=1)
        yearly_returns = (np.exp(yearly_returns.groupby('year').sum())*100-100)

        plt, fig, ax = matplot()

        len_years = len(yearly_returns)
        plot_x = range(len_years)
        plot_index = yearly_returns.index
        plot_height = yearly_returns['lr'].values
        plot_height1 = yearly_returns['bench'].values

        ax.bar([i-0.225  for i in plot_x], plot_height, width=0.45, color='C0', label='策略')
        ax.bar([i+0.225  for i in plot_x], plot_height1, width=0.45, color='C4', label=self.benchmark.columns[0])

        max_height = max(np.hstack([plot_height, plot_height1]))
        min_height = min(np.hstack([plot_height, plot_height1]))
        height = max_height-min_height
        plt.ylim(min_height-0.1*height, max_height+0.1*height)

        for x, contri in zip(plot_x, plot_height):
            if contri>0:
                plt.text(x-0.225, contri+height/30, round(contri,1), ha='center', color='C3', fontsize=8)
            else:
                plt.text(x-0.225, contri-height/20, round(contri,1), ha='center', color='C2', fontsize=8)
        for x, contri in zip(plot_x, plot_height1):
            if contri>0:
                plt.text(x+0.225, contri+height/30, round(contri,1), ha='center', color='C3', fontsize=8)
            else:
                plt.text(x+0.225, contri-height/20, round(contri,1), ha='center', color='C2', fontsize=8)
        plt.legend()
        plt.title('年度收益')
        plt.xticks(plot_x, labels=plot_index)
        plt.ylabel('(%)')
        plt.savefig('pnl_yearly.png')
        plt.show()
# 月度超额收益
    def pnl_excess_monthly(self):
        plt,fig,ax = plot_thermal(self.excess_lr)
        plt.savefig('pnl_excess_monthly.png')
        plt.show()
# 交易分析
    def trade(self):
        self.CashFlow = self.excute.apply(lambda x: x['occurance_amount'] - x['comm'] if x['BuyOrSell']=='Sell' else -x['occurance_amount'] - x['comm'], axis=1)
        # 获取持仓卖出变为0时的订单unique，记为一次交易结束(有可能会出现同时两个卖出订单误判为两次交易)
        completes = list(self.excute[(self.excute['remain_vol'] == 0)&(self.excute['BuyOrSell']=='Sell')].reset_index()['unique'])
        # 对每个标的的累积现金流
        CumCashFlow = self.CashFlow.groupby('code').cumsum()
        # 截取交易结束节点
        Close_amount = CumCashFlow.reset_index().set_index('unique').loc[completes]
        Close_amount = Close_amount.reset_index().set_index(['date', 'code', 'unique'])[0]
        # 每次交易盈亏（当前交易结束的现金流-上一次交易结束的现金流 ）
        self.Close_amount = Close_amount - Close_amount.groupby('code').shift().fillna(0)
        trade_pnl = self.Close_amount.reset_index()[['date', 'code', 0]].set_index([\
                                'date', 'code'])[0]
        self.trade_pnl = trade_pnl.sort_values()/self.abs_net
        # 筛出同月数据
        trades = self.Close_amount.reset_index()
        trades['month'] = trades['date'].apply(lambda x: x - datetime.timedelta(x.day-1)) 
        trades = trades[['month', 0]]
        self.trades = trades.set_index('month')

        # 总体数据
        self.trades_win = self.trades > 0
        self.trades_loss = self.trades < 0
        self.winrate = self.trades_win.sum().values[0]/len(self.trades)
        self.odds = -(self.trades[self.trades_win[0]].mean()/self.trades[self.trades_loss[0]].mean()).values[0]
    def trade_plot(self):
        # 交易次数
        Close_count = self.Close_amount.reset_index()[['date',0]].set_index('date')
        Close_count = Close_count.groupby('date').count().cumsum()
        # 画图
        plt, fig, ax = matplot()
        ax.plot(Close_count, label='累积交易次数 胜率：%s 盈亏比：%s'%(round(self.winrate, 2), round(self.odds, 2)))
        ax.set_xlim(self.net.index[0], self.net.index[-1])
        ax.legend()
        plt.gcf().autofmt_xdate()
        plt.savefig('trade.png')
        plt.show()
    def trade_monthly(self):
        # 月度盈利（亏损）交易的次数与平均盈利（亏损）。
        count_win = self.trades_win.groupby('month').sum()
        count_loss = self.trades_loss.groupby('month').sum()
#        mean_win = trades[trades_win[0]].groupby('month').mean()
#        mean_loss = trades[trades_loss[0]].groupby('month').mean()
        # 月度胜率和盈亏比
        self.month_winrate = count_win/(count_win + count_loss)
#        self.month_odds = -mean_win/mean_loss
        plt,fig,ax = month_thermal(self.month_winrate, 0.5)
        plt.savefig('trade_monthly.png') 
        plt.show()
# 仓位分析
    def position(self):
        plt, fig, ax = matplot()
        held_amount = self.held.groupby('date').sum()
        # 0是资产 1是现金
        df_total = pd.concat([self.held.groupby('date').sum(), self.cash], axis=1).fillna(0)
        df_max = pd.concat([self.held.groupby('date').max(), self.cash], axis=1).fillna(0)
        df_count = pd.concat([self.held.groupby('date').count(), self.cash], axis=1).fillna(0)
        ax.plot(df_total[0]/df_total.sum(axis=1), label='非现金仓位')
        ax.set_xlim(self.held.index[0][0], self.held.index[-1][0])
        ax.legend(loc = 'upper left')
        ax2 = ax.twinx()
        #ax2.plot(self.held.groupby('date').count(), c="C2", label='持有标的数量')
        # 如果不是单品种策略
        if self.world.market.index[0][1] != 'onlyone':
            ax.plot(df_max[0]/df_total.sum(axis=1), label='第一大持仓仓位')
            ax2.plot(df_count[0], c="C2", label='持有标的数量')
            ax2.legend(loc = 'upper right')
        plt.gcf().autofmt_xdate()
        plt.savefig('position.png')
        plt.show()





