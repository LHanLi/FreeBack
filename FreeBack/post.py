import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import statsmodels.api as sm
from plottable import ColumnDefinition, ColDef, Table
from matplotlib.colors import LinearSegmentedColormap
from FreeBack.display import *
import FreeBack as FB
import os
# 该模块处理策略的后处理(业绩评价、归因）工作，主要包含：
# 1. ReturnsPost 收益率序列后处理
# 2. HoldPost 持仓矩阵后处理
# 3. WorldPost barbybar模块World对象后处理

# 结果文件保存于output中
def check_output():
    if 'output' in os.listdir():
        pass
    else:
        os.mkdir('output')

############################################################################################
####################### 处理收益率序列（简单收益率，非对数收益率） ###########################
############################################################################################
class ReturnsPost():
    # benchmark dataframe 收益率序列
    def __init__(self, returns, benchmark=0, stratname='策略', rf=0.03):
        self.stratname = stratname
        self.returns = returns.fillna(0)
        # 无风险利率
        self.rf = rf
        # 基准指数
        if type(benchmark)==int:
            benchmark = pd.DataFrame(index = self.returns.index)
            benchmark['zero'] = 0
            self.benchmark = benchmark
        self.benchmark = benchmark.loc[self.returns.index].fillna(0)
        self.sigma_benchmark = np.exp(np.log(self.benchmark[\
            self.benchmark.columns[0]]+1).std())-1
        self.cal_detail()
    # 详细评价表
    def cal_detail(self):
        # 策略绝对表现
        self.net = (1+self.returns).cumprod()
        self.lr = np.log(self.returns + 1)
        #self.returns.index.name = 'date'
        self.years = (self.returns.index[-1]-self.returns.index[0]).days/365  
        self.return_total = self.net[-1]/self.net[0]-1                    
        self.return_annual = (self.return_total+1)**(1/self.years)-1   
        self.sigma = np.exp(self.lr.std())-1 
        self.sharpe = (self.return_annual - self.rf)/(self.sigma*np.sqrt(250))
        a = np.maximum.accumulate(self.net)
        self.drawdown = (a-self.net)/a
        # 超额表现
        self.excess_lr = self.lr-np.log(self.benchmark[self.benchmark.columns[0]]+1) 
        self.excess_net = np.exp(self.excess_lr.cumsum())
        self.excess_total = self.excess_net[-1]/self.excess_net[0]                 
        self.excess_return_annual = self.excess_total**(1/self.years)-1
        self.excess_sigma = np.exp(self.excess_lr.std())-1
        self.excess_sharpe = self.excess_return_annual/(self.excess_sigma*np.sqrt(250))
        a = np.maximum.accumulate(self.excess_net)
        self.excess_drawdown = (a-self.excess_net)/a
        # CAPM (无风险收益为0)
        y = self.returns.fillna(0)
        x = self.benchmark[self.benchmark.columns[0]].fillna(0)
        x = sm.add_constant(x)
        model = sm.OLS(y, x).fit()
        # 市场风险暴露
        self.beta = model.params[self.benchmark.columns[0]]
        # 市场波动无法解释的截距项
        self.alpha = model.params['const']
        #model.summary()

        col0 = pd.DataFrame(columns=['col0'])
        col0.loc[0] = '回测时间（年, 日）'
        col0.loc[1] = '%s, %s'%(round(self.years,1), len(self.net))
        col1 = pd.DataFrame(columns=['col1'])
        col1.loc[0] = '年化收益率（%）'
        col1.loc[1] = round(self.return_annual*100,1)
        col1.loc[2] = '年化超额收益率（%）'
        col1.loc[3] = round(self.excess_return_annual*100,1)
        col2 = pd.DataFrame(columns=['col2'])
        col2.loc[0] = '日胜率（%）'
        col2.loc[1] = round(100*(self.returns>0).mean(),1) 
        col2.loc[2] = '超额日胜率（%）'
        col2.loc[3] = round(100*(self.excess_lr>0).mean(),1) 
        col3 = pd.DataFrame(columns=['col3'])
        col3.loc[0] = '最大回撤（%）'
        col3.loc[1] = round(max(self.drawdown)*100, 1)
        col3.loc[2] = '超额最大回撤（%）'
        col3.loc[3] = round(max(self.excess_drawdown)*100, 1)
        col3.loc[4] = '波动率（%）'
        col3.loc[5] = round(self.sigma*np.sqrt(250)*100, 1)
        col4 = pd.DataFrame(columns=['col4'])
        col4.loc[0] = 'beta系数'
        col4.loc[1] = round(self.beta,2) 
        col4.loc[2] = '詹森指数（%）'
        col4.loc[3] = round(self.alpha*250*100,1)
        col5 = pd.DataFrame(columns=['col5'])
        col5.loc[0] = '夏普比率'
        col5.loc[1] = round(self.sharpe,2)
        col5.loc[2] = '超额夏普' 
        col5.loc[3] = round(self.excess_sharpe,2)
        col5.loc[4] = '卡玛比率'
        col5.loc[5] = round(self.return_annual/max(self.drawdown),2)
        col6 = pd.DataFrame(columns=['col6'])
        col6.loc[0] = ''
        col6.loc[1] = '' 
        col7 = pd.DataFrame(columns=['col7'])
        col7.loc[0] = 'Hurst指数' 
        col7.loc[1] = '' 
        df_details = pd.concat([col0, col1, col2, col3, \
                col4, col5, col6, col7], axis=1).fillna('')
        self.df_details = df_details
    def detail(self):
        plt, fig, ax = matplot(w=22)
        column_definitions = [ColumnDefinition(name='col0', group="基本参数"), \
                              ColumnDefinition(name='col1', group="收益能力"), \
                            ColumnDefinition(name='col2', group='收益能力'), \
                            ColumnDefinition(name='col3', group='风险水平'), \
                            ColumnDefinition(name="col4", group='风险调整'), \
                            ColumnDefinition(name="col5", group='风险调整'), \
                            ColumnDefinition(name="col6", group='策略执行'),
                            ColumnDefinition(name="col7", group='业绩持续性分析')] +\
                            [ColDef("index", title="", width=1.5, textprops={"ha":"right"})]
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
        check_output()
        plt.savefig('./output/details.png')
        plt.show()
# 时间起止（默认全部），是否显示细节,是否自定义输出图片名称，是否显示对数，是否显示超额
    def pnl(self, timerange=None, detail=True, filename=None, log=False, excess=False):
        plt, fig, ax = FB.display.matplot()
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
            # 如果基准是0就不绘制了
            if not (self.benchmark==0).all().values[0]:
                # benchmark 匹配回测时间段, 基准从0开始
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
            ax.set_xlim(returns.index[0], returns.index[-1])
            plt.gcf().autofmt_xdate()
        else:
    #评价指标
            ax.text(0.7,0.05,'年化收益率: {}%\n夏普比率:   {}\n最大回撤:   {}%\n'.format(
            round(100*self.return_annual,2), round(self.sharpe,2), 
            round(100*max(self.drawdown),2)), transform=ax.transAxes)
        # 净值与基准
            ax.plot(self.net, c='C0', label=self.stratname)
            # 如果基准是0就不绘制了
            if not (self.benchmark==0).all().values[0]:
                # benchmark 匹配回测时间段, 基准从0开始
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
        check_output()
        if type(filename) == type(None):
            plt.savefig('./output/pnl.png')
        else:
            plt.savefig('./output/'+filename)
        plt.show()
# 滚动收益与夏普
    def rolling_return(self, key='return'):
        plt, fig, ax = FB.display.matplot()
        if key=='return':
            ax.plot((self.net/self.net.shift(120)-1)*100, c='C0', label='滚动半年收益')
            ax2 = ax.twinx()
            ax2.plot((self.net/self.net.shift(250)-1)*100, c='C3', label='滚动年度收益')
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax.set_ylabel('(%)')
            ax2.set_ylabel('(%)')
        elif key=='sharpe':
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
        check_output()
        plt.savefig('./output/rolling.png')
        plt.show()
# 年度与月度收益
    def pnl_yearly(self):
        lr = self.lr
        lr.name = 'lr'
        bench = np.log(self.benchmark[self.benchmark.columns[0]].fillna(0)+1)
        bench.name = 'bench'
        year = pd.Series(dict(zip(self.returns.index, self.returns.index.map(lambda x: x.year))))
        year.name = 'year'
        yearly_returns = pd.concat([year, lr, bench], axis=1)
        yearly_returns = (np.exp(yearly_returns.groupby('year').sum())*100-100)

        plt, fig, ax = FB.display.matplot()

        len_years = len(yearly_returns)
        plot_x = range(len_years)
        plot_index = yearly_returns.index
        plot_height = yearly_returns['lr'].values
        plot_height1 = yearly_returns['bench'].values
        max_height = max(np.hstack([plot_height, plot_height1]))
        min_height = min(np.hstack([plot_height, plot_height1]))
        height = max_height-min_height
        # 如果benchmark是0的话就不画对比了
        if not (self.benchmark==0).any().values[0]:
            ax.bar([i-0.225  for i in plot_x], plot_height, width=0.45, color='C0', label='策略')
            ax.bar([i+0.225  for i in plot_x], plot_height1, width=0.45, color='C4',\
                    label=self.benchmark.columns[0])
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
        else:
            ax.bar([i  for i in plot_x], plot_height, width=0.45, color='C0', label='策略')
            for x, contri in zip(plot_x, plot_height):
                if contri>0:
                    plt.text(x, contri+height/30, round(contri,1), ha='center', color='C3', fontsize=8)
                else:
                    plt.text(x, contri-height/20, round(contri,1), ha='center', color='C2', fontsize=8)
        plt.ylim(min_height-0.1*height, max_height+0.1*height)

        plt.legend()
        plt.title('年度收益')
        plt.xticks(plot_x, labels=plot_index)
        check_output()
        plt.ylabel('(%)')
        plt.savefig('./output/pnl_yearly.png')
        plt.show()
    def pnl_monthly(self, excess=False):
        if excess:
            df = self.excess_lr
        else:
            df = self.lr
        name = (lambda x: 0 if x==None else x)(df.name) 
        df = df.reset_index()
        # 筛出同月数据
        df['month'] = df['date'].apply(lambda x: x - datetime.timedelta(x.day-1))
        df = df[['month', name]]
        df = df.set_index('month')[name]
        # 月度收益 %
        period_return = (np.exp(df.groupby('month').sum()) - 1)*100
        plt, fig, ax = FB.display.month_thermal(period_return)
        check_output()
        plt.savefig('./output/pnl_monthly.png')
        plt.show()



############################################################################################
################################### 处理持仓表 ##############################################
############################################################################################
class HoldPost(ReturnsPost):
    # 持仓表、单边交易成本、market
    def __init__(self, df_hold, comm=0/1e4, market=None, \
                 benchmark=0, stratname='策略'):
        self.df_hold = df_hold
        # 等权持仓
        df_hold['weight'] = 1
        df_hold['weight'] = df_hold['weight']/df_hold['weight'].groupby('date').sum()
        # 初始状态全仓为现金,没有现金列则
        pos_df = df_hold['weight'].unstack('code').fillna(0)
        pos_df_shift = pos_df.shift().fillna(0).copy()
        pos_df_shift.loc[pos_df.index[0], 'deposit'] = 1
        pos_df_shift.fillna(0)
        # 去掉现金列的绝对值增减之和即为换手率
        self.turnover_ser = abs(pos_df-pos_df_shift).drop(columns=['deposit', ]).sum(axis=1)
        # 收益率
        returns = (df_hold.groupby('date')['next_returns'].mean()+1)*(1-self.turnover_ser*comm)-1
        super(HoldPost, self).__init__(returns, benchmark=benchmark, stratname=stratname)
        self.df_details.loc[0, 'col6'] = '年换手' 
        self.df_details.loc[1, 'col6'] = round(self.turnover_ser.mean()*250,1)
    def turnover(self):
        plt, fig, ax = FB.display.matplot()
        ax.plot(self.turnover_ser*250, alpha=0.2)
        ax.plot(self.turnover_ser.rolling(20).mean()*250, label='20日滚动换手')
        ax.plot(self.turnover_ser.rolling(250).mean()*250, label='250日滚动换手')
        ax.legend()
        check_output()
        plt.savefig('./output/turnover.png')
        plt.show()
    def get_contribution(self):
        real_returns = self.df_hold['next_returns']/self.df_hold.groupby('date')['next_returns'].count()
        self.contribution = ((real_returns+1).groupby('code').prod()-1).sort_values()


########################################################################################################
####################################  barbybar.world  ##################################################
########################################################################################################

# 传入barbybar运行完毕的world对象
class WorldPost():
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





