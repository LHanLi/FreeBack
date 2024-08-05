import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import statsmodels.api as sm
from statsmodels.sandbox.stats.runs import runstest_1samp
from plottable import ColumnDefinition, ColDef, Table
from matplotlib.colors import LinearSegmentedColormap
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

##########################################################################################
####################### 处理收益率序列（简单收益率，非对数收益率） ###########################
##########################################################################################
class ReturnsPost():
    # returns,简单收益率序列  type:pd.Series index:pd.DatetimeIndex 
    # benchmark,基准收益率序列（可以多个） pd.DataFrame index:pd.DatetimeIndex, 0表示不设基准 
    def __init__(self, returns, benchmark=0, stratname='策略', freq='day', rf=0.03, fast=False):
        self.stratname = stratname
        self.returns = returns.fillna(0)
        # returns频率， 目前支持day, week
        if freq not in ['day', 'week']:
            print('输入频率错误')
            return
        else:
            self.freq = freq
        # 无风险利率
        self.rf = rf
        if fast:
            # 策略绝对表现
            self.bars = len(self.returns)  
            self.net = (1+self.returns).cumprod()
            self.lr = np.log(self.returns + 1)
            self.return_total = self.net.iloc[-1]-1                    
            self.years = (self.returns.index[-1]-self.returns.index[0]).days/365  
            self.return_annual = (self.return_total+1)**(1/self.years)-1   
            self.sigma = np.exp(self.lr.std())-1
            if self.freq == 'day':
                self.sharpe = (self.return_annual - self.rf)/(self.sigma*np.sqrt(250))
            elif self.freq == 'week':
                self.sharpe = (self.return_annual - self.rf)/(self.sigma*np.sqrt(48))
            a = np.maximum.accumulate(self.net)
            self.drawdown = (a-self.net)/a
            # 基准指数
            if type(benchmark)==type(0):
                benchmark = pd.DataFrame(index = self.returns.index)
                benchmark['zero'] = 0
                self.benchmark = benchmark
            self.benchmark = benchmark.loc[self.returns.index].fillna(0)
        else: 
            # 基准指数
            if type(benchmark)==type(0):
                benchmark = pd.DataFrame(index = self.returns.index)
                benchmark['zero'] = 0
                self.benchmark = benchmark
            self.benchmark = benchmark.loc[self.returns.index].fillna(0)
            self.sigma_benchmark = np.exp(np.log(self.benchmark[\
                self.benchmark.columns[0]]+1).std())-1
            self.cal_detail()
            self.detail()
    # 详细评价表
    def cal_detail(self):
        # 策略绝对表现
        self.net = (1+self.returns).cumprod()
        self.lr = np.log(self.returns + 1)
        self.bars = len(self.returns)  
        self.years = (self.returns.index[-1]-self.returns.index[0]).days/365  
        self.return_total = self.net.iloc[-1]-1                    
        self.return_annual = (self.return_total+1)**(1/self.years)-1   
        self.sigma = np.exp(self.lr.std())-1
        # 一年多少个bar
        if self.freq == 'day':
            self.anunal_num = 250
        elif self.freq == 'week':
            self.anunal_num = 48
        self.sharpe = (self.return_annual - self.rf)/(self.sigma*np.sqrt(self.anunal_num))
        a = np.maximum.accumulate(self.net)
        self.drawdown = (a-self.net)/a
        # 超额表现
        self.excess_lr = self.lr-np.log(self.benchmark[self.benchmark.columns[0]]+1)
        self.excess_net = np.exp(self.excess_lr.cumsum())
        self.excess_total = self.excess_net.iloc[-1]/self.excess_net.iloc[0]
        self.excess_return_annual = self.excess_total**(1/self.years)-1
        self.excess_sigma = np.exp(self.excess_lr.std())-1
        self.excess_sharpe = self.excess_return_annual/(self.excess_sigma*np.sqrt(self.anunal_num))
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
        if self.freq == 'day':
            col0.loc[0] = '回测时间（年, 日）'
        elif self.freq == 'week':
            col0.loc[0] = '回测时间（年, 周）'
        col0.loc[1] = '%s, %s'%(round(self.years,1), len(self.net))
        col1 = pd.DataFrame(columns=['col1'])
        col1.loc[0] = '年化收益率（%）'
        col1.loc[1] = round(self.return_annual*100,1)
        col1.loc[2] = '年化超额收益率（%）'
        col1.loc[3] = round(self.excess_return_annual*100,1)
        col2 = pd.DataFrame(columns=['col2'])
        col2.loc[0] = '日胜率（%）'  # 没亏就是赢
        col2.loc[1] = round(100*(self.returns>=0).mean(),1)
        col2.loc[2] = '超额日胜率（%）'
        col2.loc[3] = round(100*(self.excess_lr>0).mean(),1)
        col3 = pd.DataFrame(columns=['col3'])
        col3.loc[0] = '最大回撤（%）'
        col3.loc[1] = round(max(self.drawdown)*100, 1)
        col3.loc[2] = '超额最大回撤（%）'
        col3.loc[3] = round(max(self.excess_drawdown)*100, 1)
        col3.loc[4] = '波动率（%）'
        col3.loc[5] = round(self.sigma*np.sqrt(self.anunal_num)*100, 1)
        col4 = pd.DataFrame(columns=['col4'])
        col4.loc[0] = 'beta系数'
        col4.loc[1] = round(self.beta,2)
        col4.loc[2] = 'alpha（%）'
        col4.loc[3] = round(self.alpha*self.anunal_num*100,1)
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
        col7.loc[0] = '游程检验（%）'   # 拒绝随机假设的概率
        col7.loc[1] = round(100*runstest_1samp(self.returns>0)[1],2)
        df_details = pd.concat([col0, col1, col2, col3, \
                col4, col5, col6, col7], axis=1).fillna('')
        self.df_details = df_details
    def detail(self):
        plt, fig, ax = FB.display.matplot(w=22)
        column_definitions = [ColumnDefinition(name='col0', group="基本参数"), \
                              ColumnDefinition(name='col1', group="收益能力"), \
                            ColumnDefinition(name='col2', group='收益能力'), \
                            ColumnDefinition(name='col3', group='风险水平'), \
                            ColumnDefinition(name="col4", group='风险调整'), \
                            ColumnDefinition(name="col5", group='风险调整'), \
                            ColumnDefinition(name="col6", group='策略执行'),
                            ColumnDefinition(name="col7", group='业绩持续性分析')] + \
                             [ColDef("index", title="", width=0, textprops={"ha":"right"})]
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
            ax.plot(self.net, c='C0', linewidth=2, label=self.stratname)
            # 如果基准是0就不绘制了
            if not (self.benchmark==0).all().values[0]:
                # benchmark 匹配回测时间段, 基准从0开始
                benchmark = self.benchmark.loc[self.net.index[0]:self.net.index[-1]].copy()
                #benchmark.loc[self.net.index[0]] = 0
                # colors of benchmark
                colors_list = ['C4','C5','C6','C7', 'C8', 'C9']*10
                for i in range(len(benchmark.columns)):
                    ax.plot((benchmark[benchmark.columns[i]]+1).cumprod(), alpha=0.8,\
                            c=colors_list[i], label=benchmark.columns[i])
                if excess:
                    ax.plot(np.exp(self.excess_lr.cumsum()), c='C3', linewidth=1.5, label='超额收益')
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
            ax.plot((self.net/self.net.shift(self.anunal_num//2)-1)*100, c='C0', label='滚动半年收益')
            ax2 = ax.twinx()
            ax2.plot((self.net/self.net.shift(self.anunal_num)-1)*100, c='C3', label='滚动年度收益')
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax.set_ylabel('(%)')
            ax2.set_ylabel('(%)')
        elif key=='sharpe':
            halfyearly_sharpe = (np.exp(self.lr.rolling(self.anunal_num//2).mean()*self.anunal_num)-1)/\
            ((np.exp(self.lr.rolling(self.anunal_num//2).std())-1)*np.sqrt(self.anunal_num))
            yearly_sharpe = (np.exp(self.lr.rolling(self.anunal_num).mean()*self.anunal_num)-1)/\
            ((np.exp(self.lr.rolling(self.anunal_num).std())-1)*np.sqrt(self.anunal_num))
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
        df['month'] = df['date'].apply(lambda x: x - datetime.timedelta(days=x.day-1,\
                                    hours=x.hour, minutes=x.minute, \
                                        seconds=x.second, microseconds=x.microsecond))
        df = df[['month', name]]
        df = df.set_index('month')[name]
        # 月度收益 %
        period_return = (np.exp(df.groupby('month').sum()) - 1)*100
        plt, fig, ax = FB.display.month_thermal(period_return)
        check_output()
        plt.savefig('./output/pnl_monthly.png')
        plt.show()



######################################################################################################
#################################### 输入一系列收益率序列进行分析 ######################################
#####################################################################################################
class BatchPost():
    def __init___(self, batch):
        pass




##########################################################################################
####################### 分析基金收益率 ###########################
##########################################################################################
class FundPost(ReturnsPost):
    def __init__(self, returns, benchmark=None, fundname='基金名称', rf = 0.03, fast=False):
        super().__init__(returns, benchmark=benchmark, stratname=fundname, freq='week', rf=rf, fast=fast)

    # 计算策略的因子暴露
    # fsr : dateindex, dataframe, 简单收益率, 当天收益率
    def factor_expose(self, fsr):
        # 简单线性回归
        def cal_ols(x, y):
            x = x.values
            y = y.values
            X = sm.add_constant(x)
            model = sm.OLS(y, X)
            results = model.fit()
            r2 = results.rsquared
            alpha = results.params[0]
            return r2, alpha
        sr0 = self.returns
        ind = fsr.index.map(lambda x: x.year).unique()
        df = pd.DataFrame(index=ind, columns=fsr.columns)
        df.loc[9999] = np.nan
        for i in df.index:
            r2, alpha = cal_ols(x=fsr, y=sr0)
            for j in df.columns:
                if i == 9999:
                    r2, alpha = cal_ols(x=fsr[[j]], y=sr0)
                    df.loc[i, j] = r2*100
                else:
                    r2, alpha = cal_ols(x=fsr[[j]].loc[str(i)], y=sr0.loc[str(i)])
                    df.loc[i, j] = r2*100
        df = df.astype('float')
        self.fexpose = df
        return df
    
    # 收益率分布直方图
    def returns_displot(self, bins=20):
        import seaborn as sns 
        plt.figure(figsize=(12, 8))  #设置画布的大小
        sns.set_palette("hls")       #设置所有图的颜色，使用hls色彩空间
        sns.displot(self.returns*100, color="steelblue",bins=bins)
        plt.xlabel('收益率(%)',fontsize=15)           #添加x轴标签，并改变字体
        plt.ylabel('频数',fontsize=15)   #添加y轴变浅，并改变字体
        plt.grid(linestyle='-')   #添加网格线
        plt.xticks(fontsize=12)   #改变x轴字体大小
        plt.yticks(fontsize=12)   #改变y轴字体大小
        sns.despine(ax=None, top=True, right=True, left=True, bottom=True)    #将图像的框框删掉
        plt.show()


############################################################################################
################################### 处理Strat对象 ##############################################
############################################################################################
class StratPost(ReturnsPost):
    # 持仓表、单边交易成本、market()
    def __init__(self, strat0, market=None, \
                 benchmark=0, stratname='策略', freq='day', rf=0.03, fast=False, comm=0):
        #self.strat = strat0
        self.market = market
        self.comm = comm
        self.turnover = strat0.turnover
        self.df_turnover = strat0.df_turnover 
        self.df_weight = strat0.df_weight
        self.keeppool_rank = strat0.keeppool_rank  
        self.df_contri = (1+strat0.df_contri)*(1-strat0.df_turnover*comm)-1
        super().__init__((1+strat0.returns)*(1-self.turnover*comm)-1,\
                                benchmark, stratname, freq, rf, fast)
    def detail(self):
        # 空仓时间
        self.df_details.loc[2, 'col0'] = '空仓时间（日）'
        if 'cash' in self.df_weight.columns: 
            self.df_details.loc[3, 'col0'] = (self.df_weight.drop(columns='cash')==0\
                                              ).all(axis=1).sum()
        else:
            self.df_details.loc[3, 'col0'] = 0
        # 策略执行
        self.df_details.loc[0, 'col6'] = '年化换手，持股周期（日）'
        self.df_details.loc[1, 'col6'] = '%s, %s'%(round(self.turnover.sum()/self.years),\
                                                   round(500/(self.turnover.sum()/self.years)))
        super().detail()
    def plot_turnover(self):
        plt, fig, ax = FB.display.matplot()
        ax.plot(self.turnover*250, alpha=0.2)
        ax.plot(self.turnover.rolling(20).mean()*250, label='20日滚动换手')
        ax.plot(self.turnover.rolling(250).mean()*250, label='250日滚动换手')
        ax.legend()
        check_output()
        plt.savefig('./output/turnover.png')
        plt.show()
    # 输出每根K线的持仓、个股仓位、个股收益贡献（费前）、收益率、换手率
    def get_holdtable(self):
        # 持仓数量
        held = pd.DataFrame(self.df_weight.stack()[self.df_weight.stack()!=0]).\
                    rename(columns={0:'weight'})
        # 持仓标的收益贡献
        held = held.join(pd.DataFrame(self.df_contri.shift(-1).stack()).\
                         rename(columns={0:'contri'})).fillna(0).\
                            loc[self.keeppool_rank.index]
        # 是否加入持仓品种名称
        try:
            if 'name' in self.market.columns:
                held = held.join(self.market['name'])
                held.loc[held.index[held.index.get_level_values(1)=='cash'], 'name'] = '现金'
        except:
            pass
        self.held = held
        # 持仓表(名称，代码，持仓量，持仓占比)
        result_hold = pd.DataFrame()
        for date in self.held.index.get_level_values(0).unique():
            temp = self.held.loc[date].sort_values(by='weight', ascending=False)
            iamount = 0
            for idx,val in temp.iterrows():
                if 'name' in self.market.columns:
                    keystring = val['name']+'('+str(idx)+')'+ ', 仓位：'+\
                        str(round(100*val['weight'], 2))+'%'+\
                            ', 收益率：'+'%03d'%(1e4*val['contri'])
                else:
                    keystring = str(idx) + ', 仓位：'+str(round(100*val['weight'], 2))+'%'+\
                            ', 收益率：'+'%03d'%(1e4*val['contri'])
                result_hold.loc[date, 'hold%s'%iamount] = keystring
                iamount += 1
        result_hold = result_hold.join(pd.DataFrame(10000*self.returns).\
                                        rename(columns={0:'收益率(万)'}))
        result_hold = result_hold.join(pd.DataFrame(round(100*self.turnover,\
                                        2)).rename(columns={0:'换手率(%)'}))
        result_hold.index.name = '日期'
        self.result_hold = result_hold.sort_index(ascending=False)
        # excel列名
        import string
        A2Z = [i for i in string.ascii_uppercase]
        excel_columns = A2Z + [i+j for i in A2Z for j in A2Z]
        # 第一列是日期，宽度15或30，第二列到倒数第三列为持仓股票，宽度15或30，倒数两列为收益率和换手率，宽度12
        if 'name' in self.market.columns:
            col_width = {'A':20}|{excel_columns[1+i]:30 for i in range(len(self.result_hold.columns)-2)}|\
                                {excel_columns[len(self.result_hold.columns)-1+i]:8 for i in range(2)}
        else:
            col_width = {'A':20}|{excel_columns[1+i]:40 for i in range(len(self.result_hold.columns)-2)}|\
                                {excel_columns[len(self.result_hold.columns)-1+i]:8 for i in range(2)}
        FB.display.write_df(self.result_hold , "./output/持仓表", col_width=col_width, row_width={0:35})



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

        plt, fig, ax = FB.display.matplot(w=22)
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
        plt, fig, ax = FB.display.matplot(w=10)
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
        plt, fig, ax = FB.display.matplot()
        ax.plot(Close_count, label='累积交易次数 胜率：%s 盈亏比：%s'%(round(self.winrate, 2), round(self.odds, 2)))
        ax.set_xlim(self.net.index[0], self.net.index[-1])
        ax.legend()
        plt.gcf().autofmt_xdate()
        plt.savefig('trade.png')
        plt.show()
    def trade_monthly(self):
        from display import month_thermal
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
        plt, fig, ax = FB.display.matplot()
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





