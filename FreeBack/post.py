import numpy as np
import pandas as pd
import datetime

# matplot绘图
def matplot(r=1, c=1, sharex=False, sharey=False, w=8, d=5):
    import seaborn as sns
    import matplotlib.pyplot as plt
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
def plot_thermal(series_returns):
    df_returns = series_returns
    # 先转化为对数收益率
    df_lr = df_returns.apply(lambda x: np.log(x+1))
    df_lr = df_lr.reset_index()
    # 筛出同月数据
    df_lr['month'] = df_lr['date'].apply(lambda x: x - datetime.timedelta(x.day-1)) 
    df_lr = df_lr[['month', 0]]
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

# 传入barbybar运行完毕的world对象
class Post():
    #
    def __init__(self, world, benchmark=None):
        # 无风险利率
        self.rf = 0.03
        # 净值曲线
        self.net = world.series_net/world.series_net.iloc[0]
        self.returns = self.net/self.net.shift() - 1
        self.lr = np.log(self.returns + 1)
        self.returns.index.name = 'date'
        # 基准
        self.benchmark = benchmark
        # 评价指标
        # 年化收益率
        years = (self.net.index[-1]-self.net.index[0]).days/365
        return_total = self.net[-1]/self.net[0]
        self.return_annual = return_total**(1/years)-1
        # 年化波动率 shrpe
        #self.std_annual = np.std(np.log(self.returns+1))*np.sqrt(250)
        self.std_annual = np.std(self.returns)*np.sqrt(250)
        self.sharpe = (self.return_annual - self.rf)/self.std_annual
        # 回撤
        a = np.maximum.accumulate(self.net)
        self.drawdown = (a-self.net)/a
        # 持仓标的
        self.df_hold = world.df_hold
        held = world.df_hold.reset_index().melt(id_vars=['date']).set_index(['date', 'code'])
        held = held['value'] * world.market['close']
        self.held = held[(held != 0)&~np.isnan(held)]
        # 现金
        self.cash = world.series_cash
        # 交割单
        self.excute = world.df_excute.reset_index().set_index(['date', 'code', 'unique'])
        # 换手率
        occurance_amount = self.excute['occurance_amount'].groupby('date').sum()
        #turnover = (occurance_amount/(self.net*self.cash.iloc[0])).fillna(0)
        turnover = (occurance_amount/world.series_net).fillna(0)
        self.turnover = turnover.mean()*250
        # 超额收益 默认第一个benchmark
        if type(benchmark) == type(None):
            benchmark = pd.DataFrame(index = world.series_net.index)
            benchmark['zero'] = 0
            self.benchmark = benchmark
        self.excess_lr = self.lr  - np.log(benchmark[benchmark.columns[0]]+1)
# 净值曲线
    def pnl(self, timerange=None, detail=False, filename=None, log=False, excess=True):
        plt, fig, ax = matplot()
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
            ax.plot(self.net, c='C0', label='策略')
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
        ax.plot((self.net/self.net.shift(20)-1)*100, c='C0', label='month return')
        #ax.plot((post0.net/post0.net.shift(60)-1)*4, c='C1', label='quarter return')
        #ax.plot((post0.net/post0.net.shift(120)-1)*2, c='C3', label='half year return')
        ax2 = ax.twinx()
        ax2.plot((self.net/self.net.shift(250)-1)*100, c='C3', label='year return')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.set_ylabel('滚动时点向前回溯收益 (%)')
        plt.gcf().autofmt_xdate()
        plt.savefig('rolling_return.png')
# 月度收益
    def pnl_monthly(self):
        plt,fig,ax = plot_thermal(self.lr)
        plt.savefig('pnl_monthly.png')
# 月度超额收益
    def pnl_excess_monthly(self):
        plt,fig,ax = plot_thermal(self.excess_lr)
        plt.savefig('pnl_excess_monthly.png')
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

        # 筛出同月数据
        trades = self.Close_amount.reset_index()
        trades['month'] = trades['date'].apply(lambda x: x - datetime.timedelta(x.day-1)) 
        trades = trades[['month', 0]]
        trades = trades.set_index('month')

        # 月度盈利（亏损）交易的次数与平均盈利（亏损）。
        trades_win = trades > 0
        trades_loss = trades < 0
        count_win = trades_win.groupby('month').sum()
        count_loss = trades_loss.groupby('month').sum()
#        mean_win = trades[trades_win[0]].groupby('month').mean()
#        mean_loss = trades[trades_loss[0]].groupby('month').mean()
        # 月度胜率和盈亏比
        self.month_winrate = count_win/(count_win + count_loss)
#        self.month_odds = -mean_win/mean_loss
        # 总体数据
        self.winrate = trades_win.sum().values[0]/len(trades)
        self.odds = -(trades[trades_win[0]].mean()/trades[trades_loss[0]].mean()).values[0]
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
    def trade_monthly(self):
        plt,fig,ax = month_thermal(self.month_winrate, 0.5)
        plt.savefig('trade_monthly.png') 
# 仓位分析
    def position(self):
        plt, fig, ax = matplot()
        held_amount = self.held.groupby('date').sum()
        ax.plot(held_amount/(self.cash+held_amount), label='非现金仓位')
        ax.plot(self.held.groupby('date').max()/(self.cash+held_amount), label='第一大持仓仓位')
        ax.set_xlim(self.held.index[0][0], self.held.index[-1][0])
        ax.legend(loc = 'upper left')
        ax2 = ax.twinx()
        ax2.plot(self.held.groupby('date').count(), c="C2", label='持有标的数量')
        ax2.legend(loc = 'upper right')
        plt.gcf().autofmt_xdate()
        plt.savefig('position.png')

