import numpy as np

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
        # 基准
        self.benchmark = benchmark
        # 评价指标
        # 年化收益率
        years = (self.net.index[-1]-self.net.index[0]).days/365
        return_total = self.net[-1]/self.net[0]
        self.return_annual = return_total**(1/years)-1
        # 年化波动率 shrpe
        self.std_annual = np.exp(np.std(np.log(self.returns+1))*np.sqrt(250)) - 1
#        self.std_annual = np.std(np.log(self.returns))*np.sqrt(250) 
        self.sharpe = (self.return_annual - self.rf)/self.std_annual
        # 回撤
        a = np.maximum.accumulate(self.net)
        self.drawdown = (a-self.net)/a
        # 持仓标的
        held = world.df_hold.reset_index().melt(id_vars=['date']).set_index(['date', 'code'])
        held = held['value'] * world.market['close']
        self.held = held[(held != 0)&~np.isnan(held)]
# 净值曲线
    def pnl(self, timerange=None, filename=None):
        plt, fig, ax = matplot()
        # 只画一段时间内净值（用于展示局部信息,只列出sharpe）
        if type(timerange) != type(None):
            # 时间段内净值与基准
            net = self.net.loc[timerange[0]:timerange[1]]
            returns = self.returns.loc[timerange[0]:timerange[1]]
            # 计算夏普
            years = (timerange[1]-timerange[0]).days/365
            return_annual = (net[-1]/net[0])**(1/years)-1
            std_annual = np.exp(np.std(np.log(returns))*np.sqrt(250)) - 1
            sharpe = (return_annual - self.rf)/std_annual
            ax.text(0.7,0.05,'Sharpe:  {}'.format(round(sharpe,2)), transform=ax.transAxes)
            ax.plot(net/net[0], c='C0', label='p&l')
            if self.benchmark != None:
                benchmark = self.benchmark.loc[timerange[0]:timerange[1]]
        # colors of benchmark
                colors_list = ['C4','C5','C6','C7']
                for i in range(len(benchmark.columns)):
                    ax.plot((benchmark[benchmark.columns[i]]+1).cumprod(), 
                            c=colors_list[i], label=benchmark.columns[i])
            plt.gcf().autofmt_xdate()
        else: 
    #评价指标
            ax.text(0.7,0.05,'年化收益率: {}%\n夏普比率:   {}\n最大回撤:   {}%'.format(
            round(100*self.return_annual,2), round(self.sharpe,2), round(100*max(self.drawdown),2)), transform=ax.transAxes)
        # 净值与基准
            ax.plot(self.net, c='C0', label='策略')
            if self.benchmark != None:
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
            plt.gcf().autofmt_xdate()
        if type(filename) == type(None):
            plt.savefig('pnl.png')
        else:
            plt.savefig(filename)
        plt.show()