import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
import seaborn as sns
from pyecharts import options as opts
from pyecharts.charts import Kline,Bar,Grid,Line
from pyecharts.commons.utils import JsCode
import numpy as np
import datetime, xlsxwriter

###########################################################
######################## 静态图 ###########################
###########################################################


# matplot绘图
def matplot(r=1, c=1, sharex=False, sharey=False, w=13, d=7):
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

# 获取月度统计量
# method， 
# total_return 一般收益率总收益
# total_lr 对数收益率的总收益
# sum 求和
# mul 乘积 
def get_month(ser, method='total_return'):
    if ser.name == None:
        name = 0 
    else:
        name = ser.name
    df = ser.reset_index()
    # 筛出同月数据
    df['month'] = df['date'].apply(lambda x: x - datetime.timedelta(x.day-1))
    df = df.set_index('month')[name]
    # 月度统计量
    if method=='total_return':
        return (((df+1).groupby('month')).prod()-1)*100
    elif method=='total_lr':
        return (np.exp(df.groupby('month').sum()) - 1)*100
    elif method=='sum':
        return df.groupby('month').sum()
    elif method=='mul':
        return df.groupby('month').prod()

# 月度数据热力图  
# period_value 为pd.Series index month ‘2023-7-1’  value  0: ***
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
    dates, data = list(period_value.index), period_value.values
    #period_value.iloc[:,0].values
    i, j, plot, calendar = calendar_array(dates, data)
    # 绘制热力图
    plt, fig, ax = matplot()
    ax.imshow(plot, aspect='auto')
    # 设置纵坐标 年份
    i = np.array(list(set(i)))
    i.sort()
    ax.set(yticks=i)
    # 年份
    years = sorted(list(set([i.year for i in period_value.index])))
    ax.set_yticklabels(years)
    # 设置横坐标 月份
    ax.set(xticks=list(range(12)))
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
# 月度收益热力图    输入对数收益率的series 
def plot_thermal(df_returns):
    # 先转化为对数收益率
    #df_lr = df_returns.apply(lambda x: np.log(x+1))
    df_lr = df_returns.reset_index()
    # 筛出同月数据
    df_lr['month'] = df_lr['date'].apply(lambda x: x - datetime.timedelta(x.day-1))
    df_lr = df_lr[['month', (lambda x: 0 if x==None else x)(df_returns.name)]]
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


# excel表
def write_df(df, name, title=True, index=True, col_width={}, row_width={}):
    workbook = xlsxwriter.Workbook('%s.xlsx'%name)
    worksheet = workbook.add_worksheet()
    worksheet.freeze_panes(1, 1)   # 冻结首行
    # 列宽设置
    for k,v in col_width.items():
        worksheet.set_column('%s:%s'%(k,k), v)
    # 行高设置
    for k,v in row_width.items():
        worksheet.set_row(k, v)
    # 格式
    general_prop = {'font_size':10, 'align':'center', 'valign':'vcenter', 'text_wrap':True}
    format_title = workbook.add_format(dict([(k,general_prop[k]) for k in general_prop]\
                                +[('font_size',14), ('bold',True),\
                                    ('bg_color','#0066ff'), ('font_color','#ffffff')]))
    format_text = workbook.add_format(dict([(k,general_prop[k]) for k in general_prop]\
                                +[('num_format', '#,##0.0')]))
    format_date = workbook.add_format(dict([(k,general_prop[k]) for k in general_prop]\
                                +[('num_format', 'yyyy-mm-dd')]))
    format_time = workbook.add_format(dict([(k,general_prop[k]) for k in general_prop]\
                                +[('num_format', 'yyyy-mm-dd hh:MM')]))
    def judge_format(text):
        return format
        # 标题与序号 
    if index:
        if (type(df.index[0])==type(datetime.date))|\
            (type(df.index[0])==type(pd.to_datetime('2000'))):
            if (df.index[0].hour+df.index[0].minute)==0:
                # 日期格式
                worksheet.write_column("A%s"%(int(title)+1), list(df.index), format_date)
            else:
                # 精确到分钟
                worksheet.write_column("A%s"%(int(title)+1), list(df.index), format_time)
        else:
            worksheet.write_column("A%s"%(int(title)+1), list(df.index), format_text)
        if title:
            if df.index.name==None:
                worksheet.write(0, 0, '', format_title)
            else:
                worksheet.write(0, 0, df.index.name, format_title)
            worksheet.write_row("B1", list(df.columns), format_title)
    elif title:
        worksheet.write_row("A1", list(df.columns), format_title)
    # Iterate over the data and write it out row by row.
    row=int(title)
    col=int(index)
    for i, r in df.iterrows():
        for j, v in r.items():
            try:
                worksheet.write(row, col, v, format_text)
            except:
                worksheet.write(row, col, '')
            col += 1
        col = int(index)
        row += 1
    workbook.close()



###########################################################
######################## 动态交互图 ###########################
###########################################################





# plot_data:df格式,index为datetime,必要列为close,high,low,open,vol
# 其他列自由添加，默认绘制于主图
# 其中vol绘制于副图 其他列为指标列，绘制于主图
def plot_kbar(plot_data, title='个股行情'):
    # 画图大小
    big_width = 1400
    big_height = big_width*1000/1800
    
    # 主图，k线
    kbar_data = plot_data[['open', 'close', 'low', 'high']].values.tolist()
    kline = (
        Kline()
        .add_xaxis([str(i.date()) for i in plot_data.index])  # 日期坐标
        .add_yaxis(                                           # k线
            "kline",
            kbar_data,
            itemstyle_opts=opts.ItemStyleOpts(
                color="#ec0000",
                color0="#00da3c",
                border_color="#8A0000",
                border_color0="#008F28",
            ),
        )
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(is_scale=True),
            yaxis_opts=opts.AxisOpts(
                is_scale=True,
                splitarea_opts=opts.SplitAreaOpts(
                    is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
                ),
            ),
            ## 滑块控制主图和幅图，滑块位置, 滑块起始左边位置，起始右边位置
            # 隐藏幅图的滑轨
            datazoom_opts=[\
                opts.DataZoomOpts(type_='inside', xaxis_index=[0,1],\
                                             #pos_top='60%', pos_bottom='65%',\
                                             range_start=80,\
                                             range_end=100,)],
            ## 鼠标位于图中任意点展示详细信息 
            tooltip_opts=opts.TooltipOpts(
                    trigger="axis",
                    axis_pointer_type="cross",
                    background_color="rgba(245, 245, 245, 0.8)",
                    border_width=1,
                    border_color="#ccc",
                    textstyle_opts=opts.TextStyleOpts(color="#000"),
                ),
            ##
            #visualmap_opts=opts.VisualMapOpts(
            #        is_show=False,
            #        dimension=2,
            #        series_index=5,
            #        is_piecewise=True,
            #        pieces=[
            #            {"value": 1, "color": "#00da3c"},
            #            {"value": -1, "color": "#ec0000"},
            #        ],
            #    ),
            # 在主图显示幅图详情
            axispointer_opts=opts.AxisPointerOpts(
                    is_show=True,
                    link=[{"xAxisIndex": "all"}],
                    label=opts.LabelOpts(background_color="#777"),
                ),
            # 绘制阴影功能
            brush_opts=opts.BrushOpts(
                    x_axis_index="all",
                    brush_link="all",
                    out_of_brush={"colorAlpha": 0.1},
                    brush_type="lineX",
                ),
            # 标题
            title_opts=opts.TitleOpts(title="%s"%title),)
        )
    # 主图上绘制的其他数据
    kline_others = (set(plot_data.columns)-set(['close', 'high','low','open','vol']))
    lines_list = []
    for i in kline_others:
        lines_list.append((Line()
                .add_xaxis(xaxis_data=[str(i.date()) for i in plot_data.index])
                .add_yaxis(
                    series_name=i,
                    y_axis=plot_data[i].tolist(),
                    xaxis_index=1,
                    yaxis_index=1,
                    label_opts=opts.LabelOpts(is_show=False),)
            ))
    # 子图1，成交量
    bar = (
            Bar()
            .add_xaxis(xaxis_data=[str(i.date()) for i in plot_data.index])
            .add_yaxis(
                series_name="vol",
                y_axis=plot_data["vol"].tolist(),
                xaxis_index=1,
                yaxis_index=1,
                label_opts=opts.LabelOpts(is_show=False),
                itemstyle_opts=opts.ItemStyleOpts(
                    # 跟随主图颜色
                    color=JsCode(
                        """
                    function(params) {
                        var colorList;
                        if (barData[params.dataIndex][1] > barData[params.dataIndex][0]) {
                            colorList = '#ef232a';
                        } else {
                            colorList = '#14b143';
                        }
                        return colorList;
                    }
                    """
                    )
                ),
            )\
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(
                    type_="category",
                    grid_index=1,
                    axislabel_opts=opts.LabelOpts(is_show=False),
                ),
                legend_opts=opts.LegendOpts(is_show=False),
            ))
    # 总图
    grid_chart = Grid(
            init_opts=opts.InitOpts(
                width="%spx"%(big_width),
                height="%spx"%(big_height),
                animation_opts=opts.AnimationOpts(animation=False),
            )
        )
    ## 导入open、close数据到barData改变交易量每个bar的颜色
    grid_chart.add_js_funcs("var barData={}".format(plot_data[["open","close"]].values.tolist()))
    # 添加主图，副图
    for i in lines_list:
        overlap_kline = kline.overlap(i)
    grid_chart.add(
            overlap_kline,
            #kline,
            grid_opts=opts.GridOpts(pos_left="0%", pos_right="0%", height="60%"),
        )
    grid_chart.add(
            bar,
            grid_opts=opts.GridOpts(
                pos_left="0%", pos_right="0%", pos_top="70%", height="20%"
            ),
        )
    grid_chart.render("个股行情.html")
