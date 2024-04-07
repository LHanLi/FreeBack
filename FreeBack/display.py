from pyecharts import options as opts
from pyecharts.charts import Kline,Bar,Grid,Line
from pyecharts.commons.utils import JsCode




# plot_data:df格式,index为datetime,必要列为close,high,low,open,vol
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
