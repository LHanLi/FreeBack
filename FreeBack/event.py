




# 在事件发生后，持有n日后(事件次bar open至+Tbar收盘价（T=0为次bar开盘至收盘的收益）)，收益率
#dict_date:  key thscode, value list of date(事件后可以买入的日期)
# hold_days = list(range(61))
#df_price:   date thscode  open close
def event_return(dict_date, df_price, hold_days):
    # 建立dataframe 前两列为合约与日期
    columns = ['thscode', 'date']
    # return_1 代表持有1天，即当天(0)的收盘价与开盘价之比
    for i in hold_days:
        columns.append('return_%d'%(i+1))
    df_return = pd.DataFrame(columns = columns)
    # 每个合约
    for i in list(dict_date.keys()):
        # 事件日期列表
        list_date = dict_date[i]
        # 如果在行情数据中没有,输出，跳过
        if(i not in df_price.thscode.unique()):
            print('not found',i)
            continue
        # 筛选出此合约行情
        df_ = df_price[df_price.thscode == i]
        # 按日期排序
        df_ = df_.sort_values(by = 'date')
        df_ = df_.reset_index(drop=True)
        # 每一次事件
        for start_date in list_date:
            # 公告日期为实际发布公告日期后次日，在此时可以直接买入
            # 为交易日则直接买入 
            if start_date in df_.date.values:
                start = df_[df_.date == start_date]
            # 如果不是交易日则后延
            else:
                # 最多尝试30天
                try_num = 0
                while try_num < 30:
                    try_num += 1
                    start_date += datetime.timedelta(1)
                    if start_date in df_.date.values:
                        start = df_[df_.date == start_date]
                        break
                # 没有找到则下一个日期或转债
                if(try_num==30):
                    print('fail: ', i, start_date)
                    continue
            # 持有到end，需存在行情数据
            dur = [start.index[0]+dur_i for dur_i in hold_days if (start.index[0]+dur_i) < len(df_.index)]
            end = df_.loc[dur]
    #       公告日开盘价到持有日收盘价 收益率
            return_list = list((end.close/start.open.values[0]).apply(lambda x: math.log(x)))
        # 字典 value
            dict_values = [i,start_date]
            dict_values.extend(return_list)
            append_dict = dict(zip(columns, dict_values))
            df_return = df_return.append(append_dict, ignore_index=True)
        
    return df_return