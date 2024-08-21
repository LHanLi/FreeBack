import time
import datetime

# 函数运行时间装饰器
def check_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        excution_time = end_time-start_time
        print('函数', func.__name__, '执行时间为:', excution_time)
        return result
    return wrapper

# log 函数
def log(*txt):
    f = open('log.txt','a+')
    write_str = ('\n'+' '*35).join([str(i) for i in txt])
    f.write('%s,        %s\n' % (datetime.datetime.now(), write_str))
    f.close()





