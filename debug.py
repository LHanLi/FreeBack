import time

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







