import time

def timer(func):
    def wrapper(*args, **kwargs):
        t_start = time.time()
        func_res = func(*args, **kwargs)
        t_end = time.time()

        return func_res, t_end - t_start
    return wrapper