from functools import wraps


def benchmarkit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        import time
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f"{func.__name__} took {t2 - t1} seconds")
        return result

    return wrapper