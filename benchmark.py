import time

from numba import njit

from lru import LRU


@njit
def get_fibo(n, cache):
    if n < 3:
        return 1

    maybe = cache.get(n, 0)
    if maybe >= 0:
        return maybe
    result = get_fibo(n - 1, cache) + get_fibo(n - 2, cache)
    cache.put(n, result)
    return result


def warmup():
    cache = LRU(4)
    get_fibo(16, cache)


def test_fibonacci(n):
    cache = LRU(10)
    t0 = time.perf_counter_ns()
    res = get_fibo(n, cache)
    t1 = time.perf_counter_ns()
    print(f"fib({n})={res:<25} takes {(t1 - t0) /1000:.3f} us")
    return res


if __name__ == "__main__":
    warmup()
    for i in [60, 80, 92]:
        for _ in range(3):
            test_fibonacci(i)
