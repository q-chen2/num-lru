import time
import numpy as np

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


def run_fibonacci(n):
    cache = LRU(10)
    t0 = time.perf_counter_ns()
    res = get_fibo(n, cache)
    t1 = time.perf_counter_ns()
    print(f"fib({n})={res:<25} takes {(t1 - t0) /1000:.3f} us")
    return res


@njit
def benchmark_get(lru, params, move_head):
    for i in range(params.shape[0]):
        lru.get(params[i], move_head)


@njit
def benchmark_put(lru, params, check_match):
    for i in range(params.shape[0]):
        # lru.put(params[i, 0], params[i, 1], check_match)
        lru.put(params[i][0], params[i][1], check_match)


@njit
def benchmark_getput(lru, params, _):
    for i in range(params.shape[0]):
        # res = lru.get(params[i, 0])
        res = lru.get(params[i][0])
        if res < 0:
            # lru.put(params[i, 0], params[i, 1])
            lru.put(params[i][0], params[i][1])


@njit
def fill_cache_with_dummy_values(lru, overfill=1):
    if overfill:
        for i in range(lru.size * 3):
            lru.put(i, i + 1)
    for i in range(lru.size):
        lru.put(i, i + 2)


def benchmark_lru(size, steps=1000, hit_rate=0.5, seed=None):
    lru = LRU(size)
    if seed is None:
        seed = np.random.randint(0, 10000)

    t0 = time.perf_counter()
    fill_cache_with_dummy_values(lru)
    sample_size = int(size / hit_rate)
    fill_cache_time = (time.perf_counter() - t0) * 1000

    rng = np.random.RandomState(seed + 1)
    params = {}
    params["get0"] = (rng.randint(0, sample_size, steps), 0)  ## get without move head
    params["get1"] = (rng.randint(0, sample_size, steps), 1)  ## get with move head
    params["put0"] = (
        rng.randint(2 * sample_size, 3 * sample_size, (steps, 2)),
        0,
    )  ## put, no match check
    params["put1"] = (rng.randint(0, sample_size, (steps, 2)), 1)  ## put, with match check
    params["getput"] = (rng.randint(0, sample_size, (steps, 2)), 0)  ## get, if no match then put

    func_map = {}
    func_map["get0"] = func_map["get1"] = benchmark_get
    func_map["put0"] = func_map["put1"] = benchmark_put
    func_map["getput"] = benchmark_getput
    t0 = time.perf_counter()

    t_res = {}
    for test in ["get0", "get1", "put0", "put1", "getput"]:
        # print('testing', test)
        fill_cache_with_dummy_values(lru, 1)
        t00 = time.perf_counter()
        func_map[test](lru, params[test][0], params[test][1])
        t11 = time.perf_counter()
        t_res[test] = (t11 - t00) / steps * 10**9  ## ns

    dt = (time.perf_counter() - t0) * 1000  # ms, total test time
    print(
        f"{size:>8}, {steps:>5}, {hit_rate:>02.2f}, {seed:>4},   "
        f"{t_res['get0']:>7.2f}, {t_res['get1']:>7.2f}, {t_res['put0']:>7.2f}, "
        f"{t_res['put1']:>7.2f}, {t_res['getput']:>7.2f},   "
        f"{fill_cache_time:>05.2f}"
    )


def warmup():
    cache = LRU(4)
    get_fibo(16, cache)
    benchmark_lru(size=1005, steps=10, hit_rate=0.5, seed=None)


if __name__ == "__main__":
    warmup()
    for i in [60, 80, 92]:
        for _ in range(3):
            run_fibonacci(i)

    for size in (1_000, 10_000, 100_000, 500_000, 1_000_000)[:]:
        for hit_rate in (0.8, 0.95)[:]:
            benchmark_lru(size=size, steps=1000, hit_rate=hit_rate, seed=None)
            benchmark_lru(size=size, steps=1000, hit_rate=hit_rate, seed=None)
            time.sleep(0.1)
