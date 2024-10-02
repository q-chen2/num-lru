import numpy as np
import numba as nb
from typing import Any

pyhash = hash
hash = None  ## make sure old reference to `hash` is invalid.


@nb.njit
def myhash(x):
    # simple and fast naive hash function using a large-enough prime number
    return 1_000_003 * x


class LRUTemplate:
    """LRU implementation based on Numpy and Numba.

    Design notes:
        The hashtable and doubly-linked list are stored in the same table.
        (idx), key(0), prev(1), next(2)
            0,     ..,      ..,      ..
            1,     ..,      ..,      ..
        Values are stored in another array.
        Convention for values of key:
        [0, inf): valid key, slot is taken
        -1: deleted slot, need it for linear probing
        (-inf, -2]: empty slot
    """

    def __init__(self, size: int, tab_size_ratio: float = 3.0, value_dtype: Any = nb.int64):
        """value_dtype can be any numpy struct/dtype. maybe also jitclass object (not tested)"""
        self.size = size
        self.cur_size: int = 0
        self.head: int = -2  ## **index** of head node
        self.tail: int = -2
        self.tab_size: int = int(np.ceil(size * tab_size_ratio))
        ## this must be 2lines, otherwise numba won't compile bc type infered as int64
        self.table = np.ones((self.tab_size, 3), dtype=np.int32)
        self.table *= -2
        self.value = -2 * np.ones(self.tab_size, dtype=value_dtype)
        self.__debug_perf_find_calls = 0
        self.__debug_perf_find_hops = 0
        self.__debug_perf_recycle_calls = 0
        self.__debug_perf_recycle_hops = 0

    def _debug_reset_perf_count(self):
        self.__debug_perf_find_calls = 0
        self.__debug_perf_find_hops = 0
        self.__debug_perf_recycle_calls = 0
        self.__debug_perf_recycle_hops = 0

    def _debug_show_perf_count(self):
        """find_hops, find_calls, recycle_hops, recycle_calls, recycles"""
        return (
            self.__debug_perf_find_hops,
            self.__debug_perf_find_calls,
            self.__debug_perf_recycle_hops,
            self.__debug_perf_recycle_calls,
        )

    def _find_slot(self, key, get_free_only=0):
        """Returns (idx_match, idx_best).
        Some possible returns (idx always >= 0):
            * idx1, idx2 >= 0 :
                match found, idx1 is old slot, idx2 is new best slot.
                in this case, the new value should be inserted to
                idx2 (and prev/next should be updated too)
            * idx1 = -1, idx    : no match found, idx is the best empty slot
        """
        self.__debug_perf_find_calls += 1
        idx_match = idx_best = -1
        idx = myhash(key) % self.tab_size
        init_idx = idx  ## for debugging only
        end_reached_once = 0
        hops = 0  ## for debug
        while hops < 2 * self.tab_size:
            hops += 1
            if get_free_only and idx_best >= 0:
                return -1, idx_best
            self.__debug_perf_find_hops += 1
            key_cur = self.table[idx, 0]
            if key_cur >= 0:  ## slot is taken
                if key_cur == key:
                    idx_match = idx
                    if idx_best < 0:
                        idx_best = idx
                    return idx_match, idx_best
            else:  ## key_cur == -2  found EMPTY slot
                if idx_best < 0:
                    return -1, idx
                else:
                    return -1, idx_best
            idx += 1
            if idx >= self.tab_size:
                if end_reached_once:
                    return -1, idx_best
                end_reached_once = 1
                idx = 0
        raise Exception("This should be unreachable")

    def _move(self, from_, to):
        """Move node from index from_ to index to."""
        # print(f'__DEBUG move from {from_} to {to}.')
        self.value[to] = self.value[from_]
        self.table[to, :] = self.table[from_, :]
        prev, next = self.table[to, 1], self.table[to, 2]
        # self.table[from_, 0] = -1
        self.table[from_, 0] = -2  ## debug
        if prev >= 0:
            self.table[prev, 2] = to
        if next >= 0:
            self.table[next, 1] = to
        if self.head == from_:
            self.head = to
        if self.tail == from_:
            self.tail = to

    def get(self, key, move_head=1):
        """Returns the key by value if found or -2 if not found"""
        idx_match, idx_best = self._find_slot(key)
        if idx_match < 0:
            return -2
        if move_head:
            if idx_best != idx_match:
                self._move(idx_match, idx_best)
            if self.head != idx_best:
                self._make_head(idx_best)
            return self.value[idx_best]
        else:
            return self.value[idx_match]

    def put(self, key, value, check_match=1):
        """Put new <key, value> pair in the hashtable.

        Args:
            check_match: Bool. When False, insert at first empty slot without
                searching through the hashtable for a match. This is common for
                LRU use case where we usually first use get to check if there
                is match, and only put when there is no match.
        """
        idx_match, idx_best = self._find_slot(key, get_free_only=check_match)
        if idx_match >= 0:  ## match found
            if idx_best != idx_match:
                self._move(idx_match, idx_best)
            ## replace the values and make the node head
            self.value[idx_best] = value
            if self.head != idx_best:
                self._make_head(idx_best)
        else:  ## no match found, need to insert new node
            if self.cur_size == self.size:
                ## pop the old tail, cur_size - 1
                old_tail = self.tail
                old_tail_prev = self.table[old_tail, 1]
                self.tail = old_tail_prev
                self.table[self.tail, 2] = -2
                self._pop(old_tail)
                self.cur_size -= 1
                ## recycling changes position, so need to find idx_best again
                _, idx_best = self._find_slot(key, get_free_only=True)
            ## insert the new node, make head
            self.table[idx_best, 0] = key
            self.table[idx_best, 1] = -2
            self.value[idx_best] = value
            old_head = self.head
            if old_head < 0:  ## the cache is still empty
                self.tail = idx_best
            else:
                self.table[idx_best, 2] = old_head  ## new_node.next = old_head
                self.table[old_head, 1] = idx_best  ## old_head.prev = new_node
            self.head = idx_best
            self.cur_size += 1

    def _make_head(self, node):
        ## node must not be head already
        old_head = self.head

        if node == self.tail:
            self.tail = self.table[node, 1]
            self.table[self.tail, 2] = -2
        else:
            # match_node.next.prev = match_node.prev
            node_next = self.table[node, 2]
            self.table[node_next, 1] = self.table[node, 1]

        # match_node.prev.next = match_node.next
        node_prev = self.table[node, 1]
        self.table[node_prev, 2] = self.table[node, 2]
        self.table[node, 2] = old_head
        self.table[old_head, 1] = node
        self.head = node
        self.table[node, 1] = -2

    def _pop(self, node):
        """Pop the node with no-tombstone method. Assumes the node is occupied."""
        self.__debug_perf_recycle_calls += 1

        i = j = node
        while True:
            self.__debug_perf_recycle_hops += 1
            j = (j + 1) % self.tab_size
            if self.table[j, 0] == -2:
                break
            k = myhash(self.table[j, 0]) % self.tab_size
            if (j > i and (k <= i or k > j)) or (j < i and (k <= i and k > j)):
                self._move(j, i)
                i = j
        self.table[i, 0] = -2


def make_lru(value_dtype: Any = nb.int64):
    _lru_jit_spec = [
        ("size", nb.int64),
        ("cur_size", nb.int64),
        ("head", nb.int64),
        ("tail", nb.int64),
        ("tab_size_ratio", nb.float64),
        ("tab_size", nb.int64),
        ("table", nb.int32[:, :]),  ## np.iinfo('int32').max = 2_147_483_647
        ("value", value_dtype[:]),
        ("__debug_perf_find_calls", nb.int64),
        ("__debug_perf_find_hops", nb.int64),
        ("__debug_perf_recycle_calls", nb.int64),
        ("__debug_perf_recycle_hops", nb.int64),
    ]

    cls = nb.experimental.jitclass(_lru_jit_spec)(
        LRUTemplate
    )  # (size, tab_size_ratio, value_dtype)
    # JITTED[value_dtype] = cls
    # return JITTED[value_dtype](size, tab_size_ratio)
    # _ = cls(50, 3, value_dtype)
    # return cls(size, tab_size_ratio, value_dtype)

    # def get_jitted(*args, **kwargs):
    #     kwargs['value_dtype'] = value_dtype
    #     # return JITTED[value_dtype](*args, **kwargs)
    #     return cls(*args, **kwargs)

    # return get_jitted
    return cls


LRU = make_lru()
