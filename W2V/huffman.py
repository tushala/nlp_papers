# -*- coding: utf-8 -*-
import heapq
from collections import deque


class Node:
    def __init__(self, word, freq):
        self.word = word
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        if not isinstance(other, Node):
            raise ValueError(f"{other}类型错误")
        return self.freq < other.freq


class HuffmanCoding:
    def __init__(self):
        self.heap = []
        self.word2code = {}
        self.code2word = {}
        self.start = 0
    def _insert(self, freq):
        for word, freq in freq.items():
            curnode = Node(word, freq)
            heapq.heappush(self.heap, curnode)

    def _merge(self):
        while len(self.heap) > 1:
            n1 = self.heap.pop()
            n2 = self.heap.pop()
            merge_node = Node(None, n1.freq + n2.freq)
            heapq.heappush(self.heap, merge_node)

    def _make_codes(self):
        root = self.heap[0]
        self._bfs(root, "")

    def _bfs(self, node, current_str):
    def build(self, freq):
        self._insert(freq)
        self._merge()
        self._make_codes()


if __name__ == '__main__':
    from collections import Counter

    processed = ["a"] * 5 + ["b"] * 1 + ["c"] * 2 + ["w"] * 6 + ["x"] * 3
    frequency = Counter(processed)
    hc = HuffmanCoding()
    hc.build(frequency)
    print(hc.__dict__)

x = [
    {"name":"b","born":1997},
    {"name":"c","born":1998},
    {"name":"d","born":1999},
    {"name":"e","born":1996},
     ]