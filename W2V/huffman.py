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
            n1 = heapq.heappop(self.heap)
            n2 = heapq.heappop(self.heap)
            merge_node = Node(None, n1.freq + n2.freq)
            merge_node.left = n1
            merge_node.right = n2
            heapq.heappush(self.heap, merge_node)

    def _make_codes(self):
        root = heapq.heappop(self.heap)
        self._bfs(root, "")
        self.code2word = {i: w for i, w in self.word2code.items()}

    def _bfs(self, node, current_str):
        if node.left is None and node.right is None:
            self.word2code[node.word] = current_str
        if node.left is not None:
            self._bfs(node.left, current_str + "0")
        if node.right is not None:
            self._bfs(node.right, current_str + "1")

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
