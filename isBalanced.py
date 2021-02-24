
#!/bin/python3

import math
import os
import random
import re
import sys
from collections import deque

class Node(object):
    def __init__(self, data, next=None):
        self.data = data
        self.next = next
class Stack(object):
    def __init__(self, top=None):
        self.top = top

    def push(self, data):
        self.top = Node(data, self.top)

    def pop(self):
        if self.top is None: return None
        data = self.top.data
        self.top = self.top.next
        return data

    def top(self):
        return self.top.data if self.top else -1

    def is_empty(self):
        return self.top != None
class StackMin(Stack):
    def __init__(self, top=None):
        super(StackMin, self).__init__(top)
        self.min = sys.maxsize

    def getMin(self):
        return self.top.data[-1] if self.top else -1

    def push(self, data):
        if data < self.min: self.min = data
        super(StackMin, self).push((data,self.min))

    def pop(self):
        retVal, self.min = (super(StackMin, self).pop()[0], self.getMin()) if self.top else (None, sys.maxsize)
        return retVal

    def top(self):
        return self.top.data[0] if self.top else -1
openSet = set("{([")
closeSet = set("})]")
cMap = dict(zip(list("})]"),list("{([")))

class Solution:
    def isBalanced(self, s):
        d = deque()
        for c in s:
            if c in openSet: d.append(c)
            elif c in closeSet:
                if not d or d.pop() != cMap[c]: return False
        return False if d else True
    def braces(self, A):
        stax = []; i = 0
        while i < len(A):
            c = A[i]
            if c in '(+-*/': stax.append(c)
            elif c == ')':
                if stax[-1] == '(': return 1 # no operator inside ()
                else:
                    while stax and stax[-1] != '(': stax.pop() # pop expression inside ()
                    stax.pop() # pop "("
            i += 1
        return 0

if __name__ == '__main__':
    sol = Solution()
    tData = ["((a+b))", "(a + (a + b))", "(a*(a))", "(a)"]
    tRes = [1, 0, 1, 1]
    # tData = ["()", "()[]{}", "(]", "([)]", "{[]}", "(()())", "(()"]
    # tRes = [True, True, False, False, True, True, False]
    for i,tdata in enumerate(tData):
        assert sol.braces(tdata) == tRes[i]
        # assert sol.isBalanced(tdata) == tRes[i]
