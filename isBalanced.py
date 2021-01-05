
#!/bin/python3

import math
import os
import random
import re
import sys
from collections import deque

openSet = set("{([")
closeSet = set("})]")
cMap = dict(zip(list("})]"),list("{([")))

def isBalanced(s):
    d = deque()
    for c in s:
        if c in openSet:
            d.append(c)
        elif c in closeSet:
            if not d or d.pop() != cMap[c]:
                return "NO"
        continue

    return "NO" if d else "YES"

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        s = input()

        result = isBalanced(s)

        fptr.write(result + '\n')

    fptr.close()
