from collections import defaultdict
import os
from typing import List
# https://www.hackerrank.com/challenges/candies/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=dynamic-programming
def candies(n: int, a: List[int]) -> int:
    c, desc_buf = [1]*n, []
    for i in range(1, n):
        if a[i] < a[i-1]:
            if not desc_buf: desc_buf = [i-1] # descending buffer
            desc_buf.append(i)
            if not i == n-1: continue
        if a[i] > a[i-1]: # climb up
            c[i] = c[i-1] + 1
        if desc_buf:
            for extra, idx in enumerate(desc_buf[::-1]): # climb back up from bottom
                c[idx] = max(c[idx], extra + 1)
            del desc_buf[:]
    return sum(c)
def candies(n: int, a: List[int]) -> int:
    c = [1]*n; start = end = -1
    for i in range(1, n):
        if a[i] < a[i-1]:
            if start == -1: start = i-1 # descending buffer
            end = i
            if not i == n-1: continue
        if a[i] > a[i-1]: # climb up
            c[i] = c[i-1] + 1
        if start != -1:
            extra = 0
            for idx in range(end,start-1,-1): # climb back up from bottom
                c[idx] = max(c[idx], extra + 1)
                extra += 1
            start = end = -1
    return sum(c)

if __name__ == '__main__':
    fptr = open(os.path.dirname(__file__) + "/candies.ut")

    n = int(fptr.readline())

    arr = []

    for _ in range(n):
        arr_item = int(fptr.readline())
        arr.append(arr_item)

    result = candies(n, arr)

    print(161518, result)

    fptr.close()
