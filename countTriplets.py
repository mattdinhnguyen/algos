import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from timeit import timeit

@timeit    
def countTriplets0(arr, r): #N
    vMap = Counter(arr) # N
    b = Counter()
    triplets = 0
    for j in arr: #N
        i = j//r
        k = j*r
        vMap[j]-=1
        if b[i] and vMap[k] and not j%r:
            triplets += b[i] * vMap[k]
        b[j]+=1
    return triplets

@timeit
def countTriplets(arr, r):
    count = 0
    dict = defaultdict(int)
    dictPairs = defaultdict(int)
    for i in arr[::-1]:
        if i*r in dictPairs:
            count += dictPairs[i*r]
        if i*r in dict:
            dictPairs[i] += dict[i*r]
        dict[i] += 1
    return count

if __name__ == '__main__':
    fptr = open("countTriplets.ut")
    t = int(fptr.readline().rstrip())
    for _ in range(t):
        nr = fptr.readline().rstrip().split()
        n = int(nr[0])
        r = int(nr[1])
        arr = list(map(int, fptr.readline().rstrip().split()))
        ans = countTriplets(arr, r)
        print(ans, countTriplets(arr, r), countTriplets0(arr, r))
    fptr.close()
