#!/bin/python3

import math
import os
import random
import re
import sys
from heapq import heappop, heappush
from collections import defaultdict
# https://www.hackerrank.com/challenges/minimum-swaps-2/problem
# You are given an unordered array consisting of consecutive integers  [1, 2, 3, ..., n] without any duplicates.
# You are allowed to swap any two elements. You need to find the minimum number of swaps required to sort the array in ascending order.
# space/time N/N
def minimumSwaps(arr):
    minCost = 0
    i = 0
    while i < len(arr):
        if arr[i] != i+1:
            t = arr[i]-1
            arr[i], arr[t] = arr[t], arr[i]
            minCost += 1
        else:
            i += 1
    return minCost

def minimumSwapsN(arr):
    minCost = 0
    for i in range(len(a)):
        if arr[i] != i+1:
            idx = arr[i]-1
            arr[i], arr[idx] = arr[idx], arr[i]
            minCost += 1
    return minCost
    
def minimumSwaps1(arr):
    visited = set()
    minCost = 0
    for i,v in enumerate(arr):
        if i not in visited:
            visited.add(i)
            clen = 0
            j = v - 1
            while j != i:
                clen += 1
                visited.add(j)
                j = arr[j] - 1
            minCost += clen

    return minCost

def unOrderness(a):
    unorderness = 0
    for i,v in enumerate(a):
        if v != i+1:
            unorderness += 1
    return unorderness

# space/time N!/N!
def minimumSwaps0(arr):
    minCost = float("Inf")
    h = [(unOrderness(arr), 0, arr[:])]
    cache = {tuple(arr):0}

    while len(h) > 0:
        u, cost, a = heappop(h)
        # 0 == sorted
        if unOrderness(a) == 0:
            minCost = min(minCost, cost)
            if minCost <= 1:
                break

        elif cost < minCost:
            for i in range(len(a)-1):
                for j in range(i+1, len(a)):
                    b = a[:]
                    b[i], b[j] = b[j], b[i]
                    if tuple(b) not in cache or cache[tuple(b)] > cost+1:
                        cache[tuple(b)] = cost+1
                        heappush(h, (unOrderness(b), cost+1, b))
    return minCost
            
# reverse k elements
def reverse(arr, k):
    a = arr[:k]
    a.reverse()
    return  a + arr[k:]


def reverseSort(arr):
    for i in range(len(arr)-1):
        for j in range(len(arr)-1):
            if arr[j] > arr[j+1]:
                aux = reverse(arr[j:j+2], 2)
                arr = arr[:j] + aux + arr[j+2:]
    return arr

if __name__ == '__main__':
    fptr = open("swapSort.ut")
    for t in range(int(fptr.readline())):
        n = int(fptr.readline())
        arr = list(map(int, fptr.readline().rstrip().split()))
        print(arr)
        res = minimumSwaps(arr)
        print(res, )

    fptr.close()
