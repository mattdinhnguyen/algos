import math
import os
import random
import re
import sys
from timeit import timeit
from functools import cmp_to_key
import heapq

def countSwaps(a):
    count = 0
    for i in range(len(a)):
        for j in range(len(a)-1):
            if a[j] > a[j + 1]: #N**2
                a[j], a[j + 1] = a[j+1], a[j]
                count += 1
    print(f"Array is sorted in {count} swaps.")  
    print(f"First Element: {a[0]}")  
    print(f"Last Element: {a[-1]}")

class Player:
    def __init__(self, name, score):
        self.name = name
        self.score = score
        
    def __repr__(self):
        return self.name+'_'+self.score
        
    def comparator(a, b):
        if a.score > b.score:
            return -1
        elif a.score < b.score:
            return 1
        elif a.name > b.name:
            return 1
        elif a.name < b.name:
            return -1
        else:
            return 0

@timeit
def mergeSort(a):
  if len(a) > 1:
    mid = len(a)//2
    l = a[:mid]
    r = a[mid:]
    mergeSort(l)
    mergeSort(r)

    i = j = k = 0
    while i < len(l) and j < len(r):
      if l[i] < r[j]:
        a[k] = l[i]
        i += 1
      else:
        a[k] = r[j]
        j += 1
      k += 1
    while i < len(l):
      a[k] = l[i]
      k += 1
      i += 1
    while j < len(r):
      a[k] = r[j]
      j += 1
      k += 1

# Complete the insertionSort2 function below.
def insertSort2(a):
    n = len(a)
    for i in range(1,n):
        j = i - 1
        while j > 0 and a[i] < a[j]:
            a[i],a[j] = a[j],a[i]
            i -= 1
            j -= 1
        if a[j] > a[i]:
            a[i],a[j] = a[j],a[i]
        print(*a)
@timeit
def quickSort(a):
    if len(a) < 2:
        return a
    l = []
    r = []
    e = a[:1]
    pV = e[0]
    for v in a[1:]:
        if v < pV:
            l.append(v)
        elif v > pV:
            r.append(v)
        else:
            e.append(v)
    result = quickSort(l) + e + quickSort(r)
    print(' '.join(map(str, result)))
    return result

def partition(a, low, high):
  j = low-1
  pivot = a[high]
  for i in range(low,high):
      if a[i] < pivot:
        j += 1
        a[i], a[j] = a[j], a[i]
  j += 1
  a[j], a[high] = a[high], a[j]
  return j
# @timeit
def quickSort2(a, low, high):
  if low < high:
      pi = partition(a, low, high)
      quickSort2(a, low, pi-1)
      quickSort2(a, pi+1, high)

def maxheapify(a, n, i): #logN
    l = 2*i + 1
    r = l + 1
    largest = i
    if l < n and a[l] > a[largest]:
        largest = l
    if r < n and a[r] > a[largest]:
        largest = r
    if largest != i:
        a[i], a[largest] = a[largest], a[i]
        maxheapify(a, n, largest)

def maxheappop(a):
    a0 = a[0]
    a[0] = a.pop()
    maxheapify(a,len(a),0)
    return a0

def nlargest(k, a):
    nl = []
    for i in range(k):
        nl.append(maxheappop(a))
    return nl

def maxheap(a): #NlogN
    n = len(a)
    for i in range(n//2-1, -1, -1):
        maxheapify(a, n, i)

def heapsort(a): #NlogN
    n = len(a)
    for i in range(n//2-1, -1, -1):
        maxheapify(a, n, i)
    for i in range(n-1, 0, -1): #sorting: move max to end, heapify 0 for n-2 elements
        a[i], a[0] = a[0], a[i]
        maxheapify(a,i,0)

def heapqsort(iterable):
    h = iterable[:]
    heapq.heapify(h)
    return [heapq.heappop(h) for i in range(len(h))]

if __name__ == '__main__':
# insertionSort2(6, [1, 4, 3, 5, 6, 2])
# print(' '.join(map(str, quickSort([4, 5, 3, 7, 2]))))
# print(' '.join(map(str, quickSort([5, 8, 1, 3, 7, 9, 2]))))
    a = [5, 8, 1, 3, 7, 9, 2]
    b = a[:]
    c = a[:]
    d = a[:]
    e = a[:]
    maxheap(d)
    heapsort(a)
    quickSort2(b, 0, 6)
    mergeSort(c)
    print(heapqsort(a), a, b, c, quickSort([5, 8, 1, 3, 7, 9, 2]), nlargest(5,d))
    assert(a == b == c)
    countSwaps(e)
    print(e)

'''n = int(input())
data = []
for i in range(n):
    name, score = input().split()
    score = int(score)
    player = Player(name, score)
    data.append(player)
    
data = sorted(data, key=cmp_to_key(Player.comparator))
for i in data:
    print(i.name, i.score)
'''