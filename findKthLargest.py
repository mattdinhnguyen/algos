from bestFirstSearchDP import is_sorted
from heapq import heapify, nlargest, nsmallest
from collections import defaultdict
from sorts import quickSort2, partition
from bisect import bisect_left
from typing import List
import os
# space/time N/N^2
'''Suppose you have an unordered array with N values.

$input = [3,6,2,18,29,4,5]

Implement a function that finds the Kth largest value in the array and return the value.

EG: If K = 0, return 29.
If K = 1, return 18.
If K = N-1, return 2.
'''

def findKthLargest(nums: List[int], k: int) -> int:
    return nlargest(k,nums)[-1]

def findKthSmallest(nums: List[int], k: int) -> int:
    return nsmallest(k,nums)[-1]

def findKthSmallestN(arr,k):
    cnt = [0]*1000
    minVal = min(arr) #N
    for i,v in enumerate(arr):
        cnt[v-minVal] += 1
    preSum = [0]
    for i,freq in enumerate(cnt,1):
        preSum.append(preSum[i-1]+freq)
    return bisect_left(preSum, k) + minVal -1
    
def findKthSmallest(arr, l, r, k):
    if k >= 0 and k <= r - l + 1:
        pos = partition(arr, l, r)
        if pos - l == k: return arr[pos]
        if pos - l > k: return findKthSmallest(arr, l, pos - 1, k)
        return findKthSmallest(arr, pos + 1, r, k + l - pos - 1)
    return 0
def findKthLargest(nums: List[int], k: int) -> int:
    return -findKthSmallest(list(map(lambda n: -n, nums)), 0, len(nums)-1, k-1)

# 38046 78149 97560 174498 3083
if __name__ == '__main__':
    fptr = open(os.path.dirname(__file__) + "/countInversions0.ut")

    for t in range(int(fptr.readline().rstrip())):
        n = int(fptr.readline().rstrip())
        arr = list(map(int, fptr.readline().rstrip().split()))
        # ans = findKthSmallest(arr, 0, len(arr)-1, n//2)
        print(arr)
        print(sorted(arr)[-n//2], findKthSmallestN(arr, n - n//2), findKthLargest(arr, n - n//2),findKthSmallest(arr, 0, n-1, n - n//2 -1))
    fptr.close()
