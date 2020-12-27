from collections import defaultdict, Counter
from copy import copy
from timeit import timeit
from heapq import heappush, heappop, heappushpop, heapify
from typing import List

def insertSort(a, v):
    if len(a) < 2:
        return a + [v] if v > a[-1] else [v,a[0]]
    mid = len(a)//2
    if mid > 0 and a[mid] > v:
        return insertSort(a[:mid], v) + a[v:]
    elif mid+1 < len(a) and a[mid] < v:
        return a[:mid+1] + insertSort(a[mid+1:], v)
    else:
        return a[:mid] + [v] + a[mid:]
# Python program for counting sort 
# which takes negative numbers as well 

def count_sort(arr): 
    max_element = int(max(arr)) 
    min_element = int(min(arr)) 
    range_of_elements = max_element - min_element + 1
    # Create a count array to store count of individual 
    # elements and initialize count array as 0 
    count_arr = [0 for _ in range(range_of_elements)] 
    output_arr = [0 for _ in range(d)] 

    # Store count of each character 
    for i in range(0, d): 
        count_arr[arr[i]-min_element] += 1

    # Change count_arr[i] so that count_arr[i] now contains actual 
    # position of this element in output array 
    for i in range(1, len(count_arr)): 
        count_arr[i] += count_arr[i-1] 

    # Build the output character array 
    for i in range(d-1, -1, -1): 
        output_arr[count_arr[arr[i] - min_element] - 1] = arr[i] 
        count_arr[arr[i] - min_element] -= 1

    # Copy the output array to arr, so that arr now 
    # contains sorted characters 
    for i in range(0, d): 
        arr[i] = output_arr[i] 

    return arr 


# Driver program to test above function 
arr = [-5, -10, 0, -3, 8, 5, -1, 10] 
# ans = count_sort(arr) 
# print("Sorted character array is " + str(ans))
def lb2xMedian1(countArr, d):
    countFreq = copy(countArr)
    for i in range(1,len(countFreq)):
        countFreq[i] += countFreq[i-1] # prefix sum
    a = b = 0
    if d%2:
        first = d//2 + 1
        for i,freq in enumerate(countFreq):
            if first <= freq:
                a = i*2
                break
    else:
        first = d//2
        second = first + 1
        for i,freq in enumerate(countFreq):
            if first <= freq and not a:
                a = i
            if second <= freq and not b:
                b = i
                break
    return a + b

def lb2xMedian(countArr, d):
    first = d//2
    second = first + 1
    total = a = b = 0
    if d%2:
        for v,cnt in enumerate(countArr):
            total += cnt
            if second <= total:
                a = 2*v
                break
    else:
        for v,cnt in enumerate(countArr):
            total += cnt
            if first <= total and not a:
                a = v
            if second <= total:
                b = v
                break
    return a+b

@timeit
def activityNotifications(expenditure, d):
    notifies = 0
    valCounts = [0]*201
    for val in expenditure[:d]:
        valCounts[val] += 1
    for i in range(d,len(expenditure)):
        median2x = lb2xMedian(valCounts, d)
        if expenditure[i] >= median2x:
            notifies += 1
        valCounts[expenditure[i-d]] -= 1
        valCounts[expenditure[i]] += 1

    return notifies
 
class MedianFinder:
    def __init__(self):
        self.heaps = [], []

    def addNum(self, num: int) -> None:
        small, large = self.heaps
        heappush(small, -heappushpop(large, num))
        if len(small) > len(large):
            heappush(large, -heappop(small))

    def rmNum(self,val) -> None:
        small, large = self.heaps
        if val == large[0]:
            heappop(large)
        elif val > large[0]:
            large.remove(val)
            heapify(large)
        elif small[0] == -val:
            heappop(small)
        else:
            small.remove(-val)
            heapify(small)


    def findMedian(self, k) -> float:
        small, large = self.heaps
        return float(large[0]) if k & 1 else (large[0] - small[0])/2.

class Solution(MedianFinder):
    def medianSlidingWindow0(self, nums: List[int], k: int) -> List[float]:
        median = []
        for val in nums[:k]:
            self.addNum(val)
        for i in range(k,len(nums)):
            median.append(self.findMedian(k))
            self.rmNum(nums[i-k])
            self.addNum(nums[i])
        median.append(self.findMedian(k))
        return median

# https://leetcode.com/problems/sliding-window-median/discuss/262689/Python-Small-and-Large-Heaps
# time complexity is O(nlogk) and space complexity is O(logk)
    def move(self, h1, h2):
        x, i = heappop(h1)
        heappush(h2, (-x, i))
    
    def get_med(self, h1, h2, k):
        return h2[0][0] * 1. if k & 1 else (h2[0][0]-h1[0][0]) / 2.

    def medianSlidingWindow1(self, nums, k):
        small, large = [], []
        for i, x in enumerate(nums[:k]): 
            heappush(small, (-x,i))
        for _ in range(k-(k>>1)): # k - k//2
            self.move(small, large) # k = 11, move 6 largest number to heap large
        ans = [self.get_med(small, large, k)]
        for i, x in enumerate(nums[k:]):
            if x >= large[0][0]:
                heappush(large, (x, i+k)) # add new number to large
                if nums[i] <= large[0][0]: 
                    self.move(large, small) # move large_heap_top to small to make up for the nums[i] to be removed from small
            else:
                heappush(small, (-x, i+k)) # add new number to small
                if nums[i] >= large[0][0]: 
                    self.move(small, large) # move small_heap_top to large to make up for the nums[i] to be removed from large
            while small and small[0][1] <= i: # pop (drop) all heaps tops (with index <= i)
                heappop(small)
            while large and large[0][1] <= i: 
                heappop(large)
            ans.append(self.get_med(small, large, k))
        return ans

# O(nlog(n)) Solution using Two Heaps + Lazy Deletion
# Related Problem: Find Median from Data Stream
    def medianSlidingWindow(self, nums: List[int], k: int) -> List[float]:
        maxHeap = []
        minHeap = []
        
        # Step 1: Heap Initialization
        # When there are 4 elements: we have 2 in maxHeap and 2 in minHeap
        # When there are 5 elements: we have 2 in maxHeap and 3 in minHeap
        for i in range(k):
            heappush(maxHeap, (-nums[i], i))
        
        for _ in range(k - k//2):
            num, i = heappop(maxHeap)
            heappush(minHeap, (-num, i))
        
        # Step 2: Sliding Window and Maintain Heap Balance
        ans = [(-maxHeap[0][0]+minHeap[0][0])/2.0] if k%2 == 0 else [minHeap[0][0]/1.0]
        for i in range(k, len(nums)):
            
            # always add the element to minHeap
            num, j = heappushpop(maxHeap, (-nums[i], i))
            heappush(minHeap, (-num, j))
            
            # Now check the inbalance
            if (nums[i-k], i-k) < minHeap[0]: # See if to be removed value < minheap top (i.e., should be in maxheap)
                # IMPORTANT: `if nums[i-k] < minHeap[0][0]` won't work!
                # EXPLANATION:
                # there could be a number valued at nums[i-k] with a smaller than (i-k) index on top of minHeap
                # after the above maxHeap heappushpop + minHeap heappush
                # when that's the case, it means that (nums[i-k],i-k) is ACTUALLY in minHeap!
                
                # number to be removed in maxHeap, inbalance
                # transfer one element to maxHeap
                num, j = heappop(minHeap)
                heappush(maxHeap, (-num, j))
            
            
            while minHeap and minHeap[0][1] <= i-k:
                heappop(minHeap)
            while maxHeap and maxHeap[0][1] <= i-k:
                heappop(maxHeap)
            if k%2 == 0:
                ans.append((-maxHeap[0][0]+minHeap[0][0])/2.0)
            else:
                ans.append((minHeap[0][0])/1.0)            
        return ans

if __name__ == '__main__':
    # medFind = MedianFinder()
    # medFind.addNum(1)
    # medFind.addNum(2)
    # print(medFind.findMedian()) # 1.5
    # medFind.addNum(3)
    # print(medFind.findMedian()) # 2

    # fptr = open("fraudNotify.ut", "r")
    # n, d = list(map(int, fptr.readline().rstrip().split()))

    # expenditures = list(map(int, fptr.readline().rstrip().split()))

    # print(activityNotifications(expenditures, d)) # 633
    # print(activityNotifications([2, 3, 4, 2, 3, 6, 8, 4, 5], 5))
    # print(activityNotifications([10, 20, 30, 40, 50], 3))
    # print(activityNotifications([1, 2, 3, 4, 4], 4))
    # print(activityNotifications([1, 2, 3, 4, 4], 4))
    # print(activityNotifications([2,3,4,2,3,6,8,4,5], 5))
    fptr = open("2heaps.ut", "r")
    a = list(map(int, fptr.readline().rstrip().split(",")))
    k = int(fptr.readline().rstrip())
    sol = Solution()
    print(sol.medianSlidingWindow0(a, k))
