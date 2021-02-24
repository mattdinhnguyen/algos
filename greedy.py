# from functools import reduce, lru_cache, cache
from heapq import nlargest, nsmallest, heapify, heappop, heappush
import unittest
from typing import List
from timeit import timeit
from numpy import product
import random
import sys
from bisect import bisect_left, bisect, insort

# How to create greedy algos https://www.interviewbit.com/courses/programming/topics/greedy-algorithm/
class Solution:
    def maximiseTasks(self, A: List[int], T: int) -> int:
        ct = maxTasks = i = 0; A.sort()
        while i < len(A) and ct <  T:
            if ct + A[i] <=T:
                ct += A[i]; maxTasks += 1
            i += 1
        return maxTasks
    def fractionalKnapsack(self, weights: List[int], values: List[int], capacity: int) -> int:
        maxVal = i = 0; unitVals = sorted([(v//w, w, v) for v,w in zip(values, weights)], reverse=True)
        while capacity:
            u, w, v = unitVals[i]
            if capacity >= w:
                maxVal += v; capacity -= w
            else:
                maxVal += capacity*u; break
            i += 1
        return maxVal
    # https://leetcode.com/problems/maximum-product-of-three-numbers/discuss/104739/Python-O(N)-and-1-line l1ne
    # First k nums are heapified klogk, then n-k nums are heapreplace (n-k)logk: total nlogk     
    def maximumProduct(self, nums: List[int]) -> int:
        a, b = nlargest(3, nums), nsmallest(2, nums)
        return max(a[0] * a[1] * a[2], b[0] * b[1] * a[0])
    def maximumProduct(self, nums: List[int]) -> int:
        a, b = nlargest(3, nums), nsmallest(2, nums)
        return max(product(a), product(b + a[:1]))
    def canAttendMeetings(self, intervals):
        if intervals:
            heapify(intervals); _,end = heappop(intervals)
            while intervals:
                if intervals[0][0] < end: return False
                _,end = heappop(intervals)
        return True
    def minMeetingRooms(self, intervals):
        if not intervals: return 0
        heapify(intervals); rooms = [heappop(intervals)[1]]; l = len(intervals)
        while intervals:
            start,end = heappop(intervals)
            i = bisect(rooms, start)
            if i >= l:
                rooms[-1] = end
            elif i == 0: insort(rooms, end)
            else: # start >= rooms[i-1]
                del rooms[i-1]
                insort(rooms, end, lo=i-1)
        return len(rooms)
    def candy(self, a: List[int]) -> int:
        n = len(a); c = [1]*n; start = end = -1
        for i in range(1, n):
            if a[i] < a[i-1]:
                if start == -1: start = i-1 # descending interval
                end = i
                if not i == n-1: continue
            if a[i] > a[i-1]: c[i] = c[i-1] + 1 # climb up
            if start != -1:
                extra = 0
                for idx in range(end,start-1,-1): # climb back up from bottom
                    c[idx] = max(c[idx], extra + 1)
                    extra += 1
                start = end = -1
        return sum(c)
    def mice(self, mice: List[int], hole: List[int]) -> int:
        mice.sort(); hole.sort(); t = 0
        for m,h in zip(mice, hole):
            t = max(t, abs(m-h))
        return t
    def majorityElement(self, num):
        majorityIndex = 0; count = 1
        for i in range(1, len(num)):
            count += 1 if num[majorityIndex] == num[i] else -1
            if count == 0:
                majorityIndex = i; count = 1
        return num[majorityIndex]
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        tcost = curcost = sum(cost); l = len(cost); tank = start = i = 0
        if tcost > sum(gas): return -1
        while curcost:
            g, c = gas[i], cost[i]
            if g >= c or tank+g >= c:
                tank += g-c; curcost -= c
                if curcost <= 0:
                    break
            elif tank+g < c:
                start = i+1; curcost = tcost; tank = 0
            i = (i+1)%l
        return start
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        sumGas = sumCost = start = tank = 0
        for i in range(len(gas)):
            sumGas += gas[i]; sumCost += cost[i]
            tank += gas[i] - cost[i]
            if tank < 0:
                start = i + 1; tank = 0 # set start when tank <0
        return -1 if sumGas < sumCost else start # sumGas, sumCost are total from all
    def seats(self, A: str) -> int:
        p = [] # occupied seat indices
        for i,c in enumerate(A):
            if c == 'x': p.append(i)
        if len(p) < 2: return 0
        if len(p) == 2: return p[1]-p[0]-1
        l = r = len(p)//2
        if len(p)%2 == 0: l -= 1
        li, ri = p[l], p[r] # medium left val, right val
        moves = 0
        while ri-li>1: moves = ri-li-1; li += moves; p[l] = li
        while l >= 0:
            if p[l]<li:
                moves += li-p[l]-1
                li -= 1
            l -= 1
        while r < len(p):
            if p[r]>ri:
                moves += p[r]-ri-1
                ri = ri+1
            r += 1
        return moves%10000003
    # Merge an interval into a sorted non-overlapping list, maintaining non-overlapping sorted list
    # https://www.careercup.com/question?id=13014685
    # https://www.glassdoor.com/Interview/Recently-I-attended-the-interview-at-Google-and-I-was-asked-You-are-given-a-sorted-list-of-disjoint-intervals-and-an-inter-QTN_345488.htm
    def insertMergeIntervals(self, A: List[List[int]], B: List[int]) -> List[List[int]]:
        i = bisect(A, B)
        ans = A[:i]
        _, hi = ans[-1]
        if hi >= B[0]:
            if hi < B[1]: ans[-1][1] = B[1]
        else:
            ans.append(B)
        hi = ans[-1][1]
        for i in range(i,len(A)):
            if hi >= A[i][0]:
                if hi <= A[i][1]:
                    ans[-1][1] = A[i][1]
            else:
                ans += A[i:]
                break
        return ans
    # https://www.interviewbit.com/problems/disjoint-intervals/
    # Return length of maximal set of mutually disjoint intervals.
    def maxDisjointIntervals(self, A: List[List[int]]) -> int:
        A.sort(key = lambda x: x[1])
        finalList = A[:1]
        for interval in A[1:]:
            if interval[0] > finalList[-1][1]:
                finalList.append(interval)
        return len(finalList)
sol = Solution()
class TestSolution(unittest.TestCase):
    def test_none(self):
        self.assertTrue(sol.maxDisjointIntervals([[1, 9],[2, 3],[5, 7]]) == 2)
        self.assertTrue(sol.maxDisjointIntervals([[1, 4],[2, 3],[4, 6],[8, 9]]) == 3)
        self.assertTrue(sol.insertMergeIntervals([[1,5],[10,15],[20,25]],[12,27]) == [[1,5],[10,27]])
        self.assertTrue(sol.insertMergeIntervals([[1,5],[10,15],[20,25],[27,32]],[12,21]) == [[1,5],[10,25],[27,32]])
        self.assertTrue(sol.seats('....x..xx...x..') == 5)
        self.assertTrue(sol.seats('xx.....xx.x..xxxx..xxxx.xx..xx..x.xxxx') == 79)
        self.assertTrue(sol.seats('.x.x.x..x') == 5)
        self.assertTrue(sol.seats('....') == 0)
        self.assertTrue(sol.seats('............x.x') == 1)
        self.assertTrue(sol.seats('.x..x..x.') == 4)
        A = [204,918,18,17,35,739,913,14,76,555,333,535,653,667,52,987,422,553,599,765,494,298,16,285,272,485,989,627,422,399,258,959,475,983,535,699,663,152,606,406,173,671,559,594,531,824,898,884,491,193,315,652,799,979,890,916,331,77,650,996,367,86,767,542,858,796,264,64,513,955,669,694,382,711,710,962,854,784,299,606,655,517,376,764,998,244,896,725,218,663,965,660,803,881,482,505,336,279 ]
        B = [273,790,131,367,914,140,727,41,628,594,725,289,205,496,290,743,363,412,644,232,173,8,787,673,798,938,510,832,495,866,628,184,654,296,734,587,142,350,870,583,825,511,184,770,173,486,41,681,82,532,570,71,934,56,524,432,307,796,622,640,705,498,109,519,616,875,895,244,688,283,49,946,313,717,819,427,845,514,809,422,233,753,176,35,76,968,836,876,551,398,12,151,910,606,932,580,795,187 ]
        self.assertTrue(sol.canCompleteCircuit(A,B) == 31)
        self.assertTrue(sol.canCompleteCircuit([2,0,1,2,3,4,0],[0,1,0,0,0,0,11]) == 0)
        self.assertTrue(sol.canCompleteCircuit([5,1,2,3,4],[4,4,1,5,1]) == 4)
        self.assertTrue(sol.canCompleteCircuit([2,3,4],[3,4,3]) == -1)
        self.assertTrue(sol.canCompleteCircuit([1,2,3,4,5],[3,4,5,1,2]) == 3)
        self.assertTrue(sol.mice([4,-4,2],[4,0,5]) == 4)
        self.assertTrue(sol.candy([1,2]) == 3)
        self.assertTrue(sol.candy([1,5,2,1]) == 7)
        self.assertTrue(sol.minMeetingRooms([[7,10],[4,19],[19,26],[14,16],[13,18],[16,21]]) == 3)
        self.assertTrue(sol.minMeetingRooms([[1,18],[18,23],[15,29],[4,15],[2,11],[5,13]]) == 4)
        self.assertTrue(sol.minMeetingRooms([(0,30),(5,10),(15,20)]) == 2)
        self.assertTrue(sol.minMeetingRooms([(5,8),(9,15)]) == 1)
        self.assertTrue(sol.minMeetingRooms([(2,7)]) == 1)
        self.assertFalse(sol.canAttendMeetings([(0,30),(5,10),(15,20)]))
        self.assertTrue(sol.canAttendMeetings([(5,8),(9,15)]))
        self.assertTrue(sol.maximumProduct([-100,-98,-1,2,3,4]) == 39200)
        self.assertTrue(sol.maximumProduct([1,2,3,4]) == 24)
        self.assertTrue(sol.maximiseTasks([4,2,1,2,5],8) == 3)
        self.assertTrue(sol.fractionalKnapsack([10,20,30],[60,100,120],50) == 240)

if __name__ == "__main__":
    # data = random.sample(range(-1000, 1000), 1000)
    # print(timeit(
    #     f"((Solution()).maximumProduct({data}))",
    #     number=100000,
    #     setup="from __main__ import Solution"
    # ))
    unittest.main()
