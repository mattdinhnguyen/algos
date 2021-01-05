#!/bin/python3

import math
import os
import random
import re
import sys
from collections import defaultdict, Counter
from heapq import heappush, heappop
from bisect import insort
from typing import List
from itertools import chain, starmap
from operator import mul

# Complete the freqQuery function below.
ADD = 1
DEL = 2
FIND = 3
# Keep counts of freqs of values and freqs of freqs
def freqQuery(queries):
    ans = []
    cache = defaultdict(int)
    freqs = defaultdict(int)
    for cmd,val in queries:
        if cmd == ADD:
            if cache[val]:
                freqs[cache[val]] -= 1
            cache[val] += 1
            freqs[cache[val]] += 1
        elif cmd == DEL and cache[val] > 0:
            freqs[cache[val]] -= 1
            cache[val] -= 1
            if cache[val]:
                freqs[cache[val]] += 1
        elif cmd == FIND:
            ans.append(1 if freqs[val] else 0)
    return ans

class Solution:
    # https://leetcode.com/problems/top-k-frequent-elements/
    def topKFrequent0(self, nums: List[int], k: int) -> List[int]:
        valFreqMap = defaultdict(int)
        freqValSorted = []
        for val in nums:
            valFreqMap[val] += 1
        for val,freq in valFreqMap.items(): # nlogn
            insort(freqValSorted, (-freq,-val))
        return list(map(lambda t: -t[1], freqValSorted[:k]))
    def topKFrequent1(self, nums: List[int], k: int) -> List[int]:
        return [item for item, count in Counter(nums).most_common(k)]
    def topKFrequent2(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        # Use Counter to extract the top k frequent elements
        # most_common(k) return a list of tuples, where the first item of the tuple is the element,
        # and the second item of the tuple is the count
        # Thus, the built-in zip function could be used to extract the first item from the tuples
        return list(zip(*Counter(nums).most_common(k)))[0]
    def topKFrequent3(self, nums: List[int], k: int) -> List[int]:
        b = [[] for _ in range(len(nums)+1)]
        c = Counter(nums)       
        for x in c:
            b[c[x]].append(x)
        return list(chain(*b))[-k:]
    def topKFrequent(self, nums: List[int], k: int) -> List[int]: # best time
        valFreqMap = defaultdict(int)
        freqVals = defaultdict(list)
        for val in nums:
            valFreqMap[val] += 1
        bigger = 0
        for val,freq in valFreqMap.items():
            freqVals[freq].append(val)
            bigger = max(bigger, freq)
        ans = []
        for freq in range(bigger, 0, -1):
            if freq in freqVals:
                ans.extend(freqVals[freq])
                if len(ans) >= k:
                    break
        return ans if len(ans) == k else ans[:k]
    # https://leetcode.com/problems/sort-characters-by-frequency/
    def frequencySort(self, s: str) -> str:
        return "".join(char * times for char, times in Counter(str).most_common())
    def frequencySort(self, s: str) -> str:
        d = Counter(s)
        b = [[] for _ in range(len(s)+1)]
        for val, count in d.items():
            b[count].append(val)
        return ''.join(c*d[c] for c in list(chain(*b))[::-1])
    def frequencySort(self, s: str) -> str:
        chrFreqMap = defaultdict(int)
        freqChrs = defaultdict(list)
        for chr in s: chrFreqMap[chr] += 1
        bigger = 0
        for chr,freq in chrFreqMap.items(): # nlogn
            freqChrs[freq].append(chr)
            bigger = max(bigger, freq)
        return "".join((chr*freq for freq in range(bigger, 0, -1) for chr in freqChrs.get(freq,[])))
    def frequencySort(self, s: str) -> str: # respectable time
        chrFreqMap = defaultdict(int)
        freqChrSorted = []
        for chr in s:
            chrFreqMap[chr] += 1
        for chr,freq in chrFreqMap.items(): # nlogn
            insort(freqChrSorted, (-freq,chr))
        return "".join(map(lambda t: "".join(t[1]*(-t[0])), freqChrSorted))
    frequencySort = lambda _,s: ''.join(c*x for x,c in Counter(s).most_common()) # better
    frequencySort = lambda _,s: ''.join(starmap(mul, Counter(s).most_common())) # best
    def scheduleCourseDP(self, courses: List[List[int]]) -> int:
        courses.sort(key = lambda c: (c[1],c[0]))
        endDate = courses[-1][1]
        clen = len(courses)
        dp = [ [0 for j in range(clen+1)] for i in range(endDate+1)]
        for i in range(1,endDate+1):
            for j in range(1, clen+1):
                dp[i][j] = max(dp[i][j-1], 1+dp[i-courses[j-1][0]][j-1]\
                    if courses[j-1][1] >= i >= courses[j-1][0] else dp[i-1][j])
        return dp[endDate][clen]
    def scheduleCourseGreedy(self, courses: List[List[int]]) -> int: # best time
        courses.sort(key = lambda c: (c[1],c[0]))
        daysTakenByCourses = []
        daySpent = 0
        for dayTaken,closedDate in courses:
            daySpent += dayTaken
            heappush(daysTakenByCourses,-dayTaken)
            if daySpent > closedDate:
                daySpent += heappop(daysTakenByCourses)

        return len(daysTakenByCourses)
if __name__ == '__main__':
    sol = Solution()
    # print(sol.topKFrequent([1,1,1,2,2,3],2))
    # print(sol.frequencySort("tree"))
    tdata = [[[5,5],[4,6],[2,6]],
    [[100,200],[200,1300],[1000,1250],[2000,3200]],
    [[7,16],[2,3],[3,12],[3,14],[10,19],[10,16],[6,8],[6,11],[3,13],[6,16]]]
    for td in tdata:
        print(sol.scheduleCourseGreedy(td),sol.scheduleCourseDP(td)) # 2,3,4
    fptr = open("freqqueries.ut")
    fptro = open("freqqueries.uto")

    q = int(fptr.readline().strip())

    queries = []
    results = []
    findCnt = 0
    for _ in range(q):
        lst = list(map(int, fptr.readline().rstrip().split()))
        queries.append(lst)
        if lst[0] == FIND:
            findCnt += 1
        
    for _ in range(findCnt):
        results.append(int(fptro.readline().strip()))

    # ans = freqQuery(queries)
    # for i,an in enumerate(ans):
    #     if an != results[i]:
    #         print(f"{i}: expected {results[i]} got {an}")
    # print(' '.join(map(str, ans)))
    # print('\n')

    fptr.close()
    fptro.close()
