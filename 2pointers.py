from typing import List, OrderedDict
from collections import deque, defaultdict, Counter
from functools import reduce
from minSubStrHasT import min_length_substring
from bisect import bisect_left
from itertools import combinations
from sorts import partition
from math import factorial, log
import operator as op
from heapq import heapify, heappop, heappush, nlargest
from sorts import maxheap, maxheappop, nlargest as maxheapnlargest
class Solution:
    # https://leetcode.com/problems/squares-of-a-sorted-array/
    # Given an array of integers A sorted in non-decreasing order,
    # return an array of the squares of each number, also in sorted non-decreasing order.
    # We first declare a list of length, len(A) then add the larger square from the back of the list,
    # denoted by the index r - l.

    def sortedSquares0(self, A: List[int]) -> List[int]:
        l, r = 0, len(A)-1
        res = [0]*(r+1)
        while l <= r:
            if abs(A[r]) > abs(A[l]):
                res[r-l] = A[r]**2
                r -= 1
            else:
                res[r-l] = A[l]**2
                l += 1
        return res
    def sortedSquares(self, A): # faster by 15%
        answer = [0] * len(A)
        l, r = 0, len(A) - 1
        while l <= r:
            left, right = abs(A[l]), abs(A[r])
            if left > right:
                answer[r - l] = left * left
                l += 1
            else:
                answer[r - l] = right * right
                r -= 1
        return answer

    def sortedSquares(self, A):
        l, r, answer = 0, len(A) - 1, [0] * len(A)
        while l <= r:
            left, right = abs(A[l]), abs(A[r])
            answer[r - l] = max(left, right) ** 2
            l, r = l + (left > right), r - (left <= right)
        return answer

    def backspaceCompare0(self, S: str, T: str) -> bool:
        def backspaceRemove(s):
            res = deque()
            j = len(s) - 1
            bspaceCnt = 0
            while j >= 0:
                if s[j] == '#':
                    bspaceCnt += 1
                elif bspaceCnt:
                    bspaceCnt -= 1
                else:
                    res.appendleft(s[j])
                j -= 1
            return res
        return backspaceRemove(S) == backspaceRemove(T)
    # https://leetcode.com/problems/backspace-string-compare/discuss/135603/JavaC%2B%2BPython-O(N)-time-and-O(1)-space
    def backspaceCompare(self, S, T):
        def back(res, c):
            if c != '#': res.append(c)
            elif res: res.pop()
            return res
        return reduce(back, S, []) == reduce(back, T, [])
    def backspaceCompare(self, S, T):
        i, j = len(S) - 1, len(T) - 1
        backS = backT = 0
        while True:
            while i >= 0 and (backS or S[i] == '#'):
                backS += 1 if S[i] == '#' else -1
                i -= 1
            while j >= 0 and (backT or T[j] == '#'):
                backT += 1 if T[j] == '#' else -1
                j -= 1
            if not (i >= 0 and j >= 0 and S[i] == T[j]):
                return i == j == -1
            i, j = i - 1, j - 1
    # https://leetcode.com/problems/k-diff-pairs-in-an-array/discuss/100135/JavaPython-Easy-Understood-Solution
    def findPairs(self, nums, k):
        res = 0; c = Counter(nums)
        for i in c:
            if k>0 and i+k in c or k==0 and c[i] > 1: res += 1
        return res
    # https://www.interviewbit.com/problems/diffk-ii/ 
    def diffPossible(self, A, B):
        hash = set()
        for i in A:
            if i - B in hash or i + B in hash: return 1
            hash.add(i)
        return 0
    # substring concat
    def findSubstring(self, A: str, B: List[str]) -> List[int]:
        ans = []; n = len(B[0]); bcount = defaultdict(int)
        for w in B: bcount[w] += 1
        for s in range(n): # start from 0 or ... n-1
            aindices = sorted([[i,A[i:i+n]] for i in range(s,len(A),n)]) # indices of A words
            for i in range(len(B),len(aindices)+1):
                ssCount = defaultdict(int); ssidxs = aindices[i-len(B):i]
                for t in ssidxs: ssCount[t[1]] += 1
                if ssCount == bcount:
                    ans.append(ssidxs[0][0])
        return ans
    # return 1 list of smallest lexicographical indices of 2 pairs of equal sum
    def equal(self, candy, k = 2):
        valCombs = defaultdict(list) # sum value: combs of 2 indices
        def dfs(k, index, path):
            if k == 0:
                valCombs[sum(candy[i] for i in path)].append(path)
                return
            for i in range(index, len(candy)):
                dfs(k-1, i+1, path+[i]) # count down k, i+1: no replacement, path of indices
        dfs(k, 0, []) # k-comb, candy[0], path of k indices
        ans = [] # smallest lexicographical indices of 2 pairs of equal sum
        for val,combs in valCombs.items():
            if len(combs) > 1: # sum value having 2+ combs
                a,b = combs[0]
                for i in range(1,len(combs)):
                    if a not in combs[i] and b not in combs[i]:
                        c = [a,b]+combs[i]
                        ans = c if not ans else min(ans,c)
                        break
        return ans
    def equal(self, A):
        seen = dict(); res = []
        for i in range(len(A)):
            for j in range(i + 1, len(A)): # Ai < Bj
                curr_sum = A[i] + A[j]
                if curr_sum in seen: # Ci>Ai in seen, Bj not in (i,j) 
                    if i > seen[curr_sum][0] and seen[curr_sum][1] not in (i,j):
                        res.append([seen[curr_sum][0],seen[curr_sum][1],i,j])
                else:
                    seen[curr_sum] = (i,j)
        res.sort()
        return res[0]
    def equal(self, A):
        n=len(A)
        for i in range(n):
            for j in range(i+1,n):
                for k in range (1,n):
                    for l in range (k+1,n):
                        if not (i==j or i==k or k==l or j==k or j==l or i==l) and A[i]+A[j]==A[l]+A[k]:
                            return [i,j,k,l]
        return []
    # Given an array of integers nums and an integer target,
    # return indices of the two numbers such that they add up to target.
    # https://leetcode.com/problems/two-sum/
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        valIdxMap = dict()
        for i,val in enumerate(nums):
            if target-val in valIdxMap:
                return [valIdxMap[target-val],i]
            valIdxMap[val] = i
        return []
    # https://leetcode.com/problems/4sum/discuss/8545/Python-140ms-beats-100-and-works-for-N-sum-(Ngreater2)
    def threeSum(self, nums, target):
        results = []; nums.sort()
        for i in range(len(nums)-2): # -2 allows i,l,r
            l = i + 1; r = len(nums) - 1
            t = target - nums[i]
            if i == 0 or nums[i] != nums[i-1]: # starting, skip following dups
                while l < r:
                    s = nums[l] + nums[r]
                    if s == t: # found 1 candidate
                        results.append([nums[i], nums[l], nums[r]])
                        while l < r and nums[l] == nums[l+1]: l += 1 # skip left dups
                        while l < r and nums[r] == nums[r-1]: r -= 1 # skip right dups
                        l += 1; r -=1
                    elif s < t: l += 1
                    else: r -= 1
        return results
    def fourSum(self, nums, target):
        results = []; nums.sort()
        for i in range(len(nums)-3): # -3 allows for threeSum i,l,r
            if i == 0 or nums[i] != nums[i-1]: # starting, skip following dups
                threeResult = self.threeSum(nums[i+1:], target-nums[i]) # for each i find 3sum candidates, starting at i+1
                for item in threeResult:
                    results.append([nums[i]] + item)
        return results
    def fourSum(self, nums, target):
        def findNsum(l, r, target, N, result, results):
            if r-l+1 < N or N < 2 or target < nums[l]*N or target > nums[r]*N:  return # early termination
            if N == 2: # two pointers solve sorted 2-sum problem
                while l < r:
                    s = nums[l] + nums[r]
                    if s == target:
                        results.append(result + [nums[l], nums[r]]) # found 2sum candidate
                        l += 1 # next left
                        while l < r and nums[l] == nums[l-1]: l += 1 # skip left dups
                    elif s < target: l += 1
                    else: r -= 1
            else: # recursively reduce N
                for i in range(l, r+1):
                    if i == l or nums[i-1] != nums[i]: # starting at l, skip following dups
                        findNsum(i+1, r, target-nums[i], N-1, result+[nums[i]], results) # find N-1 sums starting at i+1, pass in prefix candidates
        nums.sort(); results = []
        findNsum(0, len(nums)-1, target, 4, [], results)
        return results
   # https://leetcode.com/problems/minimum-window-substring/submissions/
    def minWindow(self, s: str, t: str) -> str:
        tCnts = defaultdict(int)
        for c in t: tCnts[c] += 1
        tlen = len(t)
        minStart = start = end = 0
        minLen = len(s) + 1
        while end < len(s):
            if tCnts[s[end]] > 0: # needed chr
                tlen -= 1
            tCnts[s[end]] -= 1 # negative for undesired char
            while tlen == 0: # found all t chars in a window
                if end-start+1 < minLen: # update minLen
                    minLen = end-start+1 # begin with start = 0
                    minStart = start
                tCnts[s[start]] += 1
                if tCnts[s[start]] > 0: # found desired char, break while
                    tlen += 1
                start += 1 # add 1 till found critical char, i.e., smaller window
            end += 1 # add 1 till found replacing desired chars next window
        return "" if minLen == len(s) + 1 else s[minStart:minStart+minLen]
    # https://leetcode.com/problems/minimum-window-substring/discuss/26804/12-lines-Python
    def minWindow(self, s, t):
        need = Counter(t)            #hash table to store char frequency
        missing = len(t)                         #total number of chars we care
        start, end = 0, 0
        i = 0
        for j, char in enumerate(s, 1):          #index j from 1
            if need[char] > 0:
                missing -= 1
            need[char] -= 1
            if missing == 0:                     #match all chars
                while i < j and need[s[i]] < 0:  #remove chars to find the real start
                    need[s[i]] += 1
                    i += 1
                need[s[i]] += 1                  #make sure the first appearing char satisfies need[char]>0
                missing += 1                     #we missed this first char, so add missing by 1
                if end == 0 or j-i < end-start:  #update window
                    start, end = i, j
                i += 1                           #update i to start+1 for next window
        return s[start:end]
    #
    def minWindow(self, s, t):
        need, missing = Counter(t), len(t)
        i = I = J = 0
        for j, c in enumerate(s, 1):
            missing -= need[c] > 0
            need[c] -= 1
            if not missing:
                while need[s[i]] < 0: need[s[i]] += 1; i += 1
                if not J or j - i <= J - I: I, J = i, j
                need[s[i]] += 1; i += 1; missing += 1       # SPEEEEEEEED UP!
        return s[I : J]
    # 
    def minWindow(self, s: str, t: str) -> str:
        need, missing = Counter(t), len(t)
        i, I, J = 0, 0, 0
        for j, c in enumerate(s, 1):
            missing -= need[c] > 0
            need[c] -= 1
            if (not J or s[i] == c) and not missing:  # SPEED UP
                while i < j and need[s[i]] < 0:
                    need[s[i]] += 1
                    i += 1
                if not J or j - i < J - I:
                    I, J = i, j
        return s[I:J]
    # Longest Substring with At Most Two Distinct Characters
    def lengthOfLongestSubstringTwoDistinct(self, s: str) -> (int, str):
        sCnts = defaultdict(int)
        counter = begin = end = d = start = 0 
        for end, chr in enumerate(s):
            if sCnts[chr] == 0:
                counter += 1 # count distinct chars
            sCnts[chr] += 1
            while counter > 2: # seen 3rd distinct chars
                if sCnts[s[begin]] == 1:
                    counter -= 1
                sCnts[s[begin]] -= 1
                begin += 1
            if end-begin+1 > d:
                start = begin
                d = end-begin+1  # while valid, update d, add 1 to include char at end
        return d,s[start:start+d]
    # Longest Substring Without Repeating Characters
    def lengthOfLongestSubstring(self, s: str) -> (int, str):
        sCnts = defaultdict(int)
        counter = begin = d = start = 0
        for end, chr in enumerate(s):
            if sCnts[chr] > 0: counter += 1 # seen 2nd of s[end] char
            sCnts[chr] += 1
            while counter > 0:
                if sCnts[s[begin]] > 1: counter -= 1
                sCnts[s[begin]] -= 1; begin += 1 # move right 1 char, to drop the repeating char at begin
            if end-begin+1 > d:
                start = begin; d = end-begin+1 # while valid, update d, add 1 to include char at end
        return d,s[start:start+d]
    def threeSumClosest(self, nums: List[int], target: int) -> int: # return 3sum closest to target
        nums.sort()
        result = nums[0] + nums[1] + nums[2] # init
        for i in range(len(nums)-2): # iterate 0..nums[-3]
            if i > 0 and nums[i] == nums[i-1]: continue # update: ignore the duplicate numbers
            l, r = i + 1, len(nums) - 1 # 2 pointers: i+1 and far right
            while l < r:
                curSum = nums[l] + nums[r] + nums[i]
                if curSum == target: return target
                if abs(curSum-target) < abs(result-target): result = curSum # update closest 3sum
                if curSum < target: l += 1 # move l right
                else: r -= 1 # move r left
        return result
    def dNums(self, A, B): # distint nums in sliding B windows over A
        if B > len(A): return []
        if B == 1: return [1]*len(A)
        numCount = defaultdict(int)
        for i in range(B): numCount[A[i]] += 1
        dnums = [sum(v>0 for v in numCount.values())]; i = B
        while i < len(A):
            left, right = A[i-B], A[i]
            if left == right: dnums.append(dnums[-1])
            else:
                numCount[left] -= 1; numCount[right] += 1
                dnums.append(dnums[-1] - (numCount[left] == 0) +  (numCount[right] == 1))
            i += 1
        return dnums
    # Partion A[l,r+1] at ind position
    def part(self,A, l, r, ind):
        if l>r: return
        pivot = partition(A,l,r)
        if pivot < ind: self.part(A,pivot+1,r,ind) # move left to pivot+1
        elif pivot > ind: self.part(A,l,pivot-1,ind) # move right to pivot-1
        # elif pivot == ind: return
    def KsLargest(self, A, k):
        n = len(A); self.part(A,0,n-1,n-k)
        return A[n-k:]
    def KthLargest(self, A, k): # A unsorted, https://leetcode.com/problems/kth-largest-element-in-an-array/
        n = len(A); self.part(A,0,n-1,n-k) # n-k position
        return A[n-k]
    # https://www.interviewbit.com/problems/ways-to-form-max-heap/
    def waystoformmaxheap(self, A: int):
        def comb(r,n) :
            return factorial(n)//factorial(r)//factorial(n-r)
            # if 2*r > n : return comb(n-r,n)
            # c = 1
            # for i in range(r) : c = c*(n-i)//(i+1)
            # return c
        ans,h = [1,1], 0
        for n in range(2,A+1) :
            if 2<<h <= n : h += 1
            m = n-(1<<h)+1
            l = (1<<(h-1))-1 + min(m,1<<(h-1))
            r = (1<<(h-1))-1 + max(0,m-(1<<(h-1)))
            ans.append((comb(l,n-1)*ans[l]*ans[r])%(10**9+7))
        return ans[A]
    def nCombsr(self, n, r):
        r = min(r, n - r)
        if r == 0: return 1
        numer = reduce(op.mul, range(n, n - r, -1))
        denom = reduce(op.mul, range(1, r + 1))
        return numer // denom
    def ways2formmaxheap(self, n):
        def comb(r,n) :
            r = min(r, n - r)
            if r == 0: return 1
            return factorial(n)//factorial(r)//factorial(n-r)
        if n == 0 or n == 1: return 1
        h = int(log(n, 2)) # heap height
        m = pow(2, h) # max number of elements in h level, root is level 0
        p = n - (m - 1) # number of elements in h level in heap of size n
        if p >= m // 2: L = m - 1 # number of values in the left subtree of root
        else: L = m - 1 - (m // 2 - p)
        R = n - 1 - L
        return self.nCombsr(n - 1, L) * self.ways2formmaxheap(L) * self.ways2formmaxheap(R)
    def waystoformmaxheap0(self, A):
        return self.ways2formmaxheap(A) % 1000000007
    # https://www.geeksforgeeks.org/k-maximum-sum-combinations-two-arrays/
    def kMaxSumCombs(self,A,B,k):
        arr = [sorted(A), sorted(B)]
        maxTuple = [None]*2; ans = []; maxIdx = 0; maxVal = float('-inf')
        while k > 0:
            for i in range(len(arr)):
                if maxTuple[i] == None:
                    maxTuple[i] = arr[i].pop()
                if maxTuple[i] > maxVal:
                    maxIdx = i; maxVal = maxTuple[i]
            ans.append(sum(maxTuple))
            maxTuple[maxIdx^1] = None; k -= 1
        return ans
    def kMaxSumCombs(self,A,B,k):
        A = sorted(A, reverse=True); B = sorted(B, reverse=True)
        ans = []; hq = [(-(A[0]+B[0]), 0, 0)]; seen = set([(0,0)])
        while len(ans) < k:
            s, i, j = heappop(hq)
            ans.append(-s)
            if j+1 < len(B) and (i,j+1) not in seen:
                heappush(hq, (-(A[i]+B[j+1]), i, j+1)); seen.add((i,j+1))
            if i+1 < len(A) and (i+1,j) not in seen:
                heappush(hq, (-(A[i+1]+B[j]), i+1, j)); seen.add((i+1,j))
            if i+1 <len(A) and j+1 <len(B) and (i,j) not in seen:
                j += 1; i += 1; heappush(hq, (-(A[i]+B[j]), i, j)); seen.add((i,j))
        return ans

# python3 implementation, see https://leetcode.com/problems/lru-cache/submissions/ for python2
class LRUCache (OrderedDict):
    def __init__(self, capacity: int):
        self.maxsize = capacity

    def get(self, key: int) -> int: # O(1)?
        try:
            value = super().__getitem__(key)
            self.move_to_end(key)
            return value
        except KeyError:
            return -1

    def put(self, key: int, value: int) -> None:
        if key in self: self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]
if __name__ == '__main__':
    sol = Solution()
    assert sol.kMaxSumCombs([59, 63, 65, 6, 46, 82, 28, 62, 92, 96, 43, 28, 37, 92, 5, 3, 54, 93, 83],[59, 63, 65, 6, 46, 82, 28, 62, 92, 96, 43, 28, 37, 92, 5, 3, 54, 93, 83], 10) == [192,189,189,188,188,188,188,186,185,185]
    assert sol.kMaxSumCombs([3,2],[1,4], 2) == [7,6]
    assert sol.kMaxSumCombs([4, 2, 5, 1],[8, 0, 5, 3], 3) == [13,12,10]
    # assert sol.waystoformmaxheap(10000000) == 581401370 takes too long in python3, need C++ or Rust
    # lru_cache = LRUCache(2)
    # tdata = zip(["put", "put", "get", "put", "get", "put", "get", "get", "get"],[[1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]])
    # ops = {"put": lru_cache.put, "get": lru_cache.get}
    # expected = [None, None, 1, None, -1, None, -1, 3, 4]
    # for i,(cmd,params) in enumerate(tdata):
    #     assert ops[cmd](*params) == expected[i]
    # assert sol.KthLargest([3,2,1,5,6,4], 2) == 5
    # assert sol.KsLargest([3,2,1,5,6,4], 2) == [5,6]
    # assert sol.diffPossible([1, 5, 3],2) == 1
    # tdata = [ 66, 37, 46, 56, 49, 65, 62, 21, 7, 70, 13, 71, 93, 26, 18, 84, 96, 65, 92, 69, 97, 47, 6, 18, 17, 47, 28, 71, 70, 24, 46, 58, 71, 21, 30, 44, 78, 31, 45, 65, 16, 3, 22, 54, 51, 68, 19, 86, 44, 99, 53, 24, 40, 92, 38, 81, 4, 96, 1, 13, 45, 76, 77, 8, 88, 50, 89, 38, 60, 61, 49, 25, 10, 80, 49, 63, 95, 74, 29, 27, 52, 27, 40, 66, 38, 22, 85, 22, 91, 98, 19, 20, 78, 77, 48, 63, 27 ]
    # assert sol.diffPossible(tdata,31) == 1
    # assert sol.diffPossible([ 1,3,2 ],0) == 0
    # assert sol.dNums([1, 2, 1, 3, 4, 3], 3) == [2, 3, 3, 2]
    assert sol.dNums([1, 1, 2, 2], 1) == [1,1,1,1]
    # assert sol.dNums([1, 2, 1, 3, 4, 2, 3], 4) == [3, 4, 4, 3]
    # assert sol.equal([ 1, 1, 1, 1, 1 ]) == [0, 1, 2, 3]
    # assert sol.equal([3, 4, 7, 1, 2, 9, 8]) == [0, 2, 3, 5]
    # assert sol.findSubstring("lingmindraboofooowingdingbarrwingmonkeypoundcake",["fooo","barr","wing","ding","wing"]) == [13]
    # assert sol.findSubstring("wordgoodgoodgoodbestword",["word","good","best","good"]) == [8]
    # assert sol.findSubstring("barfoofoobarthefoobarman", ["foo","bar","the"]) == [6,9,12]
    # assert sol.findSubstring("wordgoodgoodgoodbestword", ["word","good","best","word"]) == []
    # assert sol.findSubstring("barfoothefoobarman", ["foo","bar"]) == [0,9]
    a = sol.fourSum([-5,-4,-3,-2,-1,0,0,1,2,3,4,5],0)
    expected = [[-5,-4,4,5],[-5,-3,3,5],[-5,-2,2,5],[-5,-2,3,4],[-5,-1,1,5],[-5,-1,2,4],[-5,0,0,5],[-5,0,1,4],[-5,0,2,3],[-4,-3,2,5],[-4,-3,3,4],[-4,-2,1,5],[-4,-2,2,4],[-4,-1,0,5],[-4,-1,1,4],[-4,-1,2,3],[-4,0,0,4],[-4,0,1,3],[-3,-2,0,5],[-3,-2,1,4],[-3,-2,2,3],[-3,-1,0,4],[-3,-1,1,3],[-3,0,0,3],[-3,0,1,2],[-2,-1,0,3],[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]
    assert a == expected
    st = ["ADOBECODEBANC","ABC"]
    # assert sol.minWindow(*st) == min_length_substring(*st)
    # assert sol.threeSumClosest([-1,2,1,-4],1) == 2
    # print(sol.lengthOfLongestSubstring("AABCCCEFG"))
    # print(sol.lengthOfLongestSubstringTwoDistinct("AABBCCCEFG"))
    # print(sol.lengthOfLongestSubstringTwoDistinct("AABCC"))
    # tdata = [[-4,-1,0,3,10],[-7,-3,2,3,11]]
    # tdata = [["ab##","c#d#"],["a##c", "#a#c"],["a#c", "b"]]
    # for td in tdata:
    #     # print(sol.sortedSquares0(td))
    #     print(sol.backspaceCompare(*td))
