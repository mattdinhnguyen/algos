from timeit import timeit
from bisect import bisect, bisect_left
import sys
from typing import List
from collections import Counter
from itertools import groupby
# https://www.geeksforgeeks.org/largest-sum-contiguous-subarray/?ref=leftbar-rightbar
# @timeit
def maxSubArrSum(a): #N
    max_so_far = -sys.maxsize
    max_ending_here = 0
    for i,v in enumerate(a):
        max_ending_here += v
        max_so_far = max(max_ending_here,max_so_far)
        max_ending_here = max(max_ending_here,0)
    return max_so_far

# https://www.geeksforgeeks.org/sliding-window-maximum-maximum-of-all-subarrays-of-size-k/?ref=lbp
# @timeit
def maxAllSubArraySizeK(a, k): #kN
    maxOfAllSubArrs = -sys.maxsize
    for i in range(k,len(a)+1):
        maxOfAllSubArrs = max(maxOfAllSubArrs, max(a[i-k:i]))
    return maxOfAllSubArrs

# https://www.hackerrank.com/challenges/maxsubarray/problem
# @timeit
def max_subarraysubset(numbers):
    """Find the largest sum of any contiguous subarray (kadane) and any subset."""
    best_subsetsum = best_sum = float('-inf')
    current_sum = 0
    for x in numbers:
        current_sum = current_sum + x
        best_sum = max(best_sum, current_sum)
        current_sum = max(0, current_sum)
        best_subsetsum = max(x, best_subsetsum, best_subsetsum + x)
    return [best_sum,best_subsetsum]
# https://leetcode.com/problems/longest-substring-without-repeating-characters/
class Solution:
    def lengthOfLongestSubstring0(self, s: str) -> int:
        ordA = ord(' ')
        preStrCnt = longestStrCnt = 0
        startIdx = 0
        longestIndices = tuple([-1,0])
        lastIndices = tuple([-1,0])
        aIdx = [-1]*100
        for i,chr in enumerate(s):
            chrIdx = ord(chr) - ordA
            if aIdx[chrIdx] != -1:
                if i-startIdx > preStrCnt:
                    preStrCnt = i-startIdx
                    lastIndices = (startIdx,i)
                startIdx = max(startIdx,aIdx[chrIdx]+1)
                distance = i - startIdx
                if distance > longestStrCnt:
                    longestStrCnt = distance
                    longestIndices = (startIdx,i)
            aIdx[chrIdx] = i
        longestStrCnt = len(s)-startIdx
        if longestStrCnt < preStrCnt:
            longestStrCnt = preStrCnt
            longestIndices = lastIndices
        else:
            longestIndices = (startIdx,len(s))
        return (longestStrCnt,s[longestIndices[0]:longestIndices[1]]) if len(s)>1 else (len(s),s)

    def lengthOfLongestSubstring(self, s: str) -> int:
        ordSpace = ord(' ')
        result = 0
        cache = [-1]*100
        j = 0
        for i,chr in enumerate(s):
            chrIdx = ord(chr) - ordSpace
            j = max(j, cache[chrIdx]) if cache[chrIdx] > 0 else j
            cache[chrIdx] = i + 1
            result = max(result, i - j + 1)
        return result if len(s)>1 else len(s)
    # https://leetcode.com/problems/3sum/discuss/7384/My-Python-solution-based-on-2-sum-200-ms-beat-93.37
    # The task is to have three elements add up to zero. d[-v-x]=1is a clever way to have quick look-up for a target.
    # res.add((v, -v-x, x)) makes this explicit. For each iteratation through the list of numbers,
    # we have a v and an x. what we are looking for is an element that reduces those two numbers to zero.
    # Their negatation is the exact number. When d[-v-x]=1 is found later, x will be a different number
    # but because the dictionary contains the target we know there was a previous x needed for that target
    # This algorithm works without the sort, except that duplicated lists with element in a different order can occur.
    # I think, while True, the "explanation negative of a negative is a postive" is a confusing red herring.... 'based on two sum'.
    # We know v and x what number can we add to that to make zero.
    # Add that number to a hash map and later if we encounter it we have a solution.
    def threeSum(self, nums):
        if len(nums) < 3:
            return []
        # idx0 = bisect(nums,0) failed test [1,1,-2]
        nums.sort()
        res = set()
        for i, v in enumerate(nums[:-2]):
            if i >= 1 and v == nums[i-1]:
                continue
            d = {}
            for x in nums[i+1:]:
                if x not in d:
                    d[-v-x] = 1
                else:
                    res.add((v, -v-x, x))
        return [*map(list, res)]
    # https://leetcode.com/problems/3sum/discuss/7498/Python-solution-with-detailed-explanation
    def threeSum(self, nums):
        nums.sort()
        result = []
        n = len(nums)
        idx0 = bisect(nums,0)
        for i in range(idx0):
            # if nums[i]>0: break
            if i > 0 and nums[i] == nums[i-1]:
                continue
            head = i + 1
            end = n - 1
            while head < end:
                if nums[i] + nums[head] + nums[end] == 0:
                    result.append([nums[i], nums[head], nums[end]])
                    head += 1
                    while head < end and nums[head] == nums[head-1]:
                        head += 1
                    end -= 1
                    while head < end and nums[end] == nums[end+1]:
                        end -= 1
                elif nums[i] + nums[head] + nums[end] < 0:
                    head += 1
                    while head < end and nums[head] == nums[head-1]:
                        head += 1
                else:
                    end -= 1
                    while head < end and nums[end] == nums[end+1]:
                        end -= 1
        return result
    def threeSum(self, nums):
        if len(nums) < 3: return []
        res = []
        nums.sort()
        idx0 = bisect(nums,0)
        for i in range(idx0): # iterate up to 0 index
            if i > 0 and nums[i] == nums[i-1]: continue # skip duplicates
            l, r = i+1, len(nums)-1 # 2 pointers
            while l < r:
                s = nums[i] + nums[l] + nums[r]
                if s < 0: l +=1 # move right
                elif s > 0: r -= 1 # move left
                else: # found 1 triplet
                    res.append((nums[i], nums[l], nums[r]))
                    while l < r and nums[l] == nums[l+1]: l += 1 # skip duplicates
                    while l < r and nums[r] == nums[r-1]: r -= 1
                    l += 1; r -= 1 # find next triplet
        return res
    def diffPossible(self, A, B):
        if not A or len(A) == 1: return 0
        i = 0; j = len(A)-1
        while i < j:
            if A[j] - A[i] < B:
                if A[j] - A[0] > B:
                    while i<j and A[j] == A[j-1]: j -= 1
                    i = 0; j -= 1
                else:
                    return 0
            elif A[j] - A[i] > B:
                while i+1 < j and A[i] == A[i+1]:
                    if not B: return 1
                    i += 1 # skip duplicates
                i += 1
                if i == j and j > 0 and A[j] - A[0] > B: j -= 1; i = 0
            else:
                print(A[i],B,A[j])
                return 1
        return 0
    def diffPossible(self, A, B):
        # A.sort()
        i = j = 0
        while i < len(A) and j < len(A):
            k = A[j] - A[i]
            if k == B and i != j: return 1
            elif k < B: j += 1
            else: i += 1
        return 0
    def diffPossible(self, A, B):
        for i,n in enumerate(A):
            m = B+n
            j = bisect_left(A, m, lo=i+1)
            if j < len(A) and A[j] == m:
                print(n,B,m)
                return 1
        return 0
    def merge(self, A, B):
        i = j = 0
        while i < len(A) and j < len(B):
            if A[i] > B[j]:
                A.insert(i, B[j])
                j += 1
            i += 1
        if j < len(B):
            A.extend(B[j:])
        return A
    def intersect(self, A, B):
        if not A or not B: return []
        if len(A) < len(B):
            A, B = B, A
        A.sort()
        B.sort()
        ans = []
        i = bisect_left(A, B[0]) # A start
        if i < len(A):
            j = bisect(A, B[-1], lo=i) # A end
            k = 0 # B index
            while i < j and k < len(B):
                if A[i] < B[k]: i += 1
                elif A[i] > B[k]: k += 1
                else: ans.append(A[i]); i += 1; k += 1
        return ans
    def removeDuplicates(self, A):
        id = 1
        for i in range(1,len(A)):
            if A[i] != A[i-1]: A[id] = A[i]; id += 1
        # del A[id:]
        return id
    def removeDuplicatesII(nums):
        if len(nums) < 3: return len(nums)
        idx = 2 # slow pointer
        for i in range(2, len(nums)): # fast pointer
            if nums[i] != nums[idx-2]:
                nums[idx] = nums[i]; idx += 1
        return idx
    def removeDuplicatesII(self, nums):
        slow = fast = 2
        while fast < len(nums):
            if nums[slow - 2] != nums[fast]:
                nums[slow] = nums[fast]
                slow += 1
            fast += 1
        return slow
    def removeDuplicatesK(self, nums, k):
        i = 0
        for n in nums:
            if i < k or n > nums[i-k]:
                nums[i] = n; i += 1
        return i
    def removeElement(self, A, B): # 2 pointers
        j = 0 # slow
        for n in A: # fast
            if n == B: continue
            A[j] = n; j += 1
        return j
    def sortColors(self, A):
        if len(A) < 2: return A
        def _sort(color, i, k):
            while i <= k:
                if A[i] == color: i += 1
                elif A[k] != color: k -= 1
                elif A[i] != color:
                    A[i], A[k] = A[k], A[i]; i += 1; k -= 1
            return i
        i = _sort(0, 0, len(A)-1)
        if i+1 < len(A):
            j = _sort(1, i, len(A)-1)
        return A
    def sortColors(self, nums): # https://en.wikipedia.org/wiki/Dutch_national_flag_problem
        red, white, blue = 0, 0, len(nums)-1
        while white <= blue:
            if nums[white] == 0:
                nums[red], nums[white] = nums[white], nums[red]
                white += 1; red += 1
            elif nums[white] == 1: white += 1
            else:
                nums[white], nums[blue] = nums[blue], nums[white]
                blue -= 1
        return nums
    def findMaxConsecutiveOnes(self, nums):
        # return max([len(list(group)) if key == 1 else 0 for key, group in groupby(nums)])
        return max([len(list(group)) for key, group in groupby(nums) if key == 1])
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        maxN = nums[0]
        for i in range(1, len(nums)):
            if nums[i] != 0: nums[i] += nums[i-1]
            elif nums[i-1] > maxN: maxN = nums[i-1]
        return max(maxN, nums[-1])
    def findMaxConsecutiveOnes(self, nums):
        oneMax = prev1Max = cur1Max = 0
        for i,num in enumerate(nums):
            if num: cur1Max += 1
            elif i+1 < len(nums) and nums[i+1]: #[0,1]
                if prev1Max: oneMax = max(oneMax, prev1Max+cur1Max)
                prev1Max, cur1Max = cur1Max+1, 0
            elif i+1 < len(nums): #[0,0]
                oneMax = max(oneMax, prev1Max+cur1Max)
                prev1Max = cur1Max = 0
            else:
                oneMax = max(oneMax, prev1Max+cur1Max)
        return oneMax
    # sliding window of 1's (including flipped 0's) to find largest width
    def maxone(self, A, B): # return the indices of maximum continuous series of 1s in order
        start = end = wL = wR = nZero = 0; bestWindowWidth = -1 # look for best window width
        while wR < len(A):
            if nZero <= B: # track number of 0's flipped as growing to window right
                if A[wR] == 0: nZero += 1
                wR += 1
            if nZero > B: # shrink from window left, update '0' count
                if (A[wL] == 0): nZero -= 1
                wL += 1
            if wR - wL + 1 > bestWindowWidth: # update best window as sliding 1's window right
                bestWindowWidth = wR - wL + 1
                start = wL; end = wR # start/end points to best window width
        return [i for i in range(start,end)]
    def maxone(self, A, K):
        l = i = count = 0; ans = [0,0]
        for i in range(len(A)): # grow right index
            if A[i] == 0: count += 1 # incr 0's count
            while count > K: # shrink left index
                if A[l] == 0: count -= 1 # decr 0's count
                l += 1
            if i-l > ans[1] - ans[0]:
                ans = [l,i]
        return list(range(ans[0],ans[1]+1)) # window indices    # Find the longest subarray with at most K zeros.
    def maxone(self, A, K):
        i = j = 0 # left/right window indices
        ans = [0,0]
        for j in range(len(A)): # each iteration grows j right window index
            K -= 1 - A[j] # decr K if A[j] is 0
            if K < 0: # too many 0's
                K += 1 - A[i] # shrink left index
                i += 1
            elif j-i > ans[1]-ans[0]: ans = [i,j]
        return list(range(ans[0],ans[1]+1))
    def longestOnes(self, A: List[int], B: int) -> int: # sliding window of 1's (w/ flipped 0's) to find largest width
        start = end = wL = wR = nZero = 0; bestWindowWidth = -1 # look for best window width
        while wR < len(A):
            if nZero <= B: # track number of 0's flipped as growing to window right
                if A[wR] == 0: nZero += 1
                wR += 1
            if nZero > B: # shrink from window left, update '0' count
                if (A[wL] == 0): nZero -= 1
                wL += 1
            if wR - wL + 1 > bestWindowWidth: # update best window as sliding 1's window right
                bestWindowWidth = wR - wL
        return bestWindowWidth
    # Sliding window: for each A[j], try to find the longest subarray.
    # If A[i] ~ A[j] has zeros <= K, we continue to increment j.
    # If A[i] ~ A[j] has zeros > K, we increment i (as well as j).
    def longestOnes(self, A, B):
        ans = l = i = count = 0
        for i in range(len(A)): # grow right index
            if A[i] == 0: count += 1 # incr 0's count
            while count > B: # shrink left index
                if A[l] == 0: count -= 1 # decr 0's count
                l += 1
            ans = max(ans, i-l+1)
        return ans # window width
    def longestOnes(self, A, K):
        i = j = 0 # left window index
        for j in range(len(A)): # each iteration grows j right window index
            K -= 1 - A[j] # decr K if A[j] is 0
            if K < 0: # too many 0's
                K += 1 - A[i] # shrink left index
                i += 1
        return j - i + 1
    def findPairs(self, A: List[int], B: int) -> int:
        aCounters = Counter(A); pairCount = 0
        if not A: return 0
        if B == 0: return sum([1 for n in aCounters if aCounters[n] > 1])
        for n in aCounters:
            if n+B in aCounters:
                pairCount += 1
        return pairCount
    def findPairs(self, nums, k):
        res = 0
        c = Counter(nums)
        for i in c:
            if k > 0 and i + k in c or k == 0 and c[i] > 1:
                res += 1
        return res
if __name__=="__main__":
    sol = Solution()
    assert sol.longestOnes([1,1,1,0,0,0,1,1,1,1,0],2) == 6
    assert sol.longestOnes([0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,1,1],3) == 10
    assert sol.longestOnes([ 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0 ],4) == 12
    assert sol.maxone([ 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0 ],4) == [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    assert sol.maxone([1, 1, 0, 1, 1, 0, 0, 1, 1, 1],1) == [0, 1, 2, 3, 4]
    assert sol.maxone([1, 1, 0, 1, 1, 0, 0, 1, 1, 1],2) == [3, 4, 5, 6, 7, 8, 9]
    assert sol.maxone([1,0,1,0,1],1) == [0, 1, 2]
    assert sol.maxone([1,0,1,1,0],1) == [0, 1, 2, 3]
    assert sol.maxone([int(ch) for ch in '00001111110111011110001111110000'],1) == [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    # assert sol.findMaxConsecutiveOnes([1,0,1,1,0]) == 4
    # assert sol.findMaxConsecutiveOnes([1,0,1,0,1]) == 3
    # assert sol.findMaxConsecutiveOnes([int(ch) for ch in '00001111110111011110001111110000']) == 10
    # assert sol.sortColors([1,0,0]) == sorted([1,0,0])
    # assert sol.sortColors([0]) == [0]
    # assert sol.sortColors([1]) == [1]
    # assert sol.sortColors([2,0,1]) == sorted([2,0,1])
    # assert sol.sortColors([0, 1, 2, 0, 1, 2]) == sorted([0, 1, 2, 0, 1, 2])
    # assert sol.removeElement([4, 1, 1, 2, 1, 3], 1) == 3
    # assert sol.removeElement([4, 1, 1, 2, 1, 3], 2) == 5
    # assert sol.removeDuplicates([1,1,2]) == 2
    # assert sol.removeDuplicates([0,0,1,1,1,2,2,3,3,4]) == 5
    # assert sol.removeDuplicatesII([1,1,1,2]) == 3
    # assert sol.removeDuplicatesII([1,1,1,2,2,3]) == 5
    # assert sol.removeDuplicatesII([0,0,1,1,1,1,2,3,3]) == 7
    # assert sol.intersect([],[]) == []
    # assert sol.intersect([1,2,2,1], [2,2]) == [2, 2]
    # assert sol.intersect([4,9,5], [9,4,9,8,4]) == [4, 9]
    # assert sol.intersect([1, 2, 3, 3, 4, 5, 6], [3, 3, 5]) == [3, 3, 5]
    # assert sol.intersect([1, 2, 3, 3, 4, 5, 6], [3, 5]) == [3, 5]
    # assert sol.merge([1, 5, 8], [6, 9]) == [1, 5, 6, 8, 9]
    # for a in [[6, 2, 5, 4, 5, 1, 6],[1, 2, 3, 4, 5],[1, 3, 4, 23],[1, 2, 3, 1, 4, 5, 2, 3, 6],[-2, -3, 4, -1, -2, 1, 5, -3], [1, 3, 1, 4, 23],[1, 3, 1, 4, 23]]:
    #     print(maxSubArrSum(a),max_subarraysubset(a))
    #     print(maxAllSubArraySizeK(a, 3))
    # for a,e in [([1,1,-2],[(-2, 1, 1)]),([-1,0,1,2,-1,-4],[(-1, -1, 2), (-1, 0, 1)]),([],[])]:
    #     assert sol.threeSum(a) == e
    # for s in ["cdd","abba","aab","abcabcbb","bbbbb","pwwkew","", " "]:
    #     print(sol.lengthOfLongestSubstring(s))
    # assert sol.diffPossible([0, 0, 2, 3, 8, 9, 14, 16, 16, 22, 24, 27, 30, 32, 34, 38, 39, 42, 44, 46, 58, 58, 65, 69, 83, 84], 3) == 1
    # assert sol.diffPossible([1, 4, 10, 16, 17, 20, 22, 24, 24, 25, 30, 35, 35, 36, 44, 45, 49, 49, 49, 53, 54, 57, 59, 59, 63, 65, 68, 71, 72, 87, 87, 88, 89, 96, 98, 98, 99, 100], 14) == 1
    # assert sol.diffPossible([1, 3, 5], 4) == 1
    # assert sol.diffPossible([1, 3, 5], 2) == 1
    # assert sol.diffPossible([-10, 20], 30) == 1
    # assert sol.diffPossible([5, 10, 3, 2, 50, 80], 78) == 1
    # assert sol.diffPossible([0, 1, 9, 10, 13, 17, 17, 17, 23, 25, 29, 30, 37, 38, 39, 39, 40, 41, 42, 60, 64, 70, 70, 70, 72, 75, 85, 85, 90, 91, 91, 93, 95], 83) == 1
    # assert sol.findPairs([3,1,4,1,5], 2) == 2
    # assert sol.findPairs([1,2,3,4,5], 1) == 4
    # assert sol.findPairs([1,3,1,5,4], 0) == 1
    # assert sol.findPairs([1,2,4,4,3,3,0,9,2,3], 3) == 2
    # assert sol.findPairs([-1,-2,-3], 1) == 2