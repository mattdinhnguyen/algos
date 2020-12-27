from timeit import timeit
from bisect import bisect
import sys
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
    def threeSum1(self, nums):
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
    def threeSum0(self, nums):
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
        res = []
        nums.sort()
        idx0 = bisect(nums,0)
        for i in range(idx0):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            l, r = i+1, len(nums)-1
            while l < r:
                s = nums[i] + nums[l] + nums[r]
                if s < 0:
                    l +=1 
                elif s > 0:
                    r -= 1
                else:
                    res.append((nums[i], nums[l], nums[r]))
                    while l < r and nums[l] == nums[l+1]:
                        l += 1
                    while l < r and nums[r] == nums[r-1]:
                        r -= 1
                    l += 1; r -= 1
        return res
if __name__=="__main__":
    sol = Solution()
    # for a in [[6, 2, 5, 4, 5, 1, 6],[1, 2, 3, 4, 5],[1, 3, 4, 23],[1, 2, 3, 1, 4, 5, 2, 3, 6],[-2, -3, 4, -1, -2, 1, 5, -3], [1, 3, 1, 4, 23],[1, 3, 1, 4, 23]]:
    #     print(maxSubArrSum(a),max_subarraysubset(a))
    #     print(maxAllSubArraySizeK(a, 3))
    for a in [[1,1,-2],[-1,0,1,2,-1,-4],[0],[]]:
        print(sol.threeSum(a))
    # for s in ["cdd","abba","aab","abcabcbb","bbbbb","pwwkew","", " "]:
    #     print(sol.lengthOfLongestSubstring(s))
