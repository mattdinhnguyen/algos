from timeit import timeit
from bisect import bisect, bisect_left
import sys
from typing import List
from collections import Counter, defaultdict
from itertools import combinations, groupby
from heapq import heappush, heappop, merge
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
    # https://leetcode.com/problems/longest-substring-without-repeating-characters/discuss/347818/Python3%3A-sliding-window-O(N)-with-explanation
    def lengthOfLongestSubstring(self, s: str) -> int:
        seen = {}; l = output = 0 # index of seen char so far
        for r,ch in enumerate(s):
            if ch not in seen or seen[ch] < l: # ch not seen or outside current window: increasing the window size by moving right pointer
                output = max(output,r-l+1)
            else: l = seen[ch] + 1 # ch inside the current window, move left pointer to seen[ch] + 1.
            seen[ch] = r # seen[charactor] = index
        return output
    # https://leetcode.com/problems/fraction-to-recurring-decimal/
    def fractionToDecimal(self, numerator, denominator):
        sign = '-' if numerator*denominator < 0 else ''
        numerator, denominator = abs(numerator), abs(denominator)
        n, remainder = divmod(numerator, denominator)
        result = [sign+str(n), '.']
        remainders = {} # dictionary maps the remainder to its index in result list
        while remainder > 0 and remainder not in remainders: # stop when remainder == 0 or have seen remainder
            remainders[remainder] = len(result) # indices of possibly repeating positions
            n, remainder = divmod(remainder*10, denominator)
            result.append(str(n))
        if remainder in remainders:
            idx = remainders[remainder]
            result.insert(idx, '(')
            result.append(')')
        return ''.join(result).rstrip(".") # strip "." in case no fractionals
    # https://leetcode.com/problems/combination-sum-iii/ all valid combinations of k numbers 1-9 that sum up to n, no replacement
    # https://leetcode.com/problems/combination-sum-iii/discuss/60624/Clean-167-liners-(AC)
    def combinationSum3(self, k: int, target: int) -> List[List[int]]: # backtracking, second fastest
        res = []
        def dfs(remain, index, path):
            if len(path) == k and remain == 0:
                res.append(path)
            elif len(path) < k:
                for i in range(index, len(candy)):
                    if candy[i] > remain: break
                    dfs(remain-candy[i], i+1, path+[candy[i]])
        candy = [i for i in range(1,min(10,target))]
        dfs(target, 0, [])
        return res
    def combinationSum3(self, k, n): # fast but not intuittive
        def combs(k, n, cap):
            if not k:
                return [[]] * (not n)
            ret = []
            for last in range(1, cap):
                for comb in combs(k-1, n-last, last):
                    ret.append(comb + [last])
            return ret
        return combs(k, n, 10)
    # https://leetcode.com/problems/combination-sum-iii/discuss/60805/Easy-to-understand-Python-solution-(backtracking). 39 -> 40 -> 77 -> 78 -> 90 -> 216
    def combinations(self, nums, k): # return unique k-num combs from sorted nums (having dups) itertools can't handle dups
        def dfs(nums, k, path, ret):
            if k < 0: return
            if k == 0: ret.append(path)
            for i in range(len(nums)):
                if i+1 < len(nums) and nums[i+1] == nums[i]: continue # skip dups
                dfs(nums[i+1:], k-1, path+[nums[i]], ret)
        ret = []
        dfs(nums, k, [], ret)
        return ret
    def gcd(self,a,b):
        return self.gcd(b,a%b) if b else a
    # https://leetcode.com/problems/max-points-on-a-line/discuss/47108/Python-68-ms-code https://leetcode.com/user7784J
    def maxPoints(self, points: List[List[int]]) -> int:
        def helper(currentPoint, points):
            slopes,duplicates,ans = defaultdict(int),0,0
            x1, y1 = currentPoint
            for x2, y2 in points:
                if x1 == x2 and y1 == y2: duplicates += 1 # increment duplicate counter
                else: # find the slope and add in dic
                    dy, dx = y2-y1, x2-x1
                    temp = self.gcd(dy,dx)
                    slope = (dy/temp, dx/temp)
                    slopes[slope] += 1
                    # ans = max(ans, slopes[slope])
            return max(slopes.values() or [0]) + 1 + duplicates # x2,y2 points + currentPoint + dups
        ans = 0
        while points:
            currentPoint = points.pop()
            ans = max(ans, helper(currentPoint, points))
        return ans
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
    # https://www.interviewbit.com/problems/diffk/
    def diffPossible(self, A, B) -> int: # 0/1 False/True
        if not A or len(A) == 1: return 0
        i = 0; j = len(A)-1
        while i < j:
            if A[j] - A[i] < B:
                if A[j] - A[0] > B:
                    while i<j and A[j] == A[j-1]: j -= 1
                    i = 0; j -= 1
                else: # max(A) - min(A) < B
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
    def diffPossible(self, A, B): # fastest
        i, j = 0, 1
        while i < len(A) and j < len(A):
            k = A[j] - A[i]
            if k == B: return 1
            elif k < B: j += 1
            else:
                i += 1
                if i == j: j += 1
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
    def sortColors(self, nums: List[int]) -> None: # https://en.wikipedia.org/wiki/Dutch_national_flag_problem
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
    def sortColors(self, A: List[int]) -> None:
        a=A.count(0); b=A.count(1); ab = a+b
        for i in range(len(A)):
            if i<a: A[i]=0
            elif i<ab: A[i]=1
            else:  A[i]=2
        return A
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
    # Find the smallest range that includes at least one number from each of the k sorted lists of int.
    # 1. initialize smallest_range as MAX_INT
    # 2. keep 3 pointers/index p1, p2 and p3 which points to the first elements of lists L1, L2 and L3 respectively.
    # 3. find the max value and min value pointed/indexed by p1, p2 and p3
    # 4. difference of max value and min value discovered in step 3 is the current range. compare it with smallest_range and update it, if found smaller.
    # 5. increment the pointer/index of min value found in step 3.
    # 6. repeat step 3 to 5 until the pointer/index of min value is in range.
    def find_minimum_range(self, sequences: List[List[int]]):#N
        sequences = [[(item, n) for item in seq] for n, seq in enumerate(sequences)] # which sequence each item belongs to
        heap = merge(*sequences) # Merge sequences into a single minheap, taking advantage of already sorted lists
        # Current items to test
        found_range = None
        last_range = sys.maxsize
        current_items = [None] * len(sequences)
        for item, n in heap: #N
            current_items[n] = item
            if not all(current_items): continue # List not yet filled
            # Find range of current selection
            minimum = min(current_items)
            maximum = max(current_items)
            current_range = abs(maximum-minimum)
            # Update minimum range
            if current_range < last_range:
                found_range = [minimum, maximum]
                last_range = current_range
        return last_range
        # return found_range
    # Find i, j, k such that: max(abs(A[i] - B[j]), abs(B[j] - C[k]), abs(C[k] - A[i])) is minimized.
    # Return the minimum max(abs(A[i] - B[j]), abs(B[j] - C[k]), abs(C[k] - A[i]))
    def minimize(self, A, B, C):
        i = j = k = amb = bmc = cma = 0; diffMn = sys.maxsize
        while i < len(A) and j < len(B) and k < len(C):
            amb = A[i]-B[j]; bmc = B[j]-C[k]; cma = C[k]-A[i]
            diffMn = min(diffMn, max(abs(amb), abs(bmc), abs(cma)))
            if abs(amb) >= abs(bmc) and abs(amb) >= abs(cma):
                if amb > 0: j += 1 # reduce amb
                else: i += 1
            elif abs(bmc) >= abs(cma) and abs(bmc) >= abs(amb):
                if bmc > 0: k += 1 # reduce bmc
                else: j += 1
            elif abs(cma) >= abs(amb) and abs(cma) >= abs(bmc):
                if cma > 0: i += 1 # reduce cma
                else: k += 1
        return diffMn
    # smallest range: Find 3 closest elements from given three sorted arrays
    def minimize(self, A, B, C):
        i = j = k = 0; diffMn = sys.maxsize
        ires = jres = kres = 0
        while i < len(A) and j < len(B) and k < len(C):
            mn = min(A[i], B[j], C[k])
            mx = max(A[i], B[j], C[k])
            if mx-mn < diffMn:
                diffMn = mx-mn
                ires, jres, kres = i, j, k
            if diffMn == 0: break
            if A[i] == mn: i += 1 # incr mn index
            elif B[j] == mn: j += 1
            else: k += 1
        print(A[ires], B[jres], C[kres])
        return diffMn
    def findPairs(self, A: List[int], B: int) -> int:
        if not A: return 0
        aCounters = Counter(A); pairCount = 0
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
    def maxArea(self, A):
        area = left = 0; right = len(A)-1
        while left < right:
            area = max(area, min(A[left], A[right])*(right-left))
            if A[left] < A[right]: left += 1
            else: right -= 1
        return area
if __name__=="__main__":
    sol = Solution()
    A = [1, 5, 4, 3]
    assert sol.maxArea(A) == 6
    # assert sol.fractionToDecimal(1, 2) == "0.5"
    # assert sol.fractionToDecimal(2, 1) == "2"
    # assert sol.fractionToDecimal(2, 3) == "0.(6)"
    # assert sol.fractionToDecimal(4, 333) == "0.(012)"
    # assert sol.fractionToDecimal(1, 5) == "0.2"
    # assert sol.fractionToDecimal(1, 6) == "0.1(6)"
    tdata = [[15,12],[9,10],[-16,3],[-15,15],[11,-10],[-5,20],[-3,-15],[-11,-8],[-8,-3],[3,6],[15,-14],[-16,-18],[-6,-8],[14,9],[-1,-7],[-1,-2],[3,11],[6,20],[10,-7],[0,14],[19,-18],[-10,-15],[-17,-1],[8,7],[20,-18],[-4,-9],[-9,16],[10,14],[-14,-15],[-2,-10],[-18,9],[7,-5],[-12,11],[-17,-6],[5,-17],[-2,-20],[15,-2],[-5,-16],[1,-20],[19,-12],[-14,-1],[18,10],[1,-20],[-15,19],[-18,13],[13,-3],[-16,-17],[1,0],[20,-18],[7,19],[1,-6],[-7,-11],[7,1],[-15,12],[-1,7],[-3,-13],[-11,2],[-17,-5],[-12,-14],[15,-3],[15,-11],[7,3],[19,7],[-15,19],[10,-14],[-14,5],[0,-1],[-12,-4],[4,18],[7,-3],[-5,-3],[1,-11],[1,-1],[2,16],[6,-6],[-17,9],[14,3],[-13,8],[-9,14],[-5,-1],[-18,-17],[9,-10],[19,19],[16,7],[3,7],[-18,-12],[-11,12],[-15,20],[-3,4],[-18,1],[13,17],[-16,-15],[-9,-9],[15,8],[19,-9],[9,-17]]
    assert sol.maxPoints(tdata) == 6
    assert sol.maxPoints([[0,0],[94911150,94911151],[94911151,94911152]]) == 2
    assert sol.maxPoints([[1,1],[1,1],[0,0],[3,4],[4,5],[5,6],[7,8],[8,9]]) == 5
    assert sol.maxPoints([[1,1],[2,2],[3,3],[4,4]]) == 4
    assert sol.maxPoints([[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]]) == 4
    # As = [[4, 10, 15, 24, 26],[0, 9, 12, 20],[5, 18, 22, 30]]
    # assert sol.find_minimum_range(As) == sol.minimize(*As)
    # As = [[5, 7, 9, 11, 12, 20],[4, 5, 11, 14, 16, 20],[1, 3, 4, 8, 10, 20]]
    # assert sol.find_minimum_range(As) == sol.minimize(*As)
    # As = [[1,4,12],[2,10,11],[10,11]]
    # assert sol.find_minimum_range(As) == sol.minimize(*As)
    # assert sol.minimize([1, 4, 10], [2, 15, 20], [10, 12]) == 5
    # assert sol.minimize([20, 24, 100],[2, 19, 22, 79, 800],[10, 12, 23, 24, 119]) == 2
    # assert sol.longestOnes([1,1,1,0,0,0,1,1,1,1,0],2) == 6
    # assert sol.longestOnes([0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,1,1],3) == 10
    # assert sol.longestOnes([ 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0 ],4) == 12
    # assert sol.maxone([ 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0 ],4) == [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    # assert sol.maxone([1, 1, 0, 1, 1, 0, 0, 1, 1, 1],1) == [0, 1, 2, 3, 4]
    # assert sol.maxone([1, 1, 0, 1, 1, 0, 0, 1, 1, 1],2) == [3, 4, 5, 6, 7, 8, 9]
    # assert sol.maxone([1,0,1,0,1],1) == [0, 1, 2]
    # assert sol.maxone([1,0,1,1,0],1) == [0, 1, 2, 3]
    # assert sol.maxone([int(ch) for ch in '00001111110111011110001111110000'],1) == [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    # assert sol.findMaxConsecutiveOnes([1,0,1,1,0]) == 4
    # assert sol.findMaxConsecutiveOnes([1,0,1,0,1]) == 3
    # assert sol.findMaxConsecutiveOnes([int(ch) for ch in '00001111110111011110001111110000']) == 10
    assert sol.sortColors([1,0,0]) == [0,0,1]
    assert sol.sortColors([0]) == [0]
    assert sol.sortColors([1]) == [1]
    assert sol.sortColors([2,0,1]) == [0,1,2]
    assert sol.sortColors([0, 1, 2, 0, 1, 2]) == [0,0,1,1,2,2]
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
    assert sol.diffPossible([0, 0, 2, 3, 8, 9, 14, 16, 16, 22, 24, 27, 30, 32, 34, 38, 39, 42, 44, 46, 58, 58, 65, 69, 83, 84], 3) == 1
    assert sol.diffPossible([1, 4, 10, 16, 17, 20, 22, 24, 24, 25, 30, 35, 35, 36, 44, 45, 49, 49, 49, 53, 54, 57, 59, 59, 63, 65, 68, 71, 72, 87, 87, 88, 89, 96, 98, 98, 99, 100], 14) == 1
    assert sol.diffPossible([1, 3, 5], 4) == 1
    assert sol.diffPossible([1, 3, 5], 2) == 1
    assert sol.diffPossible([-10, 20], 30) == 1
    assert sol.diffPossible([5, 10, 3, 2, 50, 80], 78) == 1
    assert sol.diffPossible([0, 1, 9, 10, 13, 17, 17, 17, 23, 25, 29, 30, 37, 38, 39, 39, 40, 41, 42, 60, 64, 70, 70, 70, 72, 75, 85, 85, 90, 91, 91, 93, 95], 83) == 1
    assert sol.findPairs([3,1,4,1,5], 2) == 2
    assert sol.findPairs([1,2,3,4,5], 1) == 4
    assert sol.findPairs([1,3,1,5,4], 0) == 1
    assert sol.findPairs([1,2,4,4,3,3,0,9,2,3], 3) == 2
    assert sol.findPairs([-1,-2,-3], 1) == 2