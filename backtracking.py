#!/usr/local/bin/python
from heapq import heapify, heappop, heappush, nlargest, nsmallest, heappushpop
from typing import List, Optional, Deque, Set
from bisect import insort, bisect, bisect_left, bisect_right
from subArraySumX import NumSubArrSumX
from collections import deque, Counter, defaultdict
from trees import BinarySearchTree, inOrder, preOrder
import string
from itertools import combinations, permutations, combinations_with_replacement
from functools import reduce, lru_cache, cache
import sys
import math
import operator
from timeit import timeit

class Solution():    
    # https://leetcode.com/problems/palindrome-partitioning/discuss/42021/Backtrack-Summary%3A-General-Solution-for-10-Questions!!!!!!!!-Python-(Combination-Sum-Subsets-Permutation-Palindrome)
    # https://leetcode.com/problems/combination-sum/
    # Elements are distinct. Same number may be chosen from candidates an unlimited number of times.
    # Return all unique combinations (frequency of at least 1 chosen number is different.
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        def dfs(remain, index, path):
            if remain == 0:
                res.append(path)
            elif remain >= candidates[index]:
                for i in range(index, len(candidates)):
                    dfs(remain-candidates[i], i, path+[candidates[i]])
        dfs(target, 0, [])
        return res
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]: # best time by a bit: sorted candy allowing the break line
        res = []
        def dfs(remain, index, path):
            if remain == 0:
                res.append(path)
            else:
                for i in range(index, len(candy)): # try all candidate combinations with replacement
                    if i>index and candy[i] == candy[i-1]: continue
                    if candy[i] > remain: break
                    dfs(remain-candy[i], i, path+[candy[i]]) # keep i for replacement
        candy = sorted(candidates)
        dfs(target, 0, [])
        return res
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]: #TLE
        res = [[target]] if target in candidates else []
        # if 1 in candidates and target>len(candidates)+1: res.append([1]*target)
        for i in range(2,max(len(candidates)+2,target+1)):
            for l in combinations_with_replacement(candidates, i):
                if sum(l) == target:
                    res.append(l)
        return res
    # https://leetcode.com/problems/combination-sum-ii/discuss/16870/DP-solution-in-Python
    # Return all unique combs, no replacement
    def combinationSum2dp(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        dp = [set() for _ in range(target + 1)]
        dp[0].add(()) # fill dp[0] as seed for the inner loop
        for c in candidates: # starting with smallest coin value
            for subtarget in range(target - c + 1): # looping thru subtarget 0..target-c to build dp[c],dp[c+1] ... dp[target] 
                dp[subtarget + c] |= {prev_list + (c,) for prev_list in dp[subtarget]} # starting at dp[0] == set(()), add c to each set member for each c
        return list(map(list, dp[-1]))
    # https://leetcode.com/problems/combination-sum-ii/discuss/16870/DP-solution-in-Python
    def combinationSum2dp(self, candidates: List[int], target: int) -> List[List[int]]: # a bit slower
        candidates.sort()
        dp = [set() for _ in range(target+1)]
        dp[0].add(())
        for num in candidates:
            for t in range(target, num-1, -1): # if you start from small value, then your dp table gets overwritten and corrupted.
                for prev in dp[t-num]:
                    dp[t].add(prev + (num,))
        return list(dp[-1])
    # 
    def combinationSum2dp(self, candidates: List[int], target: int) -> List[List[int]]:
        dp = [set([()])] + [set() for _ in range(target)]
        for candidate in sorted(candidates):
            for i in range(target, candidate - 1, -1):
                dp[i] |= {sublist + (candidate,) for sublist in dp[i - candidate]}
        return dp[target]
    # https://leetcode.com/problems/combination-sum-ii/ Each number in candidates may only be used once in the combination.
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]: # best time
        res = []
        def dfs(remain, index, path):
            if remain == 0:
                res.append(path)
            else:
                for i in range(index, len(candy)):
                    if candy[i] > remain: break
                    if i > index and candy[i] == candy[i-1]:
                        continue # Note: Skip, The solution set must not contain duplicate combinations.
                    dfs(remain-candy[i], i+1, path+[candy[i]]) # i+1, no replacement
        candy = sorted(candidates)
        dfs(target, 0, [])
        return res
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]: # best time
        res = []
        def dfs(remain, index, path):
            if remain == 0: res.append(path)
            else:
                for i in range(index, len(candidates)):
                    if i > index and candidates[i] == candidates[i-1]: continue # skip duplicates
                    if remain < candidates[i]: break
                    dfs(remain-candidates[i], i+1, path+[candidates[i]])
        candidates.sort()
        dfs(target, 0, [])
        return res
    # https://leetcode.com/problems/combination-sum-iii/ all valid combinations of k numbers 1-9 that sum up to n, no replacement
    # https://leetcode.com/problems/combination-sum-iii/discuss/60624/Clean-167-liners-(AC)
    def combinationSum3(self, k: int, target: int) -> List[List[int]]: # backtracking, second fastest
        # return [c for c in combinations(range(1, 10), k) if sum(c) == n]
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
    # https://leetcode.com/problems/combination-sum-iii/discuss/60624/Clean-167-liners-(AC)
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
    def combinationSum3(self, k, n): # iterative
        combs = [[]]
        for _ in range(k):
            combs = [[first] + comb
                     for comb in combs
                     for first in range(1, comb[0] if comb else 10)]
        return [c for c in combs if sum(c) == n]
    def combinationSum4(self, nums: List[int], target: int) -> int: # combinstions, not permutations
        # return list(filter(lambda t: sum(t) == target, 
        ls = []
        for gen in (combinations_with_replacement(nums,i) for i in range(1,target+1)):
            ls += [list(t) for t in gen if sum(t) == target]
        return ls
    # https://leetcode.com/problems/combination-sum-iv/discuss/85036/1ms-Java-DP-Solution-with-Detailed-Explanation
    def combinationSum4(self, nums: List[int], target: int) -> int: # permutations, TLE
        if target == 0: return 1
        res = 0
        for n in nums:
            if target >= n:
                res += self.combinationSum4(nums, target-n)
        return res
    def combinationSum4(self, nums: List[int], target: int) -> int: # top-down, recursive
        dp = [1] + [-1]*target
        def dfs(target):
            if dp[target] != -1: return dp[target]
            res = 0
            for n in nums:
                if target > n:
                    res += dfs(target-n)
                elif target == n:
                    res += 1
                else:
                    break
            dp[target] = res
            return res
        return dfs(target)
    def combinationSum4(self, nums: List[int], target: int) -> int: # bottom-up, iterative
        comb = [1] + [0]*target
        nums.sort()
        for t in range(1, target+1):
            for n in nums:
                if t < n: break
                comb[t] += comb[t-n]
        return comb[target]
    # return all possible combinations of k numbers out of 1 ... n.
    def combine(self, n: int, k: int) -> List[List[int]]:
        return list(combinations((i for i in range(1,n+1)), k))
    def combine(self, n: int, k: int) -> List[List[int]]: # recursive, hard to follow
        if k == 0: return [[]]
        ret = []
        for i in range(k, n+1):
            # print(n,k,i)
            for pre in self.combine(i-1, k-1):
                # print(n,k,i,pre)
                ret.append(pre + [i])
        return ret
    def combinebt(self, n: int, k: int) -> List[List[int]]: # straight forward, backtraking
        res = []
        def dfs(k, index, path):
            if k == 0:
                res.append(path)
                return
            for i in range(index, len(candy)):
                dfs(k-1, i+1, path+[candy[i]])
        candy = [i for i in range(1,n+1)]
        dfs(k, 0, [])
        return res
    def combine(self, n: int, k: int) -> List[List[int]]: # recursive, 
        if k == 1:
            return [[i] for i in range(1,n+1)]
        elif k == n:
            return [[i for i in range(1,n+1)]]
        else:
            rs = []
            rs += self.combine(n-1,k)
            part = self.combine(n-1,k-1)
            for ls in part: ls.append(n)
            rs += part
            return rs
    def combine(self, n: int, k: int) -> List[List[int]]: # iterative, best time, can follow somewhat
        combs = [[]]
        for j in range(k, 0, -1):
            _combs = []
            for c in combs:
                for i in range(j, c[0] if c else n+1):
                    _combs.append([i] + c)
                    # print(j,c,i,_combs)
            combs = _combs
        return combs
    def combine(self, n, k): # similar to iterative, using reduce in lieu of for loop
        return reduce(lambda C, _: [[i]+c for c in C for i in range(1, c[0] if c else n+1)], range(k), [[]])
    def combinations(self, iterable, r):
        # combinations('ABCD', 2) --> AB AC AD BC BD CD
        # combinations(range(4), 3) --> 012 013 023 123
        pool = tuple(iterable); n = len(pool)
        if r > n: return
        indices = list(range(r))
        yield tuple(pool[i] for i in indices) # return first r nums
        while True:
            for i in reversed(range(r)): # r-1..0
                if indices[i] != i + n - r:
                    break
            else:
                return
            indices[i] += 1
            for j in range(i+1, r):
                indices[j] = indices[j-1] + 1
            yield tuple(pool[i] for i in indices)
    def combine(self, n, k):
        return list(self.combinations((i for i in range(1,n+1)), k))
    def findPerm(self, A, B):
        if all(map(lambda c: c=='I', A)): return list(range(1,B+1))
        if all(map(lambda c: c=='D', A)): return list(range(B,0,-1))
        if A[0] == 'I':
            res = [1]
            lo, hi = 2, B
        else:
            res = [B]
            lo, hi = 1, B-1
        for i,ch in enumerate(A):
            if A[i:i+2] in ('II','DI','D'):
                res.append(lo)
                lo += 1
            elif A[i:i+2] in ('ID','I','DD'):
                res.append(hi)
                hi -= 1
        return res
    # https://leetcode.com/problems/permutations/
    def permute(self, nums: List[int]) -> List[List[int]]:
        return list(self.permutations(nums))
    def permutations(self, iterable, r=None):
        # permutations('ABCD', 2) --> AB AC AD BA BC BD CA CB CD DA DB DC
        # permutations(range(3)) --> 012 021 102 120 201 210
        pool = tuple(iterable)
        n = len(pool)
        r = n if r is None else r
        if r > n:
            return
        indices = list(range(n))
        cycles = list(range(n, n-r, -1))
        yield tuple(pool[i] for i in indices[:r])
        while n:
            for i in reversed(range(r)):
                cycles[i] -= 1
                if cycles[i] == 0:
                    indices[i:] = indices[i+1:] + indices[i:i+1]
                    cycles[i] = n - i
                else:
                    j = cycles[i]
                    indices[i], indices[-j] = indices[-j], indices[i]
                    yield tuple(pool[i] for i in indices[:r])
                    break
            else:
                return
    # https://leetcode.com/problems/permutations/discuss/18237/My-AC-simple-iterative-javapython-solution
    def permute(self, nums):
        perms = [[]]   
        for n in nums:
            new_perms = []
            for perm in perms:
                for i in range(len(perm)+1):   
                    new_perms.append(perm[:i] + [n] + perm[i:])   ###insert n
            perms = new_perms
        return perms
    # https://leetcode.com/problems/permutations/discuss/18284/Backtrack-Summary%3A-General-Solution-for-10-Questions!!!!!!!!-Python-(Combination-Sum-Subsets-Permutation-Palindrome)
    def permute(self, nums): # backtraking, close to best time
        res = []
        def backtrack(start, end):
            if start == end and nums not in res:
                res.append(nums[:])
            for i in range(start, end):
                nums[start], nums[i] = nums[i], nums[start]
                backtrack(start+1, end)
                nums[start], nums[i] = nums[i], nums[start]
        backtrack(0, len(nums))
        return res
    def permute(self, nums): # best time
        def gen(nums):
            if not nums:
                yield []
            for i, n in enumerate(nums):
                for p in gen(nums[:i] + nums[i+1:]):
                    yield [n] + p
        return list(gen(nums))
    def permuteUnique(self, nums: List[int]) -> List[List[int]]: # close to best time
        return list(set(permutations(nums)))
    # https://leetcode.com/problems/permutations-ii/discuss/18602/9-line-python-solution-with-1-line-to-handle-duplication-beat-99-of-others-%3A-)
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        perms = [[]]   
        for n in nums:
            new_perms = []
            for perm in perms:
                for i in range(len(perm)+1):   
                    new_perms.append(perm[:i] + [n] + perm[i:])   #insert n
                    if i<len(perm) and perm[i]==n: break          #handles duplication
            perms = new_perms
        return perms
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        perms = [[]]
        for n in nums:
            perms = [p[:i] + [n] + p[i:]
                 for p in perms
                 for i in range((p + [n]).index(n) + 1)] # up to index of 1st n in p+[n]
        return perms
    # https://leetcode.com/problems/permutation-sequence/discuss/22507/%22Explain-like-I'm-five%22-Java-Solution-in-O(n)
    # https://leetcode.com/problems/permutation-sequence/discuss/22512/Share-my-Python-solution-with-detailed-explanation
    # For permutations of n, the first (n-1)! permutations start with 1, next (n-1)! ones start with 2, ... and so on.
    # And in each group of (n-1)! permutations, the first (n-2)! permutations start with the smallest remaining number, ...
    def getPermutation(self, n, k):
        nums = list(map(str,range(1,10)))
        k -= 1
        factorial = [1] * n
        for i in range(1, n): factorial[i] = factorial[i - 1] * i
        res=[]
        for i in range(n-1,-1,-1): # n-1 to 0
            index = k // factorial[i] # (k-1)/(n-1)!
            res.append(nums[index])
            nums.remove(nums[index]) # O(N)
            k = k % factorial[i] 
        return ''.join(res)
    def getPermutation(self, n, k):
        numbers = list(range(1, n+1))
        permutation = ''
        k -= 1
        NN = reduce(operator.mul, numbers) # n!
        for n in range(n-1,-1,-1):
            NN = NN//(n+1) # (n-1)!
            index, k = divmod(k, NN) # get the index of current digit
            permutation += str(numbers[index])
            numbers.remove(numbers[index]) # remove handled number
        return permutation
    def getPermutation(self, n, k):
        elements = list(range(1, n+1))
        NN = reduce(operator.mul, elements) # n!
        k, result = (k-1) % NN, '' # (k-1)/n!
        while len(elements) > 0:
            NN = NN // len(elements) # (n-1)!
            i, k = k // NN, k % NN
            result += str(elements.pop(i))
        return result
    # https://leetcode.com/problems/next-permutation/discuss/13867/C%2B%2B-from-Wikipedia
    # Find the largest index k such that nums[k] < nums[k + 1]. If no such index exists, just reverse nums and done.
    # Find the largest index l > k such that nums[k] < nums[l].
    # Swap nums[k] and nums[l].
    # Reverse the sub-array nums[k + 1:]
    # 1. Find the largest index k such that nums[k] < nums[k + 1]. If no such index , just reverse
    # 2. Find the largest index l > k such that nums[k] < nums[l]
    # 3. Swap nums[k] and nums[l]
    # 4. Reverse the sub-array nums[k + 1:]
    # how to understand it:
    # step-1: easy, find the first digit that can be swapped to make permutation bigger
    # step-2: easy, find the digit bigger but closest to nums[k]
    # step-3: swap(nums[k], nums[l])
    # step-4: sort the subarray nums[k+1:end], why we can just reverse instead of sort?
    #         because we know nums[k+1:end] must be non-increasing, reason:
    #         1. at step 1, we know nums[k+1:end] is non-decreasing
    #         2. before swap in step 3, we know nums[l-1] >= nums[l] > nums[k] >= nums[l+1]
    #         3. so after swap, we still have nums[l-1] > nums[k] >= nums[l+1], so we can reverse it
    # /https://www.nayuki.io/page/next-lexicographical-permutation-algorithm
    def nextPermutation(self, nums: List[int]) -> None:
        k, l = -1, 0
        for l in range(len(nums)-1,0,-1):
            if nums[l] > nums[l-1]:
                k = l-1
                break
        if k == -1:
            nums.reverse()
            return nums
        if l < len(nums):
            for i in range(l+1,len(nums)):
                if nums[i] > nums[k]: l = i
        nums[k], nums[l] = nums[l], nums[k]
        l, r = k+1, len(nums)-1  # reverse the second part
        while l < r:
            nums[l], nums[r] = nums[r], nums[l]
            l +=1 ; r -= 1
        return nums
    # https://afteracademy.com/blog/print-all-subsets-of-a-given-set
    def subsets(self, nums: List[int]) -> List[List[int]]: # backtracking approach 2^N
        result = [[]] # generate subsets from the empty set result
        for num in nums:
            result += [i + [num] for i in result] # add next member to the result set iteratively
        return result
    # https://leetcode.com/problems/subsets/discuss/27281/A-general-approach-to-backtracking-questions-in-Java-(Subsets-Permutations-Combination-Sum-Palindrome-Partitioning)
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]: # 2^N
        result = [[]] # generate subsets from the empty list result
        res = []
        nums.sort()
        for idx,num in enumerate(nums):
            if idx and num == nums[idx-1]: # handle the duplicates
                res = [i + [num] for i in res] # buid from previous duplicate
            else:
                res = [i + [num] for i in result] # build from cummulative result
            result += res
        return result
    # https://leetcode.com/problems/subsets-ii/discuss/30305/Simple-python-solution-(DFS).
    def subsetsWithDup(self, nums):
        def dfs(nums, path, ret):
            ret.append(path)
            for i in range(len(nums)):
                if i > 0 and nums[i] == nums[i-1]: continue
                dfs(nums[i+1:], path+[nums[i]], ret)
        ret = []
        dfs(sorted(nums), [], ret)
        return ret
    # https://leetcode.com/problems/subsets-ii/submissions/
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        res = {tuple()}
        for n in sorted(nums):
            res |= {a + (n, ) for a in res}
        return res
    # https://leetcode.com/problems/target-sum/discuss/455024/DP-IS-EASY!-5-Steps-to-Think-Through-DP-Questions.
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        memo = {}
        def dp(index, curr_sum):
            if (index, curr_sum) in memo:
                return memo[(index, curr_sum)]
            if index < 0:
                return 1 if curr_sum == target else 0
            positive = dp(index-1, curr_sum + nums[index])
            negative = dp(index-1, curr_sum + -nums[index])
            memo[(index, curr_sum)] = positive + negative
            return memo[(index, curr_sum)]
        index = len(nums) - 1
        curr_sum = 0
        return dp(index, curr_sum)
    # https://leetcode.com/problems/target-sum/discuss/97439/JavaPython-Easily-Understood
    # Using 2 maps/dictionaries to implement a level order traversal
    def findTargetSumWays(self, nums, S):
        count = defaultdict(int)
        count[0] = 1
        for x in nums:
            step = defaultdict(int)
            for y in count: # store all possible sums using all the numbers with +/- signs
                step[y + x] += count[y]
                step[y - x] += count[y]
            count = step

        return count[S] # return the number of ways of the target sum in the dictionary
    # https://leetcode.com/problems/palindrome-partitioning/
    def partition(self, s: str) -> List[List[str]]: # same as below, but using index, slower
        def isPalindrome(l, r):
            while(l < r and s[l] == s[r]):
                l += 1
                r -= 1
            return True if l >= r else False
        out = []
        def dfs(curr=[], index=1):
            if index == len(s) + 1:
                out.append(list(curr))
                return
            for i in range(index, len(s)+1):
                if isPalindrome(index-1,i-1):
                    curr.append(s[index-1:i])
                    dfs(curr, i+1)
                    curr.pop()
        dfs()
        return out
    # https://leetcode.com/problems/palindrome-partitioning/discuss/41973/Python-recursiveiterative-backtracking-solution
    def partition(self, s): # 2nd best time
        res = []
        def dfs(s, path):
            if not s:
                res.append(path[:])
                return
            for i in range(1, len(s)+1):
                if s[:i] == s[i-1::-1]:
                    path.append(s[:i])
                    dfs(s[i:], path)
                    path.pop()        
        dfs(s, [])
        return res
    # https://leetcode.com/problems/palindrome-partitioning/discuss/42025/1-liner-Python-Ruby
    def partition(self, s):
        ret = []
        for i in range(1, len(s)+1):
            if s[:i] == s[i-1::-1]:
                for rest in self.partition(s[i:]):
                    ret.append([s[:i]]+rest)
        if not ret:
            return [[]]
        return ret
    # https://leetcode.com/problems/palindrome-partitioning/discuss/41973/Python-recursiveiterative-backtracking-solution
    def _helper(self, s, combination, start, result): # best time
        print(combination, start, result)
        if start == len(s):
            result.append(combination)
            return        
        for end in range(start, len(s)): # 0 0 -> ['a', 'a', 'b'] 0 1 -> ['aa', 'b']
            if self._is_palindrome(s, start, end):
                print(start,end)
                self._helper(s, combination + [s[start:end + 1]], end + 1, result)
    def _is_palindrome(self, s, left, right):
        while left < right:
            if s[left] != s[right]:
                return False
            left, right = left + 1, right - 1
        return True
    def partition(self, s: str) -> List[List[str]]:
        result = []
        self._helper(s, [], 0, result)
        return result
    def generateParenthesis(self, n: int) -> List[str]:
        if n == 0: return []
        ans = []
        def backtrack(string, open, close):
            if len(string) == 2*n:
                ans.append(string)
                return
            if open < n:
                backtrack(string+"(", open+1, close) # adding all open paren
            if close < open:
                backtrack(string+")", open, close+1) # adding matching close paren
        backtrack("", 0, 0)
        return ans
    def generateParenthesis(self, n: int) -> List[str]:
        if n == 0: return []
        res = []
        s = [("(", 1, 0)] # stack s of (string,l,r)
        while s:
            x, l, r = s.pop()
            if l - r < 0 or l > n or r > n: continue # invalid, stop stack append, keep poppinng stack till l >= r
            if l == n == r: res.append(x) # has n pairs
            s.append((x+"(", l+1, r)) # build strings of left parens till l == n
            s.append((x+")", l, r+1)) # add right parens till l == r == n
        # res.sort()
        return res[::-1]
    def letterCombinations(self, digits):
        if digits in (None,""): return []
        ans = []
        dlast = len(digits)-1
        def backtrack(string, di):
            if di == dlast:
                ans.extend([string+a for a in self.MAPPING[int(digits[di])]])
                return
            for a in self.MAPPING[int(digits[di])]:
                backtrack(string+a, di+1)
        backtrack("", 0)
        return ans
    MAPPING = ('0', '1', 'abc', 'def', 'ghi', 'jkl', 'mno', 'pqrs', 'tuv', 'wxyz')
    def letterCombinations0(self, digits):
        res = ['']
        for d in digits:
            res = [pre + c for pre in res for c in self.MAPPING[int(d)]]
        return res if len(digits) else []

if __name__ == "__main__":
    sol = Solution()
    assert sol.letterCombinations("23") == ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"]
    assert sol.generateParenthesis(3) == ["((()))","(()())","(())()","()(())","()()()"]
    assert list(sol.permutations(range(3))) == [(0, 1, 2),(0, 2, 1),(1, 0, 2),(1, 2, 0),(2, 0, 1),(2, 1, 0)]
    assert sol.permuteUnique([0,0,0,1,9])
    assert sol.permuteUnique([1,2,3,4, 5,6,7,8])
    print(sol.combine(9,2))
#   assert sol.partition("aab") == [["a","a","b"],["aa","b"]]
#   assert sol.convert2Palin("pwxu") == 0
#   assert sol.convert2Palin("iph") == 0
#   print(sol.restricted_knapsack(7, [2, 2, 5, 6], [1, 1, 2, 1]))
#   @timeit
#   def tw():
#     sol.computeMaxProfit((1,5,8,9,10,17,17,20,24,30), 10)
#   print(tw())
#   print(sol.nextPermutation([1,2,3]))
#   arr = MountainArray([1,2,3,4,5,3,1])
#   print(sol.findInMountainArray(3, arr))
#   print(sol.firstMissingPositive([1,2,0]))
#   print(sol.firstMissingPositive([3,4,-1,1]))
#   print(sol.findDuplicate([1,3,4,2,2]))
#   expected=[1,2,3,6,9,8,7,4,5]
#   print(expected)
#   print(sol.spiralOrder([[1,2,3],[4,5,6],[7,8,9]]))
#   for r in sol.generateMatrix(3): print(r)
#   for r in sol.generateMatrix(4): print(r)
#   print(sol.repeatMissingNumbers([3, 1, 2, 5, 3]))
#   print(sol.findDuplicate([1,3,4,2,2]))
#   print(sol.perfectPeak([1,3,2]))
#   print(sol.nextPermutation([100,99,98,97,96,95,94,93,92,91,90,89,88,87,86,85,84,83,82,81,80,79,78,77,76,75,74,73,72,71,70,69,68,67,66,65,64,63,62,61,60,59,58,57,56,55,54,53,52,51,50,49,48,47,46,45,44,43,42,41,40,39,38,37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]))
#   print(sol.nextPermutation([3,2,1]))
#   print(sol.getPermutation(4,13))
#   res = sol.findPerm("IIIDDDDDDDDIDDDIDDIIDDDDDDIIIIDIIIDDDIDIIIDDDIDDDDDDIIIDDDIIDDIIDIDIIIDIDIDIIIDDIIIIIDIIIIIDDIDDIDDDDIDIIDDIDIIDDIIDDIDDIDDDIIIIDIDDIDDDIIDDDDDIIDDDDDDDIIIIIDDIDIDDDIIDDIDIDIIDDDDIIIDDIDDIIIIDDDDIIDIDDDDDDDIIIDIDDDIDIDIDIIIDIDDIDDDIIIDDDIDDDDIDIIIDIIIIDIDDIIDDIIDIIDIDDDIIDDDDDIIDIIDDDIIDDDIDDDDDIDIDDDIIIDDDDIDIIIDDDIIIDIDDIIIIIDIDIIDIDIDDIIDDDIIIIDIIDDDDDDIDIIIIIDIDIIIIIDDDIIDIDDIIIDIIDIDDIIIIDIDDIIIDDDIDDIIIDIDIIDIDDDIDDIDDDIIDIIIIIDDDDDIIIDIIIIDDIDIDIDIDDDIIDDIDIDDDDDDDIIDIIIDIDDIDIIIDDDDDIDIIDDDIIIDIIIIDIDDDIDDIIDIDIDIIDDIIIDIDIDDIDIDDDDIIIDIIDIIDIIIDDIDIIDDIIIIDIIIIDIIDIIIDDIIDIIIIDDIDIDDIDDDIDDIIDIIDIIIDIDIIDIIIDIDDDIDDIIDDDDIDDIIDIDDIIDIDIDIDDDDDIIIIDDDIIDDDDIIDDDDDDIIDDIIIIDDIIDIDIDDIDDDIDIIIDDDIDDDIIIDIDIIDIIIIDIDIDIDIIDIIID",743)
#   expected = "1 2 3 743 742 741 740 739 738 737 736 4 735 734 733 5 732 731 6 7 730 729 728 727 726 725 8 9 10 11 724 12 13 14 723 722 721 15 720 16 17 18 719 718 717 19 716 715 714 713 712 711 20 21 22 710 709 708 23 24 707 706 25 26 705 27 704 28 29 30 703 31 702 32 701 33 34 35 700 699 36 37 38 39 40 698 41 42 43 44 45 697 696 46 695 694 47 693 692 691 690 48 689 49 50 688 687 51 686 52 53 685 684 54 55 683 682 56 681 680 57 679 678 677 58 59 60 61 676 62 675 674 63 673 672 671 64 65 670 669 668 667 666 66 67 665 664 663 662 661 660 659 68 69 70 71 72 658 657 73 656 74 655 654 653 75 76 652 651 77 650 78 649 79 80 648 647 646 645 81 82 83 644 643 84 642 641 85 86 87 88 640 639 638 637 89 90 636 91 635 634 633 632 631 630 629 92 93 94 628 95 627 626 625 96 624 97 623 98 622 99 100 101 621 102 620 619 103 618 617 616 104 105 106 615 614 613 107 612 611 610 609 108 608 109 110 111 607 112 113 114 115 606 116 605 604 117 118 603 602 119 120 601 121 122 600 123 599 598 597 124 125 596 595 594 593 592 126 127 591 128 129 590 589 588 130 131 587 586 585 132 584 583 582 581 580 133 579 134 578 577 576 135 136 137 575 574 573 572 138 571 139 140 141 570 569 568 142 143 144 567 145 566 565 146 147 148 149 150 564 151 563 152 153 562 154 561 155 560 559 156 157 558 557 556 158 159 160 161 555 162 163 554 553 552 551 550 549 164 548 165 166 167 168 169 547 170 546 171 172 173 174 175 545 544 543 176 177 542 178 541 540 179 180 181 539 182 183 538 184 537 536 185 186 187 188 535 189 534 533 190 191 192 532 531 530 193 529 528 194 195 196 527 197 526 198 199 525 200 524 523 522 201 521 520 202 519 518 517 203 204 516 205 206 207 208 209 515 514 513 512 511 210 211 212 510 213 214 215 216 509 508 217 507 218 506 219 505 220 504 503 502 221 222 501 500 223 499 224 498 497 496 495 494 493 492 225 226 491 227 228 229 490 230 489 488 231 487 232 233 234 486 485 484 483 482 235 481 236 237 480 479 478 238 239 240 477 241 242 243 244 476 245 475 474 473 246 472 471 247 248 470 249 469 250 468 251 252 467 466 253 254 255 465 256 464 257 463 462 258 461 259 460 459 458 457 260 261 262 456 263 264 455 265 266 454 267 268 269 453 452 270 451 271 272 450 449 273 274 275 276 448 277 278 279 280 447 281 282 446 283 284 285 445 444 286 287 443 288 289 290 291 442 441 292 440 293 439 438 294 437 436 435 295 434 433 296 297 432 298 299 431 300 301 302 430 303 429 304 305 428 306 307 308 427 309 426 425 424 310 423 422 311 312 421 420 419 418 313 417 416 314 315 415 316 414 413 317 318 412 319 411 320 410 321 409 408 407 406 405 322 323 324 325 404 403 402 326 327 401 400 399 398 328 329 397 396 395 394 393 392 330 331 391 390 332 333 334 335 389 388 336 337 387 338 386 339 385 384 340 383 382 381 341 380 342 343 344 379 378 377 345 376 375 374 346 347 348 373 349 372 350 351 371 352 353 354 355 370 356 369 357 368 358 367 359 360 366 361 362 363 365 364"
#   print(res == list(map(int,expected.split())), res)
#   print(sol.cherryPickup([[1,1,1,0,0],[0,0,1,0,1],[1,0,1,0,0],[0,0,1,0,0],[0,0,1,1,1]]))
    expected = [[[1,1,6],[1,2,5],[1,7],[2,6]],[[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 2], [1, 1, 1, 1, 1, 1, 3], [1, 1, 1, 1, 1, 2, 2], [1, 1, 1, 1, 2, 3], [1, 1, 1, 1, 5], [1, 1, 1, 2, 2, 2], [1, 1, 1, 3, 3], [1, 1, 1, 6], [1, 1, 2, 2, 3], [1, 1, 2, 5], [1, 1, 7], [1, 2, 2, 2, 2], [1, 2, 3, 3], [1, 2, 6], [1, 3, 5], [2, 2, 2, 3], [2, 2, 5], [2, 7], [3, 3, 3], [3, 6]],
[[1, 1, 1, 1], [1, 1, 2], [2, 2]], [[1, 1]], [[1]], [], [[2, 2, 2, 2], [2, 3, 3], [3, 5]], [[2, 2, 3], [7]]]
    tdata = [([10,1,2,7,6,5],8),([2,7,6,3,5,1],9),([1,2],4),([1],2),([1],1),([2],1),([2,3,5],8),([2,3,6,7],7)]
    tdata = [800,[[695,229],[199,149],[443,397],[258,247],[781,667],[350,160],[678,629],[467,166],[500,450],[477,107],[483,151],[792,785],[752,368],[659,623],[316,224],[487,268],[743,206],[552,211],[314,20],[720,196],[421,103],[493,288],[762,24],[528,318],[472,32],[684,502],[641,354],[586,480],[629,54],[611,412],[719,680],[733,42],[549,519],[697,316],[791,634],[546,70],[676,587],[460,58],[605,530],[617,579],[484,89],[571,482],[767,200],[555,547],[771,695],[624,542],[708,551],[432,266],[656,468],[724,317],[423,248],[621,593],[781,399],[535,528],[578,12],[770,549],[576,295],[318,247],[400,372],[465,363],[786,482],[441,398],[577,411],[524,30],[741,540],[459,59],[758,96],[550,89],[402,295],[476,336],[645,346],[750,116],[551,207],[343,226],[568,498],[530,228],[525,84],[507,128],[526,210],[535,381],[635,330],[654,535],[710,275],[397,213],[412,44],[131,70],[508,49],[679,223],[519,11],[626,286],[242,160],[778,199],[606,281],[226,16],[340,46],[578,127],[212,208],[674,343],[778,108],[749,451],[735,105],[544,131],[600,229],[691,314],[608,74],[613,491],[754,500],[722,449],[486,11],[786,70],[212,23],[717,11],[692,410],[503,157],[783,177],[220,215],[419,363],[182,17],[321,54],[711,78],[312,106],[560,101],[501,178],[583,403],[577,9],[595,227],[601,386],[792,619],[550,167],[589,431],[793,243],[395,76],[197,3],[357,6],[763,7],[599,48],[178,92],[325,307],[620,10],[334,117],[556,296],[454,394],[485,236],[140,80],[404,301],[651,58],[504,455],[101,93],[712,42],[559,421],[594,230],[505,98],[719,654],[672,283],[109,73],[556,183],[617,94],[133,100],[771,515],[613,587],[285,50],[579,432],[282,244],[669,527],[783,494],[628,560],[716,661],[177,127],[430,166],[383,159],[746,19],[653,284],[495,243],[376,57],[560,143],[679,198],[751,355],[339,157],[409,140],[729,389],[518,315],[623,352],[651,133],[761,269],[442,44],[379,245],[313,180],[773,583],[291,221],[271,54],[799,44],[200,102],[568,67],[695,167],[327,36],[431,73],[782,167],[611,129],[630,122],[563,497],[697,93],[596,436],[611,131],[627,256],[658,559],[591,419],[193,156],[302,52],[409,33],[405,249],[384,151],[214,142],[558,164],[565,557],[492,445],[681,271],[797,396],[251,195],[784,266],[607,179],[671,30],[752,179],[787,390],[749,532],[618,220],[659,298],[567,134],[229,208],[298,147],[787,459],[572,359],[794,351],[53,14],[646,422],[234,66],[274,255],[744,626],[730,462],[498,428],[573,288],[688,355],[603,25],[191,16],[793,544],[750,682],[415,156],[460,209],[749,85],[269,186],[441,338],[319,278],[505,18],[672,260],[420,233],[493,134],[493,19],[308,302],[582,282],[755,60],[641,626],[669,69],[772,29],[132,111],[666,120],[605,58],[534,252],[636,491],[777,3],[602,368],[533,287],[401,147],[782,669],[517,161],[686,49],[789,639],[776,379],[376,65],[696,545],[423,81],[448,336],[631,605],[501,387],[413,94],[777,563],[661,332],[756,359],[646,36],[650,283],[656,347],[522,7],[383,382],[438,102],[762,305],[650,15],[249,180],[784,467],[763,122],[163,115],[775,734],[166,132],[634,2],[668,584],[767,274],[595,552],[11,7],[693,407],[789,751],[613,556],[715,402],[751,516],[646,199],[625,52],[572,106],[724,332],[617,409],[573,526],[760,18],[382,202],[207,139],[416,392],[672,358],[233,212],[668,22],[765,452],[294,76],[259,47],[593,271],[510,450],[592,132],[770,558],[296,43],[419,86],[752,347],[615,605],[635,554],[794,635],[613,316],[563,61],[770,715],[771,251],[646,582],[423,79],[576,249],[604,97],[767,348],[736,239],[775,56],[619,601],[790,546],[531,384],[507,84],[564,337],[432,310],[600,543],[747,341],[556,392],[661,113],[449,282],[575,288],[637,7],[635,325],[735,574],[574,387],[705,603],[704,15],[684,588],[495,132],[718,223],[517,206],[272,34],[677,416],[788,167],[649,525],[619,427],[541,277],[489,405],[608,259],[603,264],[435,317],[623,26],[544,511],[72,69],[623,17],[600,544],[551,367],[404,52],[324,272],[706,205],[778,446],[341,155],[581,173],[666,192],[588,529],[554,506],[250,39],[772,116],[569,77],[526,132],[563,221],[655,597],[649,224],[57,4],[679,199],[265,157],[380,335],[558,35],[726,388],[763,567],[437,426],[643,103],[773,181],[726,68],[164,50],[717,427],[681,618],[477,172],[697,423],[525,383],[794,132],[149,70],[704,414],[581,139],[678,204],[107,46],[352,68],[645,178],[758,156],[627,365],[331,144],[547,340],[788,36],[633,259],[588,66],[321,102],[528,322],[212,36],[288,179],[434,189],[749,490],[753,508],[784,341],[550,159],[741,206],[758,688],[766,758],[586,70],[657,654],[701,104],[548,184],[613,162],[620,320],[506,430],[517,65],[571,291],[771,517],[796,756],[735,459],[625,367],[759,345],[582,468],[469,73],[790,352],[493,284],[664,567],[342,207],[669,108],[611,182],[764,485],[214,102],[544,202],[713,447],[793,378],[147,129],[407,198],[608,271],[695,667],[680,277],[222,163],[744,527],[280,116],[430,367],[281,228],[688,488],[733,92],[529,190],[750,718],[793,99],[626,169],[486,329],[620,0],[782,460],[329,16],[753,142],[338,172],[518,361],[688,168],[497,490],[484,365],[365,325],[107,98],[622,407],[527,277],[659,74],[552,538],[493,469],[638,147],[304,3],[573,201],[411,169],[719,309],[287,160],[742,175],[573,299],[562,473],[705,328],[261,98],[580,203],[740,26],[418,296],[764,170],[656,89],[724,536],[730,91],[796,290],[735,270],[512,20],[402,246],[46,30],[426,290],[296,57],[725,222],[324,317],[547,0],[661,136],[636,271],[261,56],[750,668],[647,402],[773,390],[677,62],[249,53],[574,4],[393,304],[701,44],[109,66],[275,109],[679,509],[725,21],[409,311],[368,156],[605,514],[538,42],[690,602],[411,343],[424,240],[78,40],[750,273],[367,230],[167,58],[738,200],[634,341],[409,170],[644,373],[741,296],[702,342],[746,233],[411,67],[526,436],[796,438],[647,312],[717,347],[548,54],[725,50],[549,92],[610,294],[668,350],[578,445],[446,93],[727,246],[526,355],[344,246],[145,54],[355,256],[751,46],[454,271],[587,49],[728,79],[627,49],[522,260],[270,250],[491,113],[337,258],[609,470],[387,147],[656,237],[366,357],[160,98],[761,692],[753,627],[718,5],[335,22],[640,78],[687,317],[315,295],[471,93],[481,147],[724,580],[687,177],[409,41],[355,276],[393,366],[770,85],[697,358],[187,115],[671,318],[716,530],[767,140],[566,543],[318,238],[341,336],[648,204],[496,202],[505,191],[360,32],[408,138],[537,489],[668,102],[400,9],[472,3],[727,469],[713,5],[530,292],[465,381],[551,262],[514,227],[394,315],[551,121],[655,402],[755,83],[153,144],[303,60],[766,578],[668,527],[796,391],[692,571],[617,616],[229,154],[798,690],[706,504],[610,569],[655,624],[408,108],[569,463],[461,151],[507,13],[781,314],[780,469],[506,171],[552,312],[189,164],[336,171],[571,432],[688,224],[160,119],[470,311],[663,114],[665,420],[556,492],[709,358],[202,99],[170,149],[340,154],[666,385],[617,383],[502,132],[220,42],[778,393],[444,68],[526,357],[217,7],[597,76],[586,406],[481,44],[486,240],[513,217],[790,447],[275,245],[396,1],[369,224],[485,159],[680,151],[387,312],[721,70],[733,25],[457,216],[798,297],[329,169],[766,212],[286,160],[703,164],[765,77],[620,142],[510,35],[475,400],[784,8],[768,189],[668,328],[697,2],[389,169],[550,223],[514,268],[579,285],[419,53],[318,96],[335,117],[729,27],[694,281],[349,137],[545,221],[679,100],[382,116],[707,140],[62,48],[664,312],[499,369],[547,350],[509,279],[778,76],[186,17],[741,683],[635,531],[441,391],[493,385],[354,218],[304,128],[651,271],[693,360],[613,112],[798,393],[743,190],[115,62],[725,592],[525,233],[621,517],[327,70],[501,358],[504,346],[787,321],[94,74],[729,339],[50,13],[603,265],[163,29],[781,373],[586,459],[797,741],[624,364],[411,277],[360,161],[690,686],[746,639],[553,325],[631,328],[388,330],[619,210],[573,43],[559,100],[210,152],[378,5],[776,447],[615,181],[365,299],[708,310],[718,690],[268,225],[639,90],[318,5],[196,89],[361,184],[762,690],[772,465],[729,721],[541,331],[567,350],[269,58],[656,78],[579,163],[711,223],[282,268],[760,533],[404,280],[473,384],[94,48],[340,12],[727,364],[264,221],[591,487],[514,466],[305,168],[372,248],[639,499],[560,435],[541,142],[462,83],[594,353],[618,485],[95,33],[602,595],[605,289],[715,207],[448,293],[752,170],[641,203],[532,198],[608,13],[707,114],[744,211],[110,3],[298,228],[622,496],[286,26],[683,178],[706,192],[751,358],[486,461],[561,251],[466,193],[342,62],[221,37],[731,325],[205,132],[518,173],[502,261],[640,49],[541,522],[747,110],[756,591],[124,76],[639,603],[765,482],[388,5],[34,12],[514,344],[495,254],[770,751],[730,597],[708,105],[683,586],[528,288],[386,225],[287,26],[649,262],[753,670],[789,85],[632,439],[570,176],[672,652],[445,399],[400,226],[655,522],[469,249],[557,500],[275,6],[397,296],[725,43],[605,533],[425,220],[637,118],[628,215],[654,431],[697,421],[512,121],[237,36],[151,85],[574,217],[320,233],[492,272],[552,220],[739,81],[712,219],[612,590],[410,66],[548,40],[320,211],[381,95],[633,482],[742,535],[704,131],[682,435],[508,48],[435,337],[534,96],[663,653],[283,205],[715,74],[484,376],[585,366],[635,479],[753,719],[793,548],[396,171],[156,112],[575,380],[717,464],[612,576],[569,319],[736,259],[406,227],[711,709],[793,132],[528,295],[592,48],[731,217],[408,299],[373,137],[786,327],[791,166],[712,285],[772,603],[723,338],[531,121],[572,548],[786,167],[670,401],[724,440],[280,229],[497,453],[265,70],[733,144],[689,434],[504,384],[93,64],[563,397],[550,106],[224,198],[372,177],[249,31],[667,372],[263,78],[783,446],[791,59],[438,64],[630,270],[216,160],[704,261],[674,506],[704,23],[378,4],[784,437],[196,118],[681,314],[698,663],[397,274],[499,440],[737,265],[697,625],[139,84],[440,231],[453,150],[266,55],[377,11],[728,60],[431,202],[268,47],[763,123],[347,339],[470,117],[466,298],[344,142],[584,55],[417,175],[439,392],[548,55],[714,701],[643,71],[357,69],[649,459],[789,541],[626,5],[752,619],[711,267],[639,12],[750,364],[620,249],[769,721],[636,97],[233,15],[171,72],[488,421],[251,139],[750,98],[199,64],[768,344],[759,537],[435,154],[425,185],[336,221],[418,395],[390,136],[618,603]]]
    tdata = [5000,[[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11],[11,12],[12,13],[13,14],[14,15],[15,16],[16,17],[17,18],[18,19],[19,20],[20,21],[21,22],[22,23],[23,24],[24,25],[25,26],[26,27],[27,28],[28,29],[29,30],[30,31],[31,32],[32,33],[33,34],[34,35],[35,36],[36,37],[37,38],[38,39],[39,40],[40,41],[41,42],[42,43],[43,44],[44,45],[45,46],[46,47],[47,48],[48,49],[49,50],[50,51],[51,52],[52,53],[53,54],[54,55],[55,56],[56,57],[57,58],[58,59],[59,60],[60,61],[61,62],[62,63],[63,64],[64,65],[65,66],[66,67],[67,68],[68,69],[69,70],[70,71],[71,72],[72,73],[73,74],[74,75],[75,76],[76,77],[77,78],[78,79],[79,80],[80,81],[81,82],[82,83],[83,84],[84,85],[85,86],[86,87],[87,88],[88,89],[89,90],[90,91],[91,92],[92,93],[93,94],[94,95],[95,96],[96,97],[97,98],[98,99],[99,100],[100,101],[101,102],[102,103],[103,104],[104,105],[105,106],[106,107],[107,108],[108,109],[109,110],[110,111],[111,112],[112,113],[113,114],[114,115],[115,116],[116,117],[117,118],[118,119],[119,120],[120,121],[121,122],[122,123],[123,124],[124,125],[125,126],[126,127],[127,128],[128,129],[129,130],[130,131],[131,132],[132,133],[133,134],[134,135],[135,136],[136,137],[137,138],[138,139],[139,140],[140,141],[141,142],[142,143],[143,144],[144,145],[145,146],[146,147],[147,148],[148,149],[149,150],[150,151],[151,152],[152,153],[153,154],[154,155],[155,156],[156,157],[157,158],[158,159],[159,160],[160,161],[161,162],[162,163],[163,164],[164,165],[165,166],[166,167],[167,168],[168,169],[169,170],[170,171],[171,172],[172,173],[173,174],[174,175],[175,176],[176,177],[177,178],[178,179],[179,180],[180,181],[181,182],[182,183],[183,184],[184,185],[185,186],[186,187],[187,188],[188,189],[189,190],[190,191],[191,192],[192,193],[193,194],[194,195],[195,196],[196,197],[197,198],[198,199],[199,200],[200,201],[201,202],[202,203],[203,204],[204,205],[205,206],[206,207],[207,208],[208,209],[209,210],[210,211],[211,212],[212,213],[213,214],[214,215],[215,216],[216,217],[217,218],[218,219],[219,220],[220,221],[221,222],[222,223],[223,224],[224,225],[225,226],[226,227],[227,228],[228,229],[229,230],[230,231],[231,232],[232,233],[233,234],[234,235],[235,236],[236,237],[237,238],[238,239],[239,240],[240,241],[241,242],[242,243],[243,244],[244,245],[245,246],[246,247],[247,248],[248,249],[249,250],[250,251],[251,252],[252,253],[253,254],[254,255],[255,256],[256,257],[257,258],[258,259],[259,260],[260,261],[261,262],[262,263],[263,264],[264,265],[265,266],[266,267],[267,268],[268,269],[269,270],[270,271],[271,272],[272,273],[273,274],[274,275],[275,276],[276,277],[277,278],[278,279],[279,280],[280,281],[281,282],[282,283],[283,284],[284,285],[285,286],[286,287],[287,288],[288,289],[289,290],[290,291],[291,292],[292,293],[293,294],[294,295],[295,296],[296,297],[297,298],[298,299],[299,300],[300,301],[301,302],[302,303],[303,304],[304,305],[305,306],[306,307],[307,308],[308,309],[309,310],[310,311],[311,312],[312,313],[313,314],[314,315],[315,316],[316,317],[317,318],[318,319],[319,320],[320,321],[321,322],[322,323],[323,324],[324,325],[325,326],[326,327],[327,328],[328,329],[329,330],[330,331],[331,332],[332,333],[333,334],[334,335],[335,336],[336,337],[337,338],[338,339],[339,340],[340,341],[341,342],[342,343],[343,344],[344,345],[345,346],[346,347],[347,348],[348,349],[349,350],[350,351],[351,352],[352,353],[353,354],[354,355],[355,356],[356,357],[357,358],[358,359],[359,360],[360,361],[361,362],[362,363],[363,364],[364,365],[365,366],[366,367],[367,368],[368,369],[369,370],[370,371],[371,372],[372,373],[373,374],[374,375],[375,376],[376,377],[377,378],[378,379],[379,380],[380,381],[381,382],[382,383],[383,384],[384,385],[385,386],[386,387],[387,388],[388,389],[389,390],[390,391],[391,392],[392,393],[393,394],[394,395],[395,396],[396,397],[397,398],[398,399],[399,400],[400,401],[401,402],[402,403],[403,404],[404,405],[405,406],[406,407],[407,408],[408,409],[409,410],[410,411],[411,412],[412,413],[413,414],[414,415],[415,416],[416,417],[417,418],[418,419],[419,420],[420,421],[421,422],[422,423],[423,424],[424,425],[425,426],[426,427],[427,428],[428,429],[429,430],[430,431],[431,432],[432,433],[433,434],[434,435],[435,436],[436,437],[437,438],[438,439],[439,440],[440,441],[441,442],[442,443],[443,444],[444,445],[445,446],[446,447],[447,448],[448,449],[449,450],[450,451],[451,452],[452,453],[453,454],[454,455],[455,456],[456,457],[457,458],[458,459],[459,460],[460,461],[461,462],[462,463],[463,464],[464,465],[465,466],[466,467],[467,468],[468,469],[469,470],[470,471],[471,472],[472,473],[473,474],[474,475],[475,476],[476,477],[477,478],[478,479],[479,480],[480,481],[481,482],[482,483],[483,484],[484,485],[485,486],[486,487],[487,488],[488,489],[489,490],[490,491],[491,492],[492,493],[493,494],[494,495],[495,496],[496,497],[497,498],[498,499],[499,500],[500,501],[501,502],[502,503],[503,504],[504,505],[505,506],[506,507],[507,508],[508,509],[509,510],[510,511],[511,512],[512,513],[513,514],[514,515],[515,516],[516,517],[517,518],[518,519],[519,520],[520,521],[521,522],[522,523],[523,524],[524,525],[525,526],[526,527],[527,528],[528,529],[529,530],[530,531],[531,532],[532,533],[533,534],[534,535],[535,536],[536,537],[537,538],[538,539],[539,540],[540,541],[541,542],[542,543],[543,544],[544,545],[545,546],[546,547],[547,548],[548,549],[549,550],[550,551],[551,552],[552,553],[553,554],[554,555],[555,556],[556,557],[557,558],[558,559],[559,560],[560,561],[561,562],[562,563],[563,564],[564,565],[565,566],[566,567],[567,568],[568,569],[569,570],[570,571],[571,572],[572,573],[573,574],[574,575],[575,576],[576,577],[577,578],[578,579],[579,580],[580,581],[581,582],[582,583],[583,584],[584,585],[585,586],[586,587],[587,588],[588,589],[589,590],[590,591],[591,592],[592,593],[593,594],[594,595],[595,596],[596,597],[597,598],[598,599],[599,600],[600,601],[601,602],[602,603],[603,604],[604,605],[605,606],[606,607],[607,608],[608,609],[609,610],[610,611],[611,612],[612,613],[613,614],[614,615],[615,616],[616,617],[617,618],[618,619],[619,620],[620,621],[621,622],[622,623],[623,624],[624,625],[625,626],[626,627],[627,628],[628,629],[629,630],[630,631],[631,632],[632,633],[633,634],[634,635],[635,636],[636,637],[637,638],[638,639],[639,640],[640,641],[641,642],[642,643],[643,644],[644,645],[645,646],[646,647],[647,648],[648,649],[649,650],[650,651],[651,652],[652,653],[653,654],[654,655],[655,656],[656,657],[657,658],[658,659],[659,660],[660,661],[661,662],[662,663],[663,664],[664,665],[665,666],[666,667],[667,668],[668,669],[669,670],[670,671],[671,672],[672,673],[673,674],[674,675],[675,676],[676,677],[677,678],[678,679],[679,680],[680,681],[681,682],[682,683],[683,684],[684,685],[685,686],[686,687],[687,688],[688,689],[689,690],[690,691],[691,692],[692,693],[693,694],[694,695],[695,696],[696,697],[697,698],[698,699],[699,700],[700,701],[701,702],[702,703],[703,704],[704,705],[705,706],[706,707],[707,708],[708,709],[709,710],[710,711],[711,712],[712,713],[713,714],[714,715],[715,716],[716,717],[717,718],[718,719],[719,720],[720,721],[721,722],[722,723],[723,724],[724,725],[725,726],[726,727],[727,728],[728,729],[729,730],[730,731],[731,732],[732,733],[733,734],[734,735],[735,736],[736,737],[737,738],[738,739],[739,740],[740,741],[741,742],[742,743],[743,744],[744,745],[745,746],[746,747],[747,748],[748,749],[749,750],[750,751],[751,752],[752,753],[753,754],[754,755],[755,756],[756,757],[757,758],[758,759],[759,760],[760,761],[761,762],[762,763],[763,764],[764,765],[765,766],[766,767],[767,768],[768,769],[769,770],[770,771],[771,772],[772,773],[773,774],[774,775],[775,776],[776,777],[777,778],[778,779],[779,780],[780,781],[781,782],[782,783],[783,784],[784,785],[785,786],[786,787],[787,788],[788,789],[789,790],[790,791],[791,792],[792,793],[793,794],[794,795],[795,796],[796,797],[797,798],[798,799],[799,800],[800,801],[801,802],[802,803],[803,804],[804,805],[805,806],[806,807],[807,808],[808,809],[809,810],[810,811],[811,812],[812,813],[813,814],[814,815],[815,816],[816,817],[817,818],[818,819],[819,820],[820,821],[821,822],[822,823],[823,824],[824,825],[825,826],[826,827],[827,828],[828,829],[829,830],[830,831],[831,832],[832,833],[833,834],[834,835],[835,836],[836,837],[837,838],[838,839],[839,840],[840,841],[841,842],[842,843],[843,844],[844,845],[845,846],[846,847],[847,848],[848,849],[849,850],[850,851],[851,852],[852,853],[853,854],[854,855],[855,856],[856,857],[857,858],[858,859],[859,860],[860,861],[861,862],[862,863],[863,864],[864,865],[865,866],[866,867],[867,868],[868,869],[869,870],[870,871],[871,872],[872,873],[873,874],[874,875],[875,876],[876,877],[877,878],[878,879],[879,880],[880,881],[881,882],[882,883],[883,884],[884,885],[885,886],[886,887],[887,888],[888,889],[889,890],[890,891],[891,892],[892,893],[893,894],[894,895],[895,896],[896,897],[897,898],[898,899],[899,900],[900,901],[901,902],[902,903],[903,904],[904,905],[905,906],[906,907],[907,908],[908,909],[909,910],[910,911],[911,912],[912,913],[913,914],[914,915],[915,916],[916,917],[917,918],[918,919],[919,920],[920,921],[921,922],[922,923],[923,924],[924,925],[925,926],[926,927],[927,928],[928,929],[929,930],[930,931],[931,932],[932,933],[933,934],[934,935],[935,936],[936,937],[937,938],[938,939],[939,940],[940,941],[941,942],[942,943],[943,944],[944,945],[945,946],[946,947],[947,948],[948,949],[949,950],[950,951],[951,952],[952,953],[953,954],[954,955],[955,956],[956,957],[957,958],[958,959],[959,960],[960,961],[961,962],[962,963],[963,964],[964,965],[965,966],[966,967],[967,968],[968,969],[969,970],[970,971],[971,972],[972,973],[973,974],[974,975],[975,976],[976,977],[977,978],[978,979],[979,980],[980,981],[981,982],[982,983],[983,984],[984,985],[985,986],[986,987],[987,988],[988,989],[989,990],[990,991],[991,992],[992,993],[993,994],[994,995],[995,996],[996,997],[997,998],[998,999],[999,1000],[1000,1001],[1001,1002],[1002,1003],[1003,1004],[1004,1005],[1005,1006],[1006,1007],[1007,1008],[1008,1009],[1009,1010],[1010,1011],[1011,1012],[1012,1013],[1013,1014],[1014,1015],[1015,1016],[1016,1017],[1017,1018],[1018,1019],[1019,1020],[1020,1021],[1021,1022],[1022,1023],[1023,1024],[1024,1025],[1025,1026],[1026,1027],[1027,1028],[1028,1029],[1029,1030],[1030,1031],[1031,1032],[1032,1033],[1033,1034],[1034,1035],[1035,1036],[1036,1037],[1037,1038],[1038,1039],[1039,1040],[1040,1041],[1041,1042],[1042,1043],[1043,1044],[1044,1045],[1045,1046],[1046,1047],[1047,1048],[1048,1049],[1049,1050],[1050,1051],[1051,1052],[1052,1053],[1053,1054],[1054,1055],[1055,1056],[1056,1057],[1057,1058],[1058,1059],[1059,1060],[1060,1061],[1061,1062],[1062,1063],[1063,1064],[1064,1065],[1065,1066],[1066,1067],[1067,1068],[1068,1069],[1069,1070],[1070,1071],[1071,1072],[1072,1073],[1073,1074],[1074,1075],[1075,1076],[1076,1077],[1077,1078],[1078,1079],[1079,1080],[1080,1081],[1081,1082],[1082,1083],[1083,1084],[1084,1085],[1085,1086],[1086,1087],[1087,1088],[1088,1089],[1089,1090],[1090,1091],[1091,1092],[1092,1093],[1093,1094],[1094,1095],[1095,1096],[1096,1097],[1097,1098],[1098,1099],[1099,1100],[1100,1101],[1101,1102],[1102,1103],[1103,1104],[1104,1105],[1105,1106],[1106,1107],[1107,1108],[1108,1109],[1109,1110],[1110,1111],[1111,1112],[1112,1113],[1113,1114],[1114,1115],[1115,1116],[1116,1117],[1117,1118],[1118,1119],[1119,1120],[1120,1121],[1121,1122],[1122,1123],[1123,1124],[1124,1125],[1125,1126],[1126,1127],[1127,1128],[1128,1129],[1129,1130],[1130,1131],[1131,1132],[1132,1133],[1133,1134],[1134,1135],[1135,1136],[1136,1137],[1137,1138],[1138,1139],[1139,1140],[1140,1141],[1141,1142],[1142,1143],[1143,1144],[1144,1145],[1145,1146],[1146,1147],[1147,1148],[1148,1149],[1149,1150],[1150,1151],[1151,1152],[1152,1153],[1153,1154],[1154,1155],[1155,1156],[1156,1157],[1157,1158],[1158,1159],[1159,1160],[1160,1161],[1161,1162],[1162,1163],[1163,1164],[1164,1165],[1165,1166],[1166,1167],[1167,1168],[1168,1169],[1169,1170],[1170,1171],[1171,1172],[1172,1173],[1173,1174],[1174,1175],[1175,1176],[1176,1177],[1177,1178],[1178,1179],[1179,1180],[1180,1181],[1181,1182],[1182,1183],[1183,1184],[1184,1185],[1185,1186],[1186,1187],[1187,1188],[1188,1189],[1189,1190],[1190,1191],[1191,1192],[1192,1193],[1193,1194],[1194,1195],[1195,1196],[1196,1197],[1197,1198],[1198,1199],[1199,1200],[1200,1201],[1201,1202],[1202,1203],[1203,1204],[1204,1205],[1205,1206],[1206,1207],[1207,1208],[1208,1209],[1209,1210],[1210,1211],[1211,1212],[1212,1213],[1213,1214],[1214,1215],[1215,1216],[1216,1217],[1217,1218],[1218,1219],[1219,1220],[1220,1221],[1221,1222],[1222,1223],[1223,1224],[1224,1225],[1225,1226],[1226,1227],[1227,1228],[1228,1229],[1229,1230],[1230,1231],[1231,1232],[1232,1233],[1233,1234],[1234,1235],[1235,1236],[1236,1237],[1237,1238],[1238,1239],[1239,1240],[1240,1241],[1241,1242],[1242,1243],[1243,1244],[1244,1245],[1245,1246],[1246,1247],[1247,1248],[1248,1249],[1249,1250],[1250,1251],[1251,1252],[1252,1253],[1253,1254],[1254,1255],[1255,1256],[1256,1257],[1257,1258],[1258,1259],[1259,1260],[1260,1261],[1261,1262],[1262,1263],[1263,1264],[1264,1265],[1265,1266],[1266,1267],[1267,1268],[1268,1269],[1269,1270],[1270,1271],[1271,1272],[1272,1273],[1273,1274],[1274,1275],[1275,1276],[1276,1277],[1277,1278],[1278,1279],[1279,1280],[1280,1281],[1281,1282],[1282,1283],[1283,1284],[1284,1285],[1285,1286],[1286,1287],[1287,1288],[1288,1289],[1289,1290],[1290,1291],[1291,1292],[1292,1293],[1293,1294],[1294,1295],[1295,1296],[1296,1297],[1297,1298],[1298,1299],[1299,1300],[1300,1301],[1301,1302],[1302,1303],[1303,1304],[1304,1305],[1305,1306],[1306,1307],[1307,1308],[1308,1309],[1309,1310],[1310,1311],[1311,1312],[1312,1313],[1313,1314],[1314,1315],[1315,1316],[1316,1317],[1317,1318],[1318,1319],[1319,1320],[1320,1321],[1321,1322],[1322,1323],[1323,1324],[1324,1325],[1325,1326],[1326,1327],[1327,1328],[1328,1329],[1329,1330],[1330,1331],[1331,1332],[1332,1333],[1333,1334],[1334,1335],[1335,1336],[1336,1337],[1337,1338],[1338,1339],[1339,1340],[1340,1341],[1341,1342],[1342,1343],[1343,1344],[1344,1345],[1345,1346],[1346,1347],[1347,1348],[1348,1349],[1349,1350],[1350,1351],[1351,1352],[1352,1353],[1353,1354],[1354,1355],[1355,1356],[1356,1357],[1357,1358],[1358,1359],[1359,1360],[1360,1361],[1361,1362],[1362,1363],[1363,1364],[1364,1365],[1365,1366],[1366,1367],[1367,1368],[1368,1369],[1369,1370],[1370,1371],[1371,1372],[1372,1373],[1373,1374],[1374,1375],[1375,1376],[1376,1377],[1377,1378],[1378,1379],[1379,1380],[1380,1381],[1381,1382],[1382,1383],[1383,1384],[1384,1385],[1385,1386],[1386,1387],[1387,1388],[1388,1389],[1389,1390],[1390,1391],[1391,1392],[1392,1393],[1393,1394],[1394,1395],[1395,1396],[1396,1397],[1397,1398],[1398,1399],[1399,1400],[1400,1401],[1401,1402],[1402,1403],[1403,1404],[1404,1405],[1405,1406],[1406,1407],[1407,1408],[1408,1409],[1409,1410],[1410,1411],[1411,1412],[1412,1413],[1413,1414],[1414,1415],[1415,1416],[1416,1417],[1417,1418],[1418,1419],[1419,1420],[1420,1421],[1421,1422],[1422,1423],[1423,1424],[1424,1425],[1425,1426],[1426,1427],[1427,1428],[1428,1429],[1429,1430],[1430,1431],[1431,1432],[1432,1433],[1433,1434],[1434,1435],[1435,1436],[1436,1437],[1437,1438],[1438,1439],[1439,1440],[1440,1441],[1441,1442],[1442,1443],[1443,1444],[1444,1445],[1445,1446],[1446,1447],[1447,1448],[1448,1449],[1449,1450],[1450,1451],[1451,1452],[1452,1453],[1453,1454],[1454,1455],[1455,1456],[1456,1457],[1457,1458],[1458,1459],[1459,1460],[1460,1461],[1461,1462],[1462,1463],[1463,1464],[1464,1465],[1465,1466],[1466,1467],[1467,1468],[1468,1469],[1469,1470],[1470,1471],[1471,1472],[1472,1473],[1473,1474],[1474,1475],[1475,1476],[1476,1477],[1477,1478],[1478,1479],[1479,1480],[1480,1481],[1481,1482],[1482,1483],[1483,1484],[1484,1485],[1485,1486],[1486,1487],[1487,1488],[1488,1489],[1489,1490],[1490,1491],[1491,1492],[1492,1493],[1493,1494],[1494,1495],[1495,1496],[1496,1497],[1497,1498],[1498,1499],[1499,1500],[1500,1501],[1501,1502],[1502,1503],[1503,1504],[1504,1505],[1505,1506],[1506,1507],[1507,1508],[1508,1509],[1509,1510],[1510,1511],[1511,1512],[1512,1513],[1513,1514],[1514,1515],[1515,1516],[1516,1517],[1517,1518],[1518,1519],[1519,1520],[1520,1521],[1521,1522],[1522,1523],[1523,1524],[1524,1525],[1525,1526],[1526,1527],[1527,1528],[1528,1529],[1529,1530],[1530,1531],[1531,1532],[1532,1533],[1533,1534],[1534,1535],[1535,1536],[1536,1537],[1537,1538],[1538,1539],[1539,1540],[1540,1541],[1541,1542],[1542,1543],[1543,1544],[1544,1545],[1545,1546],[1546,1547],[1547,1548],[1548,1549],[1549,1550],[1550,1551],[1551,1552],[1552,1553],[1553,1554],[1554,1555],[1555,1556],[1556,1557],[1557,1558],[1558,1559],[1559,1560],[1560,1561],[1561,1562],[1562,1563],[1563,1564],[1564,1565],[1565,1566],[1566,1567],[1567,1568],[1568,1569],[1569,1570],[1570,1571],[1571,1572],[1572,1573],[1573,1574],[1574,1575],[1575,1576],[1576,1577],[1577,1578],[1578,1579],[1579,1580],[1580,1581],[1581,1582],[1582,1583],[1583,1584],[1584,1585],[1585,1586],[1586,1587],[1587,1588],[1588,1589],[1589,1590],[1590,1591],[1591,1592],[1592,1593],[1593,1594],[1594,1595],[1595,1596],[1596,1597],[1597,1598],[1598,1599],[1599,1600],[1600,1601],[1601,1602],[1602,1603],[1603,1604],[1604,1605],[1605,1606],[1606,1607],[1607,1608],[1608,1609],[1609,1610],[1610,1611],[1611,1612],[1612,1613],[1613,1614],[1614,1615],[1615,1616],[1616,1617],[1617,1618],[1618,1619],[1619,1620],[1620,1621],[1621,1622],[1622,1623],[1623,1624],[1624,1625],[1625,1626],[1626,1627],[1627,1628],[1628,1629],[1629,1630],[1630,1631],[1631,1632],[1632,1633],[1633,1634],[1634,1635],[1635,1636],[1636,1637],[1637,1638],[1638,1639],[1639,1640],[1640,1641],[1641,1642],[1642,1643],[1643,1644],[1644,1645],[1645,1646],[1646,1647],[1647,1648],[1648,1649],[1649,1650],[1650,1651],[1651,1652],[1652,1653],[1653,1654],[1654,1655],[1655,1656],[1656,1657],[1657,1658],[1658,1659],[1659,1660],[1660,1661],[1661,1662],[1662,1663],[1663,1664],[1664,1665],[1665,1666],[1666,1667],[1667,1668],[1668,1669],[1669,1670],[1670,1671],[1671,1672],[1672,1673],[1673,1674],[1674,1675],[1675,1676],[1676,1677],[1677,1678],[1678,1679],[1679,1680],[1680,1681],[1681,1682],[1682,1683],[1683,1684],[1684,1685],[1685,1686],[1686,1687],[1687,1688],[1688,1689],[1689,1690],[1690,1691],[1691,1692],[1692,1693],[1693,1694],[1694,1695],[1695,1696],[1696,1697],[1697,1698],[1698,1699],[1699,1700],[1700,1701],[1701,1702],[1702,1703],[1703,1704],[1704,1705],[1705,1706],[1706,1707],[1707,1708],[1708,1709],[1709,1710],[1710,1711],[1711,1712],[1712,1713],[1713,1714],[1714,1715],[1715,1716],[1716,1717],[1717,1718],[1718,1719],[1719,1720],[1720,1721],[1721,1722],[1722,1723],[1723,1724],[1724,1725],[1725,1726],[1726,1727],[1727,1728],[1728,1729],[1729,1730],[1730,1731],[1731,1732],[1732,1733],[1733,1734],[1734,1735],[1735,1736],[1736,1737],[1737,1738],[1738,1739],[1739,1740],[1740,1741],[1741,1742],[1742,1743],[1743,1744],[1744,1745],[1745,1746],[1746,1747],[1747,1748],[1748,1749],[1749,1750],[1750,1751],[1751,1752],[1752,1753],[1753,1754],[1754,1755],[1755,1756],[1756,1757],[1757,1758],[1758,1759],[1759,1760],[1760,1761],[1761,1762],[1762,1763],[1763,1764],[1764,1765],[1765,1766],[1766,1767],[1767,1768],[1768,1769],[1769,1770],[1770,1771],[1771,1772],[1772,1773],[1773,1774],[1774,1775],[1775,1776],[1776,1777],[1777,1778],[1778,1779],[1779,1780],[1780,1781],[1781,1782],[1782,1783],[1783,1784],[1784,1785],[1785,1786],[1786,1787],[1787,1788],[1788,1789],[1789,1790],[1790,1791],[1791,1792],[1792,1793],[1793,1794],[1794,1795],[1795,1796],[1796,1797],[1797,1798],[1798,1799],[1799,1800],[1800,1801],[1801,1802],[1802,1803],[1803,1804],[1804,1805],[1805,1806],[1806,1807],[1807,1808],[1808,1809],[1809,1810],[1810,1811],[1811,1812],[1812,1813],[1813,1814],[1814,1815],[1815,1816],[1816,1817],[1817,1818],[1818,1819],[1819,1820],[1820,1821],[1821,1822],[1822,1823],[1823,1824],[1824,1825],[1825,1826],[1826,1827],[1827,1828],[1828,1829],[1829,1830],[1830,1831],[1831,1832],[1832,1833],[1833,1834],[1834,1835],[1835,1836],[1836,1837],[1837,1838],[1838,1839],[1839,1840],[1840,1841],[1841,1842],[1842,1843],[1843,1844],[1844,1845],[1845,1846],[1846,1847],[1847,1848],[1848,1849],[1849,1850],[1850,1851],[1851,1852],[1852,1853],[1853,1854],[1854,1855],[1855,1856],[1856,1857],[1857,1858],[1858,1859],[1859,1860],[1860,1861],[1861,1862],[1862,1863],[1863,1864],[1864,1865],[1865,1866],[1866,1867],[1867,1868],[1868,1869],[1869,1870],[1870,1871],[1871,1872],[1872,1873],[1873,1874],[1874,1875],[1875,1876],[1876,1877],[1877,1878],[1878,1879],[1879,1880],[1880,1881],[1881,1882],[1882,1883],[1883,1884],[1884,1885],[1885,1886],[1886,1887],[1887,1888],[1888,1889],[1889,1890],[1890,1891],[1891,1892],[1892,1893],[1893,1894],[1894,1895],[1895,1896],[1896,1897],[1897,1898],[1898,1899],[1899,1900],[1900,1901],[1901,1902],[1902,1903],[1903,1904],[1904,1905],[1905,1906],[1906,1907],[1907,1908],[1908,1909],[1909,1910],[1910,1911],[1911,1912],[1912,1913],[1913,1914],[1914,1915],[1915,1916],[1916,1917],[1917,1918],[1918,1919],[1919,1920],[1920,1921],[1921,1922],[1922,1923],[1923,1924],[1924,1925],[1925,1926],[1926,1927],[1927,1928],[1928,1929],[1929,1930],[1930,1931],[1931,1932],[1932,1933],[1933,1934],[1934,1935],[1935,1936],[1936,1937],[1937,1938],[1938,1939],[1939,1940],[1940,1941],[1941,1942],[1942,1943],[1943,1944],[1944,1945],[1945,1946],[1946,1947],[1947,1948],[1948,1949],[1949,1950],[1950,1951],[1951,1952],[1952,1953],[1953,1954],[1954,1955],[1955,1956],[1956,1957],[1957,1958],[1958,1959],[1959,1960],[1960,1961],[1961,1962],[1962,1963],[1963,1964],[1964,1965],[1965,1966],[1966,1967],[1967,1968],[1968,1969],[1969,1970],[1970,1971],[1971,1972],[1972,1973],[1973,1974],[1974,1975],[1975,1976],[1976,1977],[1977,1978],[1978,1979],[1979,1980],[1980,1981],[1981,1982],[1982,1983],[1983,1984],[1984,1985],[1985,1986],[1986,1987],[1987,1988],[1988,1989],[1989,1990],[1990,1991],[1991,1992],[1992,1993],[1993,1994],[1994,1995],[1995,1996],[1996,1997],[1997,1998],[1998,1999],[1999,2000],[2000,2001],[2001,2002],[2002,2003],[2003,2004],[2004,2005],[2005,2006],[2006,2007],[2007,2008],[2008,2009],[2009,2010],[2010,2011],[2011,2012],[2012,2013],[2013,2014],[2014,2015],[2015,2016],[2016,2017],[2017,2018],[2018,2019],[2019,2020],[2020,2021],[2021,2022],[2022,2023],[2023,2024],[2024,2025],[2025,2026],[2026,2027],[2027,2028],[2028,2029],[2029,2030],[2030,2031],[2031,2032],[2032,2033],[2033,2034],[2034,2035],[2035,2036],[2036,2037],[2037,2038],[2038,2039],[2039,2040],[2040,2041],[2041,2042],[2042,2043],[2043,2044],[2044,2045],[2045,2046],[2046,2047],[2047,2048],[2048,2049],[2049,2050],[2050,2051],[2051,2052],[2052,2053],[2053,2054],[2054,2055],[2055,2056],[2056,2057],[2057,2058],[2058,2059],[2059,2060],[2060,2061],[2061,2062],[2062,2063],[2063,2064],[2064,2065],[2065,2066],[2066,2067],[2067,2068],[2068,2069],[2069,2070],[2070,2071],[2071,2072],[2072,2073],[2073,2074],[2074,2075],[2075,2076],[2076,2077],[2077,2078],[2078,2079],[2079,2080],[2080,2081],[2081,2082],[2082,2083],[2083,2084],[2084,2085],[2085,2086],[2086,2087],[2087,2088],[2088,2089],[2089,2090],[2090,2091],[2091,2092],[2092,2093],[2093,2094],[2094,2095],[2095,2096],[2096,2097],[2097,2098],[2098,2099],[2099,2100],[2100,2101],[2101,2102],[2102,2103],[2103,2104],[2104,2105],[2105,2106],[2106,2107],[2107,2108],[2108,2109],[2109,2110],[2110,2111],[2111,2112],[2112,2113],[2113,2114],[2114,2115],[2115,2116],[2116,2117],[2117,2118],[2118,2119],[2119,2120],[2120,2121],[2121,2122],[2122,2123],[2123,2124],[2124,2125],[2125,2126],[2126,2127],[2127,2128],[2128,2129],[2129,2130],[2130,2131],[2131,2132],[2132,2133],[2133,2134],[2134,2135],[2135,2136],[2136,2137],[2137,2138],[2138,2139],[2139,2140],[2140,2141],[2141,2142],[2142,2143],[2143,2144],[2144,2145],[2145,2146],[2146,2147],[2147,2148],[2148,2149],[2149,2150],[2150,2151],[2151,2152],[2152,2153],[2153,2154],[2154,2155],[2155,2156],[2156,2157],[2157,2158],[2158,2159],[2159,2160],[2160,2161],[2161,2162],[2162,2163],[2163,2164],[2164,2165],[2165,2166],[2166,2167],[2167,2168],[2168,2169],[2169,2170],[2170,2171],[2171,2172],[2172,2173],[2173,2174],[2174,2175],[2175,2176],[2176,2177],[2177,2178],[2178,2179],[2179,2180],[2180,2181],[2181,2182],[2182,2183],[2183,2184],[2184,2185],[2185,2186],[2186,2187],[2187,2188],[2188,2189],[2189,2190],[2190,2191],[2191,2192],[2192,2193],[2193,2194],[2194,2195],[2195,2196],[2196,2197],[2197,2198],[2198,2199],[2199,2200],[2200,2201],[2201,2202],[2202,2203],[2203,2204],[2204,2205],[2205,2206],[2206,2207],[2207,2208],[2208,2209],[2209,2210],[2210,2211],[2211,2212],[2212,2213],[2213,2214],[2214,2215],[2215,2216],[2216,2217],[2217,2218],[2218,2219],[2219,2220],[2220,2221],[2221,2222],[2222,2223],[2223,2224],[2224,2225],[2225,2226],[2226,2227],[2227,2228],[2228,2229],[2229,2230],[2230,2231],[2231,2232],[2232,2233],[2233,2234],[2234,2235],[2235,2236],[2236,2237],[2237,2238],[2238,2239],[2239,2240],[2240,2241],[2241,2242],[2242,2243],[2243,2244],[2244,2245],[2245,2246],[2246,2247],[2247,2248],[2248,2249],[2249,2250],[2250,2251],[2251,2252],[2252,2253],[2253,2254],[2254,2255],[2255,2256],[2256,2257],[2257,2258],[2258,2259],[2259,2260],[2260,2261],[2261,2262],[2262,2263],[2263,2264],[2264,2265],[2265,2266],[2266,2267],[2267,2268],[2268,2269],[2269,2270],[2270,2271],[2271,2272],[2272,2273],[2273,2274],[2274,2275],[2275,2276],[2276,2277],[2277,2278],[2278,2279],[2279,2280],[2280,2281],[2281,2282],[2282,2283],[2283,2284],[2284,2285],[2285,2286],[2286,2287],[2287,2288],[2288,2289],[2289,2290],[2290,2291],[2291,2292],[2292,2293],[2293,2294],[2294,2295],[2295,2296],[2296,2297],[2297,2298],[2298,2299],[2299,2300],[2300,2301],[2301,2302],[2302,2303],[2303,2304],[2304,2305],[2305,2306],[2306,2307],[2307,2308],[2308,2309],[2309,2310],[2310,2311],[2311,2312],[2312,2313],[2313,2314],[2314,2315],[2315,2316],[2316,2317],[2317,2318],[2318,2319],[2319,2320],[2320,2321],[2321,2322],[2322,2323],[2323,2324],[2324,2325],[2325,2326],[2326,2327],[2327,2328],[2328,2329],[2329,2330],[2330,2331],[2331,2332],[2332,2333],[2333,2334],[2334,2335],[2335,2336],[2336,2337],[2337,2338],[2338,2339],[2339,2340],[2340,2341],[2341,2342],[2342,2343],[2343,2344],[2344,2345],[2345,2346],[2346,2347],[2347,2348],[2348,2349],[2349,2350],[2350,2351],[2351,2352],[2352,2353],[2353,2354],[2354,2355],[2355,2356],[2356,2357],[2357,2358],[2358,2359],[2359,2360],[2360,2361],[2361,2362],[2362,2363],[2363,2364],[2364,2365],[2365,2366],[2366,2367],[2367,2368],[2368,2369],[2369,2370],[2370,2371],[2371,2372],[2372,2373],[2373,2374],[2374,2375],[2375,2376],[2376,2377],[2377,2378],[2378,2379],[2379,2380],[2380,2381],[2381,2382],[2382,2383],[2383,2384],[2384,2385],[2385,2386],[2386,2387],[2387,2388],[2388,2389],[2389,2390],[2390,2391],[2391,2392],[2392,2393],[2393,2394],[2394,2395],[2395,2396],[2396,2397],[2397,2398],[2398,2399],[2399,2400],[2400,2401],[2401,2402],[2402,2403],[2403,2404],[2404,2405],[2405,2406],[2406,2407],[2407,2408],[2408,2409],[2409,2410],[2410,2411],[2411,2412],[2412,2413],[2413,2414],[2414,2415],[2415,2416],[2416,2417],[2417,2418],[2418,2419],[2419,2420],[2420,2421],[2421,2422],[2422,2423],[2423,2424],[2424,2425],[2425,2426],[2426,2427],[2427,2428],[2428,2429],[2429,2430],[2430,2431],[2431,2432],[2432,2433],[2433,2434],[2434,2435],[2435,2436],[2436,2437],[2437,2438],[2438,2439],[2439,2440],[2440,2441],[2441,2442],[2442,2443],[2443,2444],[2444,2445],[2445,2446],[2446,2447],[2447,2448],[2448,2449],[2449,2450],[2450,2451],[2451,2452],[2452,2453],[2453,2454],[2454,2455],[2455,2456],[2456,2457],[2457,2458],[2458,2459],[2459,2460],[2460,2461],[2461,2462],[2462,2463],[2463,2464],[2464,2465],[2465,2466],[2466,2467],[2467,2468],[2468,2469],[2469,2470],[2470,2471],[2471,2472],[2472,2473],[2473,2474],[2474,2475],[2475,2476],[2476,2477],[2477,2478],[2478,2479],[2479,2480],[2480,2481],[2481,2482],[2482,2483],[2483,2484],[2484,2485],[2485,2486],[2486,2487],[2487,2488],[2488,2489],[2489,2490],[2490,2491],[2491,2492],[2492,2493],[2493,2494],[2494,2495],[2495,2496],[2496,2497],[2497,2498],[2498,2499],[2499,2500],[2500,2501],[2501,2502],[2502,2503],[2503,2504],[2504,2505],[2505,2506],[2506,2507],[2507,2508],[2508,2509],[2509,2510],[2510,2511],[2511,2512],[2512,2513],[2513,2514],[2514,2515],[2515,2516],[2516,2517],[2517,2518],[2518,2519],[2519,2520],[2520,2521],[2521,2522],[2522,2523],[2523,2524],[2524,2525],[2525,2526],[2526,2527],[2527,2528],[2528,2529],[2529,2530],[2530,2531],[2531,2532],[2532,2533],[2533,2534],[2534,2535],[2535,2536],[2536,2537],[2537,2538],[2538,2539],[2539,2540],[2540,2541],[2541,2542],[2542,2543],[2543,2544],[2544,2545],[2545,2546],[2546,2547],[2547,2548],[2548,2549],[2549,2550],[2550,2551],[2551,2552],[2552,2553],[2553,2554],[2554,2555],[2555,2556],[2556,2557],[2557,2558],[2558,2559],[2559,2560],[2560,2561],[2561,2562],[2562,2563],[2563,2564],[2564,2565],[2565,2566],[2566,2567],[2567,2568],[2568,2569],[2569,2570],[2570,2571],[2571,2572],[2572,2573],[2573,2574],[2574,2575],[2575,2576],[2576,2577],[2577,2578],[2578,2579],[2579,2580],[2580,2581],[2581,2582],[2582,2583],[2583,2584],[2584,2585],[2585,2586],[2586,2587],[2587,2588],[2588,2589],[2589,2590],[2590,2591],[2591,2592],[2592,2593],[2593,2594],[2594,2595],[2595,2596],[2596,2597],[2597,2598],[2598,2599],[2599,2600],[2600,2601],[2601,2602],[2602,2603],[2603,2604],[2604,2605],[2605,2606],[2606,2607],[2607,2608],[2608,2609],[2609,2610],[2610,2611],[2611,2612],[2612,2613],[2613,2614],[2614,2615],[2615,2616],[2616,2617],[2617,2618],[2618,2619],[2619,2620],[2620,2621],[2621,2622],[2622,2623],[2623,2624],[2624,2625],[2625,2626],[2626,2627],[2627,2628],[2628,2629],[2629,2630],[2630,2631],[2631,2632],[2632,2633],[2633,2634],[2634,2635],[2635,2636],[2636,2637],[2637,2638],[2638,2639],[2639,2640],[2640,2641],[2641,2642],[2642,2643],[2643,2644],[2644,2645],[2645,2646],[2646,2647],[2647,2648],[2648,2649],[2649,2650],[2650,2651],[2651,2652],[2652,2653],[2653,2654],[2654,2655],[2655,2656],[2656,2657],[2657,2658],[2658,2659],[2659,2660],[2660,2661],[2661,2662],[2662,2663],[2663,2664],[2664,2665],[2665,2666],[2666,2667],[2667,2668],[2668,2669],[2669,2670],[2670,2671],[2671,2672],[2672,2673],[2673,2674],[2674,2675],[2675,2676],[2676,2677],[2677,2678],[2678,2679],[2679,2680],[2680,2681],[2681,2682],[2682,2683],[2683,2684],[2684,2685],[2685,2686],[2686,2687],[2687,2688],[2688,2689],[2689,2690],[2690,2691],[2691,2692],[2692,2693],[2693,2694],[2694,2695],[2695,2696],[2696,2697],[2697,2698],[2698,2699],[2699,2700],[2700,2701],[2701,2702],[2702,2703],[2703,2704],[2704,2705],[2705,2706],[2706,2707],[2707,2708],[2708,2709],[2709,2710],[2710,2711],[2711,2712],[2712,2713],[2713,2714],[2714,2715],[2715,2716],[2716,2717],[2717,2718],[2718,2719],[2719,2720],[2720,2721],[2721,2722],[2722,2723],[2723,2724],[2724,2725],[2725,2726],[2726,2727],[2727,2728],[2728,2729],[2729,2730],[2730,2731],[2731,2732],[2732,2733],[2733,2734],[2734,2735],[2735,2736],[2736,2737],[2737,2738],[2738,2739],[2739,2740],[2740,2741],[2741,2742],[2742,2743],[2743,2744],[2744,2745],[2745,2746],[2746,2747],[2747,2748],[2748,2749],[2749,2750],[2750,2751],[2751,2752],[2752,2753],[2753,2754],[2754,2755],[2755,2756],[2756,2757],[2757,2758],[2758,2759],[2759,2760],[2760,2761],[2761,2762],[2762,2763],[2763,2764],[2764,2765],[2765,2766],[2766,2767],[2767,2768],[2768,2769],[2769,2770],[2770,2771],[2771,2772],[2772,2773],[2773,2774],[2774,2775],[2775,2776],[2776,2777],[2777,2778],[2778,2779],[2779,2780],[2780,2781],[2781,2782],[2782,2783],[2783,2784],[2784,2785],[2785,2786],[2786,2787],[2787,2788],[2788,2789],[2789,2790],[2790,2791],[2791,2792],[2792,2793],[2793,2794],[2794,2795],[2795,2796],[2796,2797],[2797,2798],[2798,2799],[2799,2800],[2800,2801],[2801,2802],[2802,2803],[2803,2804],[2804,2805],[2805,2806],[2806,2807],[2807,2808],[2808,2809],[2809,2810],[2810,2811],[2811,2812],[2812,2813],[2813,2814],[2814,2815],[2815,2816],[2816,2817],[2817,2818],[2818,2819],[2819,2820],[2820,2821],[2821,2822],[2822,2823],[2823,2824],[2824,2825],[2825,2826],[2826,2827],[2827,2828],[2828,2829],[2829,2830],[2830,2831],[2831,2832],[2832,2833],[2833,2834],[2834,2835],[2835,2836],[2836,2837],[2837,2838],[2838,2839],[2839,2840],[2840,2841],[2841,2842],[2842,2843],[2843,2844],[2844,2845],[2845,2846],[2846,2847],[2847,2848],[2848,2849],[2849,2850],[2850,2851],[2851,2852],[2852,2853],[2853,2854],[2854,2855],[2855,2856],[2856,2857],[2857,2858],[2858,2859],[2859,2860],[2860,2861],[2861,2862],[2862,2863],[2863,2864],[2864,2865],[2865,2866],[2866,2867],[2867,2868],[2868,2869],[2869,2870],[2870,2871],[2871,2872],[2872,2873],[2873,2874],[2874,2875],[2875,2876],[2876,2877],[2877,2878],[2878,2879],[2879,2880],[2880,2881],[2881,2882],[2882,2883],[2883,2884],[2884,2885],[2885,2886],[2886,2887],[2887,2888],[2888,2889],[2889,2890],[2890,2891],[2891,2892],[2892,2893],[2893,2894],[2894,2895],[2895,2896],[2896,2897],[2897,2898],[2898,2899],[2899,2900],[2900,2901],[2901,2902],[2902,2903],[2903,2904],[2904,2905],[2905,2906],[2906,2907],[2907,2908],[2908,2909],[2909,2910],[2910,2911],[2911,2912],[2912,2913],[2913,2914],[2914,2915],[2915,2916],[2916,2917],[2917,2918],[2918,2919],[2919,2920],[2920,2921],[2921,2922],[2922,2923],[2923,2924],[2924,2925],[2925,2926],[2926,2927],[2927,2928],[2928,2929],[2929,2930],[2930,2931],[2931,2932],[2932,2933],[2933,2934],[2934,2935],[2935,2936],[2936,2937],[2937,2938],[2938,2939],[2939,2940],[2940,2941],[2941,2942],[2942,2943],[2943,2944],[2944,2945],[2945,2946],[2946,2947],[2947,2948],[2948,2949],[2949,2950],[2950,2951],[2951,2952],[2952,2953],[2953,2954],[2954,2955],[2955,2956],[2956,2957],[2957,2958],[2958,2959],[2959,2960],[2960,2961],[2961,2962],[2962,2963],[2963,2964],[2964,2965],[2965,2966],[2966,2967],[2967,2968],[2968,2969],[2969,2970],[2970,2971],[2971,2972],[2972,2973],[2973,2974],[2974,2975],[2975,2976],[2976,2977],[2977,2978],[2978,2979],[2979,2980],[2980,2981],[2981,2982],[2982,2983],[2983,2984],[2984,2985],[2985,2986],[2986,2987],[2987,2988],[2988,2989],[2989,2990],[2990,2991],[2991,2992],[2992,2993],[2993,2994],[2994,2995],[2995,2996],[2996,2997],[2997,2998],[2998,2999],[2999,3000],[3000,3001],[3001,3002],[3002,3003],[3003,3004],[3004,3005],[3005,3006],[3006,3007],[3007,3008],[3008,3009],[3009,3010],[3010,3011],[3011,3012],[3012,3013],[3013,3014],[3014,3015],[3015,3016],[3016,3017],[3017,3018],[3018,3019],[3019,3020],[3020,3021],[3021,3022],[3022,3023],[3023,3024],[3024,3025],[3025,3026],[3026,3027],[3027,3028],[3028,3029],[3029,3030],[3030,3031],[3031,3032],[3032,3033],[3033,3034],[3034,3035],[3035,3036],[3036,3037],[3037,3038],[3038,3039],[3039,3040],[3040,3041],[3041,3042],[3042,3043],[3043,3044],[3044,3045],[3045,3046],[3046,3047],[3047,3048],[3048,3049],[3049,3050],[3050,3051],[3051,3052],[3052,3053],[3053,3054],[3054,3055],[3055,3056],[3056,3057],[3057,3058],[3058,3059],[3059,3060],[3060,3061],[3061,3062],[3062,3063],[3063,3064],[3064,3065],[3065,3066],[3066,3067],[3067,3068],[3068,3069],[3069,3070],[3070,3071],[3071,3072],[3072,3073],[3073,3074],[3074,3075],[3075,3076],[3076,3077],[3077,3078],[3078,3079],[3079,3080],[3080,3081],[3081,3082],[3082,3083],[3083,3084],[3084,3085],[3085,3086],[3086,3087],[3087,3088],[3088,3089],[3089,3090],[3090,3091],[3091,3092],[3092,3093],[3093,3094],[3094,3095],[3095,3096],[3096,3097],[3097,3098],[3098,3099],[3099,3100],[3100,3101],[3101,3102],[3102,3103],[3103,3104],[3104,3105],[3105,3106],[3106,3107],[3107,3108],[3108,3109],[3109,3110],[3110,3111],[3111,3112],[3112,3113],[3113,3114],[3114,3115],[3115,3116],[3116,3117],[3117,3118],[3118,3119],[3119,3120],[3120,3121],[3121,3122],[3122,3123],[3123,3124],[3124,3125],[3125,3126],[3126,3127],[3127,3128],[3128,3129],[3129,3130],[3130,3131],[3131,3132],[3132,3133],[3133,3134],[3134,3135],[3135,3136],[3136,3137],[3137,3138],[3138,3139],[3139,3140],[3140,3141],[3141,3142],[3142,3143],[3143,3144],[3144,3145],[3145,3146],[3146,3147],[3147,3148],[3148,3149],[3149,3150],[3150,3151],[3151,3152],[3152,3153],[3153,3154],[3154,3155],[3155,3156],[3156,3157],[3157,3158],[3158,3159],[3159,3160],[3160,3161],[3161,3162],[3162,3163],[3163,3164],[3164,3165],[3165,3166],[3166,3167],[3167,3168],[3168,3169],[3169,3170],[3170,3171],[3171,3172],[3172,3173],[3173,3174],[3174,3175],[3175,3176],[3176,3177],[3177,3178],[3178,3179],[3179,3180],[3180,3181],[3181,3182],[3182,3183],[3183,3184],[3184,3185],[3185,3186],[3186,3187],[3187,3188],[3188,3189],[3189,3190],[3190,3191],[3191,3192],[3192,3193],[3193,3194],[3194,3195],[3195,3196],[3196,3197],[3197,3198],[3198,3199],[3199,3200],[3200,3201],[3201,3202],[3202,3203],[3203,3204],[3204,3205],[3205,3206],[3206,3207],[3207,3208],[3208,3209],[3209,3210],[3210,3211],[3211,3212],[3212,3213],[3213,3214],[3214,3215],[3215,3216],[3216,3217],[3217,3218],[3218,3219],[3219,3220],[3220,3221],[3221,3222],[3222,3223],[3223,3224],[3224,3225],[3225,3226],[3226,3227],[3227,3228],[3228,3229],[3229,3230],[3230,3231],[3231,3232],[3232,3233],[3233,3234],[3234,3235],[3235,3236],[3236,3237],[3237,3238],[3238,3239],[3239,3240],[3240,3241],[3241,3242],[3242,3243],[3243,3244],[3244,3245],[3245,3246],[3246,3247],[3247,3248],[3248,3249],[3249,3250],[3250,3251],[3251,3252],[3252,3253],[3253,3254],[3254,3255],[3255,3256],[3256,3257],[3257,3258],[3258,3259],[3259,3260],[3260,3261],[3261,3262],[3262,3263],[3263,3264],[3264,3265],[3265,3266],[3266,3267],[3267,3268],[3268,3269],[3269,3270],[3270,3271],[3271,3272],[3272,3273],[3273,3274],[3274,3275],[3275,3276],[3276,3277],[3277,3278],[3278,3279],[3279,3280],[3280,3281],[3281,3282],[3282,3283],[3283,3284],[3284,3285],[3285,3286],[3286,3287],[3287,3288],[3288,3289],[3289,3290],[3290,3291],[3291,3292],[3292,3293],[3293,3294],[3294,3295],[3295,3296],[3296,3297],[3297,3298],[3298,3299],[3299,3300],[3300,3301],[3301,3302],[3302,3303],[3303,3304],[3304,3305],[3305,3306],[3306,3307],[3307,3308],[3308,3309],[3309,3310],[3310,3311],[3311,3312],[3312,3313],[3313,3314],[3314,3315],[3315,3316],[3316,3317],[3317,3318],[3318,3319],[3319,3320],[3320,3321],[3321,3322],[3322,3323],[3323,3324],[3324,3325],[3325,3326],[3326,3327],[3327,3328],[3328,3329],[3329,3330],[3330,3331],[3331,3332],[3332,3333],[3333,3334],[3334,3335],[3335,3336],[3336,3337],[3337,3338],[3338,3339],[3339,3340],[3340,3341],[3341,3342],[3342,3343],[3343,3344],[3344,3345],[3345,3346],[3346,3347],[3347,3348],[3348,3349],[3349,3350],[3350,3351],[3351,3352],[3352,3353],[3353,3354],[3354,3355],[3355,3356],[3356,3357],[3357,3358],[3358,3359],[3359,3360],[3360,3361],[3361,3362],[3362,3363],[3363,3364],[3364,3365],[3365,3366],[3366,3367],[3367,3368],[3368,3369],[3369,3370],[3370,3371],[3371,3372],[3372,3373],[3373,3374],[3374,3375],[3375,3376],[3376,3377],[3377,3378],[3378,3379],[3379,3380],[3380,3381],[3381,3382],[3382,3383],[3383,3384],[3384,3385],[3385,3386],[3386,3387],[3387,3388],[3388,3389],[3389,3390],[3390,3391],[3391,3392],[3392,3393],[3393,3394],[3394,3395],[3395,3396],[3396,3397],[3397,3398],[3398,3399],[3399,3400],[3400,3401],[3401,3402],[3402,3403],[3403,3404],[3404,3405],[3405,3406],[3406,3407],[3407,3408],[3408,3409],[3409,3410],[3410,3411],[3411,3412],[3412,3413],[3413,3414],[3414,3415],[3415,3416],[3416,3417],[3417,3418],[3418,3419],[3419,3420],[3420,3421],[3421,3422],[3422,3423],[3423,3424],[3424,3425],[3425,3426],[3426,3427],[3427,3428],[3428,3429],[3429,3430],[3430,3431],[3431,3432],[3432,3433],[3433,3434],[3434,3435],[3435,3436],[3436,3437],[3437,3438],[3438,3439],[3439,3440],[3440,3441],[3441,3442],[3442,3443],[3443,3444],[3444,3445],[3445,3446],[3446,3447],[3447,3448],[3448,3449],[3449,3450],[3450,3451],[3451,3452],[3452,3453],[3453,3454],[3454,3455],[3455,3456],[3456,3457],[3457,3458],[3458,3459],[3459,3460],[3460,3461],[3461,3462],[3462,3463],[3463,3464],[3464,3465],[3465,3466],[3466,3467],[3467,3468],[3468,3469],[3469,3470],[3470,3471],[3471,3472],[3472,3473],[3473,3474],[3474,3475],[3475,3476],[3476,3477],[3477,3478],[3478,3479],[3479,3480],[3480,3481],[3481,3482],[3482,3483],[3483,3484],[3484,3485],[3485,3486],[3486,3487],[3487,3488],[3488,3489],[3489,3490],[3490,3491],[3491,3492],[3492,3493],[3493,3494],[3494,3495],[3495,3496],[3496,3497],[3497,3498],[3498,3499],[3499,3500],[3500,3501],[3501,3502],[3502,3503],[3503,3504],[3504,3505],[3505,3506],[3506,3507],[3507,3508],[3508,3509],[3509,3510],[3510,3511],[3511,3512],[3512,3513],[3513,3514],[3514,3515],[3515,3516],[3516,3517],[3517,3518],[3518,3519],[3519,3520],[3520,3521],[3521,3522],[3522,3523],[3523,3524],[3524,3525],[3525,3526],[3526,3527],[3527,3528],[3528,3529],[3529,3530],[3530,3531],[3531,3532],[3532,3533],[3533,3534],[3534,3535],[3535,3536],[3536,3537],[3537,3538],[3538,3539],[3539,3540],[3540,3541],[3541,3542],[3542,3543],[3543,3544],[3544,3545],[3545,3546],[3546,3547],[3547,3548],[3548,3549],[3549,3550],[3550,3551],[3551,3552],[3552,3553],[3553,3554],[3554,3555],[3555,3556],[3556,3557],[3557,3558],[3558,3559],[3559,3560],[3560,3561],[3561,3562],[3562,3563],[3563,3564],[3564,3565],[3565,3566],[3566,3567],[3567,3568],[3568,3569],[3569,3570],[3570,3571],[3571,3572],[3572,3573],[3573,3574],[3574,3575],[3575,3576],[3576,3577],[3577,3578],[3578,3579],[3579,3580],[3580,3581],[3581,3582],[3582,3583],[3583,3584],[3584,3585],[3585,3586],[3586,3587],[3587,3588],[3588,3589],[3589,3590],[3590,3591],[3591,3592],[3592,3593],[3593,3594],[3594,3595],[3595,3596],[3596,3597],[3597,3598],[3598,3599],[3599,3600],[3600,3601],[3601,3602],[3602,3603],[3603,3604],[3604,3605],[3605,3606],[3606,3607],[3607,3608],[3608,3609],[3609,3610],[3610,3611],[3611,3612],[3612,3613],[3613,3614],[3614,3615],[3615,3616],[3616,3617],[3617,3618],[3618,3619],[3619,3620],[3620,3621],[3621,3622],[3622,3623],[3623,3624],[3624,3625],[3625,3626],[3626,3627],[3627,3628],[3628,3629],[3629,3630],[3630,3631],[3631,3632],[3632,3633],[3633,3634],[3634,3635],[3635,3636],[3636,3637],[3637,3638],[3638,3639],[3639,3640],[3640,3641],[3641,3642],[3642,3643],[3643,3644],[3644,3645],[3645,3646],[3646,3647],[3647,3648],[3648,3649],[3649,3650],[3650,3651],[3651,3652],[3652,3653],[3653,3654],[3654,3655],[3655,3656],[3656,3657],[3657,3658],[3658,3659],[3659,3660],[3660,3661],[3661,3662],[3662,3663],[3663,3664],[3664,3665],[3665,3666],[3666,3667],[3667,3668],[3668,3669],[3669,3670],[3670,3671],[3671,3672],[3672,3673],[3673,3674],[3674,3675],[3675,3676],[3676,3677],[3677,3678],[3678,3679],[3679,3680],[3680,3681],[3681,3682],[3682,3683],[3683,3684],[3684,3685],[3685,3686],[3686,3687],[3687,3688],[3688,3689],[3689,3690],[3690,3691],[3691,3692],[3692,3693],[3693,3694],[3694,3695],[3695,3696],[3696,3697],[3697,3698],[3698,3699],[3699,3700],[3700,3701],[3701,3702],[3702,3703],[3703,3704],[3704,3705],[3705,3706],[3706,3707],[3707,3708],[3708,3709],[3709,3710],[3710,3711],[3711,3712],[3712,3713],[3713,3714],[3714,3715],[3715,3716],[3716,3717],[3717,3718],[3718,3719],[3719,3720],[3720,3721],[3721,3722],[3722,3723],[3723,3724],[3724,3725],[3725,3726],[3726,3727],[3727,3728],[3728,3729],[3729,3730],[3730,3731],[3731,3732],[3732,3733],[3733,3734],[3734,3735],[3735,3736],[3736,3737],[3737,3738],[3738,3739],[3739,3740],[3740,3741],[3741,3742],[3742,3743],[3743,3744],[3744,3745],[3745,3746],[3746,3747],[3747,3748],[3748,3749],[3749,3750],[3750,3751],[3751,3752],[3752,3753],[3753,3754],[3754,3755],[3755,3756],[3756,3757],[3757,3758],[3758,3759],[3759,3760],[3760,3761],[3761,3762],[3762,3763],[3763,3764],[3764,3765],[3765,3766],[3766,3767],[3767,3768],[3768,3769],[3769,3770],[3770,3771],[3771,3772],[3772,3773],[3773,3774],[3774,3775],[3775,3776],[3776,3777],[3777,3778],[3778,3779],[3779,3780],[3780,3781],[3781,3782],[3782,3783],[3783,3784],[3784,3785],[3785,3786],[3786,3787],[3787,3788],[3788,3789],[3789,3790],[3790,3791],[3791,3792],[3792,3793],[3793,3794],[3794,3795],[3795,3796],[3796,3797],[3797,3798],[3798,3799],[3799,3800],[3800,3801],[3801,3802],[3802,3803],[3803,3804],[3804,3805],[3805,3806],[3806,3807],[3807,3808],[3808,3809],[3809,3810],[3810,3811],[3811,3812],[3812,3813],[3813,3814],[3814,3815],[3815,3816],[3816,3817],[3817,3818],[3818,3819],[3819,3820],[3820,3821],[3821,3822],[3822,3823],[3823,3824],[3824,3825],[3825,3826],[3826,3827],[3827,3828],[3828,3829],[3829,3830],[3830,3831],[3831,3832],[3832,3833],[3833,3834],[3834,3835],[3835,3836],[3836,3837],[3837,3838],[3838,3839],[3839,3840],[3840,3841],[3841,3842],[3842,3843],[3843,3844],[3844,3845],[3845,3846],[3846,3847],[3847,3848],[3848,3849],[3849,3850],[3850,3851],[3851,3852],[3852,3853],[3853,3854],[3854,3855],[3855,3856],[3856,3857],[3857,3858],[3858,3859],[3859,3860],[3860,3861],[3861,3862],[3862,3863],[3863,3864],[3864,3865],[3865,3866],[3866,3867],[3867,3868],[3868,3869],[3869,3870],[3870,3871],[3871,3872],[3872,3873],[3873,3874],[3874,3875],[3875,3876],[3876,3877],[3877,3878],[3878,3879],[3879,3880],[3880,3881],[3881,3882],[3882,3883],[3883,3884],[3884,3885],[3885,3886],[3886,3887],[3887,3888],[3888,3889],[3889,3890],[3890,3891],[3891,3892],[3892,3893],[3893,3894],[3894,3895],[3895,3896],[3896,3897],[3897,3898],[3898,3899],[3899,3900],[3900,3901],[3901,3902],[3902,3903],[3903,3904],[3904,3905],[3905,3906],[3906,3907],[3907,3908],[3908,3909],[3909,3910],[3910,3911],[3911,3912],[3912,3913],[3913,3914],[3914,3915],[3915,3916],[3916,3917],[3917,3918],[3918,3919],[3919,3920],[3920,3921],[3921,3922],[3922,3923],[3923,3924],[3924,3925],[3925,3926],[3926,3927],[3927,3928],[3928,3929],[3929,3930],[3930,3931],[3931,3932],[3932,3933],[3933,3934],[3934,3935],[3935,3936],[3936,3937],[3937,3938],[3938,3939],[3939,3940],[3940,3941],[3941,3942],[3942,3943],[3943,3944],[3944,3945],[3945,3946],[3946,3947],[3947,3948],[3948,3949],[3949,3950],[3950,3951],[3951,3952],[3952,3953],[3953,3954],[3954,3955],[3955,3956],[3956,3957],[3957,3958],[3958,3959],[3959,3960],[3960,3961],[3961,3962],[3962,3963],[3963,3964],[3964,3965],[3965,3966],[3966,3967],[3967,3968],[3968,3969],[3969,3970],[3970,3971],[3971,3972],[3972,3973],[3973,3974],[3974,3975],[3975,3976],[3976,3977],[3977,3978],[3978,3979],[3979,3980],[3980,3981],[3981,3982],[3982,3983],[3983,3984],[3984,3985],[3985,3986],[3986,3987],[3987,3988],[3988,3989],[3989,3990],[3990,3991],[3991,3992],[3992,3993],[3993,3994],[3994,3995],[3995,3996],[3996,3997],[3997,3998],[3998,3999],[3999,4000],[4000,4001],[4001,4002],[4002,4003],[4003,4004],[4004,4005],[4005,4006],[4006,4007],[4007,4008],[4008,4009],[4009,4010],[4010,4011],[4011,4012],[4012,4013],[4013,4014],[4014,4015],[4015,4016],[4016,4017],[4017,4018],[4018,4019],[4019,4020],[4020,4021],[4021,4022],[4022,4023],[4023,4024],[4024,4025],[4025,4026],[4026,4027],[4027,4028],[4028,4029],[4029,4030],[4030,4031],[4031,4032],[4032,4033],[4033,4034],[4034,4035],[4035,4036],[4036,4037],[4037,4038],[4038,4039],[4039,4040],[4040,4041],[4041,4042],[4042,4043],[4043,4044],[4044,4045],[4045,4046],[4046,4047],[4047,4048],[4048,4049],[4049,4050],[4050,4051],[4051,4052],[4052,4053],[4053,4054],[4054,4055],[4055,4056],[4056,4057],[4057,4058],[4058,4059],[4059,4060],[4060,4061],[4061,4062],[4062,4063],[4063,4064],[4064,4065],[4065,4066],[4066,4067],[4067,4068],[4068,4069],[4069,4070],[4070,4071],[4071,4072],[4072,4073],[4073,4074],[4074,4075],[4075,4076],[4076,4077],[4077,4078],[4078,4079],[4079,4080],[4080,4081],[4081,4082],[4082,4083],[4083,4084],[4084,4085],[4085,4086],[4086,4087],[4087,4088],[4088,4089],[4089,4090],[4090,4091],[4091,4092],[4092,4093],[4093,4094],[4094,4095],[4095,4096],[4096,4097],[4097,4098],[4098,4099],[4099,4100],[4100,4101],[4101,4102],[4102,4103],[4103,4104],[4104,4105],[4105,4106],[4106,4107],[4107,4108],[4108,4109],[4109,4110],[4110,4111],[4111,4112],[4112,4113],[4113,4114],[4114,4115],[4115,4116],[4116,4117],[4117,4118],[4118,4119],[4119,4120],[4120,4121],[4121,4122],[4122,4123],[4123,4124],[4124,4125],[4125,4126],[4126,4127],[4127,4128],[4128,4129],[4129,4130],[4130,4131],[4131,4132],[4132,4133],[4133,4134],[4134,4135],[4135,4136],[4136,4137],[4137,4138],[4138,4139],[4139,4140],[4140,4141],[4141,4142],[4142,4143],[4143,4144],[4144,4145],[4145,4146],[4146,4147],[4147,4148],[4148,4149],[4149,4150],[4150,4151],[4151,4152],[4152,4153],[4153,4154],[4154,4155],[4155,4156],[4156,4157],[4157,4158],[4158,4159],[4159,4160],[4160,4161],[4161,4162],[4162,4163],[4163,4164],[4164,4165],[4165,4166],[4166,4167],[4167,4168],[4168,4169],[4169,4170],[4170,4171],[4171,4172],[4172,4173],[4173,4174],[4174,4175],[4175,4176],[4176,4177],[4177,4178],[4178,4179],[4179,4180],[4180,4181],[4181,4182],[4182,4183],[4183,4184],[4184,4185],[4185,4186],[4186,4187],[4187,4188],[4188,4189],[4189,4190],[4190,4191],[4191,4192],[4192,4193],[4193,4194],[4194,4195],[4195,4196],[4196,4197],[4197,4198],[4198,4199],[4199,4200],[4200,4201],[4201,4202],[4202,4203],[4203,4204],[4204,4205],[4205,4206],[4206,4207],[4207,4208],[4208,4209],[4209,4210],[4210,4211],[4211,4212],[4212,4213],[4213,4214],[4214,4215],[4215,4216],[4216,4217],[4217,4218],[4218,4219],[4219,4220],[4220,4221],[4221,4222],[4222,4223],[4223,4224],[4224,4225],[4225,4226],[4226,4227],[4227,4228],[4228,4229],[4229,4230],[4230,4231],[4231,4232],[4232,4233],[4233,4234],[4234,4235],[4235,4236],[4236,4237],[4237,4238],[4238,4239],[4239,4240],[4240,4241],[4241,4242],[4242,4243],[4243,4244],[4244,4245],[4245,4246],[4246,4247],[4247,4248],[4248,4249],[4249,4250],[4250,4251],[4251,4252],[4252,4253],[4253,4254],[4254,4255],[4255,4256],[4256,4257],[4257,4258],[4258,4259],[4259,4260],[4260,4261],[4261,4262],[4262,4263],[4263,4264],[4264,4265],[4265,4266],[4266,4267],[4267,4268],[4268,4269],[4269,4270],[4270,4271],[4271,4272],[4272,4273],[4273,4274],[4274,4275],[4275,4276],[4276,4277],[4277,4278],[4278,4279],[4279,4280],[4280,4281],[4281,4282],[4282,4283],[4283,4284],[4284,4285],[4285,4286],[4286,4287],[4287,4288],[4288,4289],[4289,4290],[4290,4291],[4291,4292],[4292,4293],[4293,4294],[4294,4295],[4295,4296],[4296,4297],[4297,4298],[4298,4299],[4299,4300],[4300,4301],[4301,4302],[4302,4303],[4303,4304],[4304,4305],[4305,4306],[4306,4307],[4307,4308],[4308,4309],[4309,4310],[4310,4311],[4311,4312],[4312,4313],[4313,4314],[4314,4315],[4315,4316],[4316,4317],[4317,4318],[4318,4319],[4319,4320],[4320,4321],[4321,4322],[4322,4323],[4323,4324],[4324,4325],[4325,4326],[4326,4327],[4327,4328],[4328,4329],[4329,4330],[4330,4331],[4331,4332],[4332,4333],[4333,4334],[4334,4335],[4335,4336],[4336,4337],[4337,4338],[4338,4339],[4339,4340],[4340,4341],[4341,4342],[4342,4343],[4343,4344],[4344,4345],[4345,4346],[4346,4347],[4347,4348],[4348,4349],[4349,4350],[4350,4351],[4351,4352],[4352,4353],[4353,4354],[4354,4355],[4355,4356],[4356,4357],[4357,4358],[4358,4359],[4359,4360],[4360,4361],[4361,4362],[4362,4363],[4363,4364],[4364,4365],[4365,4366],[4366,4367],[4367,4368],[4368,4369],[4369,4370],[4370,4371],[4371,4372],[4372,4373],[4373,4374],[4374,4375],[4375,4376],[4376,4377],[4377,4378],[4378,4379],[4379,4380],[4380,4381],[4381,4382],[4382,4383],[4383,4384],[4384,4385],[4385,4386],[4386,4387],[4387,4388],[4388,4389],[4389,4390],[4390,4391],[4391,4392],[4392,4393],[4393,4394],[4394,4395],[4395,4396],[4396,4397],[4397,4398],[4398,4399],[4399,4400],[4400,4401],[4401,4402],[4402,4403],[4403,4404],[4404,4405],[4405,4406],[4406,4407],[4407,4408],[4408,4409],[4409,4410],[4410,4411],[4411,4412],[4412,4413],[4413,4414],[4414,4415],[4415,4416],[4416,4417],[4417,4418],[4418,4419],[4419,4420],[4420,4421],[4421,4422],[4422,4423],[4423,4424],[4424,4425],[4425,4426],[4426,4427],[4427,4428],[4428,4429],[4429,4430],[4430,4431],[4431,4432],[4432,4433],[4433,4434],[4434,4435],[4435,4436],[4436,4437],[4437,4438],[4438,4439],[4439,4440],[4440,4441],[4441,4442],[4442,4443],[4443,4444],[4444,4445],[4445,4446],[4446,4447],[4447,4448],[4448,4449],[4449,4450],[4450,4451],[4451,4452],[4452,4453],[4453,4454],[4454,4455],[4455,4456],[4456,4457],[4457,4458],[4458,4459],[4459,4460],[4460,4461],[4461,4462],[4462,4463],[4463,4464],[4464,4465],[4465,4466],[4466,4467],[4467,4468],[4468,4469],[4469,4470],[4470,4471],[4471,4472],[4472,4473],[4473,4474],[4474,4475],[4475,4476],[4476,4477],[4477,4478],[4478,4479],[4479,4480],[4480,4481],[4481,4482],[4482,4483],[4483,4484],[4484,4485],[4485,4486],[4486,4487],[4487,4488],[4488,4489],[4489,4490],[4490,4491],[4491,4492],[4492,4493],[4493,4494],[4494,4495],[4495,4496],[4496,4497],[4497,4498],[4498,4499],[4499,4500],[4500,4501],[4501,4502],[4502,4503],[4503,4504],[4504,4505],[4505,4506],[4506,4507],[4507,4508],[4508,4509],[4509,4510],[4510,4511],[4511,4512],[4512,4513],[4513,4514],[4514,4515],[4515,4516],[4516,4517],[4517,4518],[4518,4519],[4519,4520],[4520,4521],[4521,4522],[4522,4523],[4523,4524],[4524,4525],[4525,4526],[4526,4527],[4527,4528],[4528,4529],[4529,4530],[4530,4531],[4531,4532],[4532,4533],[4533,4534],[4534,4535],[4535,4536],[4536,4537],[4537,4538],[4538,4539],[4539,4540],[4540,4541],[4541,4542],[4542,4543],[4543,4544],[4544,4545],[4545,4546],[4546,4547],[4547,4548],[4548,4549],[4549,4550],[4550,4551],[4551,4552],[4552,4553],[4553,4554],[4554,4555],[4555,4556],[4556,4557],[4557,4558],[4558,4559],[4559,4560],[4560,4561],[4561,4562],[4562,4563],[4563,4564],[4564,4565],[4565,4566],[4566,4567],[4567,4568],[4568,4569],[4569,4570],[4570,4571],[4571,4572],[4572,4573],[4573,4574],[4574,4575],[4575,4576],[4576,4577],[4577,4578],[4578,4579],[4579,4580],[4580,4581],[4581,4582],[4582,4583],[4583,4584],[4584,4585],[4585,4586],[4586,4587],[4587,4588],[4588,4589],[4589,4590],[4590,4591],[4591,4592],[4592,4593],[4593,4594],[4594,4595],[4595,4596],[4596,4597],[4597,4598],[4598,4599],[4599,4600],[4600,4601],[4601,4602],[4602,4603],[4603,4604],[4604,4605],[4605,4606],[4606,4607],[4607,4608],[4608,4609],[4609,4610],[4610,4611],[4611,4612],[4612,4613],[4613,4614],[4614,4615],[4615,4616],[4616,4617],[4617,4618],[4618,4619],[4619,4620],[4620,4621],[4621,4622],[4622,4623],[4623,4624],[4624,4625],[4625,4626],[4626,4627],[4627,4628],[4628,4629],[4629,4630],[4630,4631],[4631,4632],[4632,4633],[4633,4634],[4634,4635],[4635,4636],[4636,4637],[4637,4638],[4638,4639],[4639,4640],[4640,4641],[4641,4642],[4642,4643],[4643,4644],[4644,4645],[4645,4646],[4646,4647],[4647,4648],[4648,4649],[4649,4650],[4650,4651],[4651,4652],[4652,4653],[4653,4654],[4654,4655],[4655,4656],[4656,4657],[4657,4658],[4658,4659],[4659,4660],[4660,4661],[4661,4662],[4662,4663],[4663,4664],[4664,4665],[4665,4666],[4666,4667],[4667,4668],[4668,4669],[4669,4670],[4670,4671],[4671,4672],[4672,4673],[4673,4674],[4674,4675],[4675,4676],[4676,4677],[4677,4678],[4678,4679],[4679,4680],[4680,4681],[4681,4682],[4682,4683],[4683,4684],[4684,4685],[4685,4686],[4686,4687],[4687,4688],[4688,4689],[4689,4690],[4690,4691],[4691,4692],[4692,4693],[4693,4694],[4694,4695],[4695,4696],[4696,4697],[4697,4698],[4698,4699],[4699,4700],[4700,4701],[4701,4702],[4702,4703],[4703,4704],[4704,4705],[4705,4706],[4706,4707],[4707,4708],[4708,4709],[4709,4710],[4710,4711],[4711,4712],[4712,4713],[4713,4714],[4714,4715],[4715,4716],[4716,4717],[4717,4718],[4718,4719],[4719,4720],[4720,4721],[4721,4722],[4722,4723],[4723,4724],[4724,4725],[4725,4726],[4726,4727],[4727,4728],[4728,4729],[4729,4730],[4730,4731],[4731,4732],[4732,4733],[4733,4734],[4734,4735],[4735,4736],[4736,4737],[4737,4738],[4738,4739],[4739,4740],[4740,4741],[4741,4742],[4742,4743],[4743,4744],[4744,4745],[4745,4746],[4746,4747],[4747,4748],[4748,4749],[4749,4750],[4750,4751],[4751,4752],[4752,4753],[4753,4754],[4754,4755],[4755,4756],[4756,4757],[4757,4758],[4758,4759],[4759,4760],[4760,4761],[4761,4762],[4762,4763],[4763,4764],[4764,4765],[4765,4766],[4766,4767],[4767,4768],[4768,4769],[4769,4770],[4770,4771],[4771,4772],[4772,4773],[4773,4774],[4774,4775],[4775,4776],[4776,4777],[4777,4778],[4778,4779],[4779,4780],[4780,4781],[4781,4782],[4782,4783],[4783,4784],[4784,4785],[4785,4786],[4786,4787],[4787,4788],[4788,4789],[4789,4790],[4790,4791],[4791,4792],[4792,4793],[4793,4794],[4794,4795],[4795,4796],[4796,4797],[4797,4798],[4798,4799],[4799,4800],[4800,4801],[4801,4802],[4802,4803],[4803,4804],[4804,4805],[4805,4806],[4806,4807],[4807,4808],[4808,4809],[4809,4810],[4810,4811],[4811,4812],[4812,4813],[4813,4814],[4814,4815],[4815,4816],[4816,4817],[4817,4818],[4818,4819],[4819,4820],[4820,4821],[4821,4822],[4822,4823],[4823,4824],[4824,4825],[4825,4826],[4826,4827],[4827,4828],[4828,4829],[4829,4830],[4830,4831],[4831,4832],[4832,4833],[4833,4834],[4834,4835],[4835,4836],[4836,4837],[4837,4838],[4838,4839],[4839,4840],[4840,4841],[4841,4842],[4842,4843],[4843,4844],[4844,4845],[4845,4846],[4846,4847],[4847,4848],[4848,4849],[4849,4850],[4850,4851],[4851,4852],[4852,4853],[4853,4854],[4854,4855],[4855,4856],[4856,4857],[4857,4858],[4858,4859],[4859,4860],[4860,4861],[4861,4862],[4862,4863],[4863,4864],[4864,4865],[4865,4866],[4866,4867],[4867,4868],[4868,4869],[4869,4870],[4870,4871],[4871,4872],[4872,4873],[4873,4874],[4874,4875],[4875,4876],[4876,4877],[4877,4878],[4878,4879],[4879,4880],[4880,4881],[4881,4882],[4882,4883],[4883,4884],[4884,4885],[4885,4886],[4886,4887],[4887,4888],[4888,4889],[4889,4890],[4890,4891],[4891,4892],[4892,4893],[4893,4894],[4894,4895],[4895,4896],[4896,4897],[4897,4898],[4898,4899],[4899,4900],[4900,4901],[4901,4902],[4902,4903],[4903,4904],[4904,4905],[4905,4906],[4906,4907],[4907,4908],[4908,4909],[4909,4910],[4910,4911],[4911,4912],[4912,4913],[4913,4914],[4914,4915],[4915,4916],[4916,4917],[4917,4918],[4918,4919],[4919,4920],[4920,4921],[4921,4922],[4922,4923],[4923,4924],[4924,4925],[4925,4926],[4926,4927],[4927,4928],[4928,4929],[4929,4930],[4930,4931],[4931,4932],[4932,4933],[4933,4934],[4934,4935],[4935,4936],[4936,4937],[4937,4938],[4938,4939],[4939,4940],[4940,4941],[4941,4942],[4942,4943],[4943,4944],[4944,4945],[4945,4946],[4946,4947],[4947,4948],[4948,4949],[4949,4950],[4950,4951],[4951,4952],[4952,4953],[4953,4954],[4954,4955],[4955,4956],[4956,4957],[4957,4958],[4958,4959],[4959,4960],[4960,4961],[4961,4962],[4962,4963],[4963,4964],[4964,4965],[4965,4966],[4966,4967],[4967,4968],[4968,4969],[4969,4970],[4970,4971],[4971,4972],[4972,4973],[4973,4974],[4974,4975],[4975,4976],[4976,4977],[4977,4978],[4978,4979],[4979,4980],[4980,4981],[4981,4982],[4982,4983],[4983,4984],[4984,4985],[4985,4986],[4986,4987],[4987,4988],[4988,4989],[4989,4990],[4990,4991],[4991,4992],[4992,4993],[4993,4994],[4994,4995],[4995,4996],[4996,4997],[4997,4998],[4998,4999]]]
    # for i in range(5):
    # res = sol.findNumberOfLIS([3,1,2])
    # res = sol.findNumberOfLIS([2,2,2,2,2])
    # res = sol.findNumberOfLIS([1,2,4,3,5,4,7,2])
    # res = sol.canPartitionKSubsets([605,454,322,218,8,19,651,2220,175,710,2666,350,252,2264,327,1843],4)
    # res = sol.canFinish(3,[[0,1],[1,0]])
    # res = sol.canFinish(3,[[1,0],[0,2],[2,1]])
    # res = sol.canFinish(8,[[1,0],[2,6],[1,7],[6,4],[7,0],[0,5]])
    # res = sol.findMinHeightTrees(3,[[0,1],[0,2]])
    # res = sol.findMinHeightTrees(*tdata)
    # res = sol.pacificAtlantic([[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]])
    # res = sol.search([1,2,1],2)
    res = sol.subsetsWithDup([1,2,2,3])
    # tdata = ["iibazqymhssjxnaurfsddydbwftetatwyosdcnnujzjzbpdeozpwpaixxghuyrlwrxdsyvpgoycuumxtryrqpgfmgyodbvtmybzkaqoobecgosekwvcyyfzuaiweqjnwfssjmbrcfkppdclmqpayziyzwhktkxlmhjgjjrkdvfseatybsltvlsklnxxywzcgvvqxrnkkccfvbwpjfnpqrkfqgoajiksmtmogouhivogjuriyfwushmnyqcrdkkfkrhfieaujweckpjjtzlitjhghukqllqttszuzrnwyafumimkiljsatlklrwwdbxcaunklenvcbgq","quitwlnicl"]
    # tdata = ["cnhnczmccqouqadqtnmjjzl","nm"]
    # tdata = ["tizglvnpwadrarimeufxexveuxwrvaatjjkssklxnentjuyumobfuyixktaqztiodzoliiwppdqvplwrnrbapiddjptzlkzysmdijvisrxpsuxvmgzovkcekddgzhxwcisxapnwwszhlageihuxgewuicxiafrrceswnqiuujzruaegsfxwapmkxchfvmrhuaqnumtjilygfhdwxbevciazptagpdouknowqdpvbqqhebfiorzngqswdfxczzgxfnmacrpwgmhvgolotlfvobosdvwptomodjqemdjbexcqjzbslwrjqrecfgltkefoizfgyrhoruvaovolifrwvsgcuaftjjurufxydvhmsuttiuylsfwfbhfynlmjvrpsrvfudfqnlbhhsfsjocyoadxhfobudlgygaykvxcmlhvonkxvoycrukbucdplorwhpvazbmqhhiorpdmzvzavomruxemiwuurdmmkqztwiyfytjvzltnriaypqckiecgkyosyhdqhpiicfwyncmcocatdtjdrrqomlidoglzewffvcbggcbptawnovoziqnedvlbkrravyzpthenvikrrmriwovruucbdymufjveymajnvheqfrcnffdfnomyfjydsbousktbtlwpbxcfznxarnubwbaowmahkscrtifvrntjjyhpevxxsqcbpqlvgcpcxdskgklcsxjvoqybjurprlnhvikumrbdfdpqxijqsdbkjsykjekrehcawsqetimvyewwxfcchvjmwsenvnpxqtyeilwdmskarrpaomhhseguozehmrmiybifmfppjayngcniyiagbmtrajgxdewahvintzwjskweoardirqnljiaolcwzhuadtcpjkxhdgladruonehzjhvqdfeoqmwmpokbiwrubldtklomsxdsjjcnynqpcqblnsnexzvbpovooxwdgftnaxbszrlxpmnjrfmgmogeichqpgvrayawcnqruduneqksbnoasjcwoistelhuublbcmqgjliawqkkzgvocdpecitonnpprxtpssoectcycvlzhqzliuhnvxzgwrywhbikkjfpgpnljqrfdtuxmocsezzpgjudgmmwxupivfypbrxakoxropyioqhtqfqiiosjhvooddnclwvmjftelcnfzxhtcefnzsusvttdlkciinxdaibfvubmnigthlpbapgixdltnoiffnqiystaodlwiqvldxrgucibsusfasswagevqitlosqxtertfoidbffvoeptcwlaoeujigjurxkatfeouduqsqjenbtjtsoydgfqyipmcqokdlfowuexrlexjudlisndlvkwbulhevojwcjccozjjtzwjlpqoxecivazokcxgoxmchrisvtptsfdcroogjaexismjqwjrykytuthrvzeiiitxctkbcdcxsylkshrhfdzsxylghicenofniexzfmfmmntkynsnrjjzozlzizanemvpbzjemxpigviqbqhvhinyjpowboyowwyxdwutuuheqvzhvwvbxyedpssqlpyjhznemclizqnmshtktjwzxdnagtfpwxfesubqtpyllbwqgzeyslpdfvtrddwvwjhsbjzvtwcjcyxmissxxjihcscyndramqucqylktufczepwkylruyjgrodpsldhxtmleogheudvoivxowmsuphghmulfzyudxrnfcdafpgmsmzjypuitpjyxfhkbnyzgnuylfenwiyjrarzmkmdtiwpeeqfwpnjhspzrzvgbsvmdpnclunmoqwfkbwpmolqvdmqkahclvhxtozotpsqbnjjtqzlwbvomqbprmzlsxjoxjcamubqwwfccmjlevpcgkjzrgswiibedvsffqecnzvcjqfcxeqyfuibpawvomjipphmectdwsvhlmghrxttwwuxfmdcdawaesdygngkyblfryduloaocpnxzlssguujmqblufiunozgjimsxuulaqrdrmdcmmgrrqineccootgzojbxguajvwlcmmlivyzcvwdesyuacaffsotclwsycwrvlssritecbicfmdgzdiicmkcbopvqjugrihbpkxliakuggncfojumzplpiwjfpzlkrgtupykimfmtdkyzpiywfvpubgwpzgsijdrhnapuzppqwmjkdncmsrzidnrnqddohmbrwojqatrhsuopryisnmjyiwnbabmlqubywqaiowtaimpjkgsyqsamnouglustpcteeciiacebwinelocxhjgrlgtptfzaxanidwioqsxeyzsgmiiizstcbybxzpkzdirhlgqbnxwuyrfoilkluihrwwjcggxduxgttroczucrpjnudjtjamceprrehibvttnybckjdyxejpjwukgubdknjqzigxfvxrgacthcryzdwkmiquznifqkbhuhqbfvddxgmuybrdehwcitwpummdauylabxnixgzotbxxwdzhivmisqjazecqhtpleshqjmgyhkihikettlsndjauqwhyeeukojwqiggprcpzicbvgyhasrqcxbdrakmdchtqjisiznuwkhzxyzyqzsypuzqopyznemqyrrvrkytzelqofpebvszvfxizwveboxtanhbkiujnpakyjxoixfyvtpdtwjqredevgysvxtpmjublzksgfbfaiqpjcuwcytatztkwvkvwukuggqlxtqdtzkegsdjoubufcxmnphdudphlprcizhudedgccqxexjvgttjyffkqxdsdgimwcotlteijibxdvdeaxcnwojrfyaiiwwtltzbgexavgadabnhekmmvxyvchmzgyccxzagypsvkypveetypcjmafkhdyvriicprptukcwaxnaosrprvtkqlpchtvedkcakixxqwxdshiiqvatrhjxaurwtpyedpwuunneyylgsbyhefxibsjvjtipjsjfkqxhzuxvyugkgopwprumuheoxzbgvfrftlprwmrjfdbcpidnowgjslapskibjrjuethuwydvrzcnnzbppvkqgikdknwpumfaluccgmltmjgfjxvstjaobjxvvploxhrywhbycamasoyfpnfcakhjgujnywyjhazjyokcwcrkeoxechuailpviskfjjrnwubndayhleptuyfowrlyrneafuqiylmfsmvzvmbyobymbzyemqnbxarnrlbqsokclvliaznnsnvxsoqyvqdcdgurknpfuqalxtenehrjwmiffeiyxwxbumrvobmoecozroncffyzopecsvwlngcjbhrhfpqlbvydevpplfflnehnvkrzdjxffzcqbwdpodgyurzhuuqgfwtrsigricbfkxpsjhlcuxahpvmaimabcdjopyxphykduphqbzhuvwibgbfsqfnwnoqwwyshdthcsmzzqexowrbpuqozjfypqjbwtqdahfcwbtcasnsyfrlseqsdjmuxgcgahvfyudkchptzedkfvvsocjcomumfauxorcsyutcgjqtgyvrkteixgovpsisgmpwffizcpzkfhhmpkwmsrmjkqinarobkqctsmqxppefadlgqbtainshlsdstsxkjubzgfcccqmqtqhqwryjwktlhzbhdqvpbzyiabvkxhyntrswoaxzqilrppeyzjngmwoxnugrlfrftiysgukxumnlwsucaqecpyuzyzghnernvrciacwchqqqdeigglktpikizxvyrzcpvfrehjdmagwwpztfpmmqgoxdwmwonixjeomsmuhfbhckkedqilidmlqtplruwdsgpflydzwrlrqjlprpqouvrdzzicfvevaiamhjutpaocknpuxotcvwhbcfzthbbjwdbrhfivmqyyevfzkzamdkmsawxbyjngttwfoldnazzqjabywvxjpuyvtylwjbpxwbqhybtfiwpolgeeelbtpafwudfwouudrbbfbbssaqcbtfarqvwicijorgfbzjykfdbdxatpumbnbrnvudxhwevomvipapbtuhbovxfwfhjlcjglydissiawiqkqgbunuhdpevqccrrqycsywratpnuuccdrznqzfwceotztuzzdyhtosrqudtkhscjtfvskilocrgwavaflxfgbevpfuhqttmetognhgkswflvbwgcraorzmamktsytqxtnegtywvslyvjxipfuheuuduxkunppkmlbnjmtcuaqquxofmjlrsmlzuznzgrqdovzxbrvkaxlfpyuxsvwuohjewkbxntbqbxlxkvdxthvihhvmyuogxhqtxddjvcufrddstecbtmmeunkurguegiygqjsdclrsypeobwdutywohybvwbunbdnizddvtuxhoasubnampccqpukhnphzlczrhzqfqgcgqaavevacjqfdjywiocsvgtzugpmnbodgwpyydkifmbfnvhogyzcidkfzfsppjrmuxqverlwqvetgfifhuwvzizthgkfyigszxgbmsixsuyizhjfbrvbshebdrlunvyniezhrwgywjurjyrqraeoucirgtbukwysfdcdrwxjrqintjdheebylmottpfrjoadhcqxrnlmlaxoqkkylamvplmyydvcehmpcvrysfreytulqfhsrjvvpgsvnwfoisbxvtigvoxezkxxhnecenqopsqkklfalglhleescnnldxnrzrmxsqgmamhnsbmdzgoppcmgyvsrpxcfeihpszxnksznhxlkbmewdydxppbimofrebpflgptpddchiquovfbfwwslmpitjticvrtlibgvdtqxpmwfahxdhzstijznzqgyblfvnftejchqtqtvfnshttbldrwyfcxowwxvzqeellrcwgtynoypwrrtzenyfdumxzqfrqzoiyrxpakakbeykspipzaunxapexnsplojrbhpujgavntapfhdeomndnnvpxriurqhmautsfswyqoozizevbrkxubbwrsbycmjhgrxveecwveszbqgkbcdquoqwutsszmqbppdywvfqlavbnxcyowximjqnntdgyatyfscnufmmabpsozltvivkghgrsqihnivdgiluhxkcoerrwdocxaqghsarormtqvhwvsyhbbdwemoddqfxcjozkfqkxtodtoqbwrwszlgrfokjsoqiyczxadnsnhvcobrpnhfxfqpqctgahiniwjmqkxsimwxcjdcvmuslhrevfixovocupeqfvgoyhvoyzgqrgfuqkegrodshqgbpcdscieajehzvtwolzlaibkfqmgyncaxngcgoaiezzcfegkdhdqiswldvcnaupdqopsiesaietwyroyeqaivumcpztedvogffzioygeymoydcvshwptushayrebmdxxkpejxcgueucnxfilfuktrwkabgqllexcgczwwhnvafvjcnzbxcumtbqxjekodxrqrnezpuicmyznvinqcalcbjszmhkclpzvsabfyshxyjhdxeivfbyhdyhuuecmfexyypqrrdearknqcltlljadmdzditbmehrckxpaznivyjlgyihnbxzhvluiwerudguipbeshncltvhvgkijhrbohhxmzawsdjpguoeskzzbytijoioadoirjhrrlyzpucsxkrmjdsfbvgkhdlajeooyxxnkruwpuemzbghdwprajxrijcvuzzymdltvaqwmcxtecumgycvbrasqfhiimezgkzndjskveefkalffxzaybeevtdkddqxhpsemwgihjwpdhvenrghigpphyrnopqshufrmyfixecazukulkegkmuwhsfazkeoygqnsdmwovroiztmndpovbovohqoqurhplqlhrdxqgajgyapcsumifyfmlwpgarlpnpdbtofafvwcohnijhklppmarbcymsukiiluvgyhusaudwnuzbdsjjslnmgnxhwnolnhhbwvkonhmhnvlrzazlkcmvqappmzcnrmyypsvxdslxrgsrcqwsuawklfnywkraencxgnqnxpvoawhdmsxakxcuzedssmgdyrvimbiaipgsujmwufabecmjpyglqtjbliugbripyiuvbraoyjktdbtzytyzqpthhfrzbghnjcthzxltzynbxtqlksscfqbimmnhjilkcrvzneeuykmqibiofkigkwpimpawcstawyzwfxsvcapmyzoeqzuwnkdzvacblyfseykgwncmlanmyuqakupbjndoxmbmoihwzilszttiujdvzqitukvzkobceovszzscwyiccngnbpbiksvfqkofqsjpuqpfmwqlozftqproekcyfdnriphlgdihzavoowhcpuavnzrvsphacvfduwhlqpfldewmlabpdksvujhajwbpwifrxxrolmwcxyzygoadrzbdsyytfbmcbiitwspshqqrfqgasegsqclnnyymxqbqetrwvxfceilcgxuodkbwznstnmqtogtcrmmxcqlwcnmdojdyfyeobgpzbkeansuxhlivvgbpvhnxprnohonlybqcvwcjouygalvkxlyqizvyxzralmqzimkieazcncezqhnkctnadbeiagmsxudignwygycjbrkltnqzcskvhasesulpvfnfcosvwbpoyhkkimnggxxqvnodklnzgeotxlihbjdoctkkbwrqsrcatkeeqpfabbdwdijltlzmurzsomheisvadnnbaczwordozqejwcwgvtbmhwregiztqfexhvbwuadcdpdltvuxkltldpefjcksabxxjapznubaxtvdrfpnlurynzkbdipqinxuvdiyidmesdhnptghiodcmwlhwololxpusaahxuimjmwjrrwfwgoonkvdotigheuqzbuphsdlnukoolwhvoqkwcagucwwipyqaztvtzvssxhmydouujlihaljtenypjbzlaaufcpxmgxxafosdtvslusumyktxaxebmwnyohjosbkiytnctmoyysbsrhzfkbqjaoopqadpxmafoujerjatnedjjzohhgvbsmmhtqogsdidiihmzluphxntpkrzbuybpjjgmvzcuxqyvmwmsnwdqbrsffgvzqlzkrikhuejebofvpmgejefjgqidtacinwvpndvgwunsnoycbqbevwudfsdreybibtvnitcivyiclhmsxqcifecaloqzjvziegtwlyerreuokxkjildifawjzbtasqkrhokkbdzkkxirkytfgzbqoroavyedewpzaovuymebduiwtryhwvxwrfbkpvuhkiedjsdcjevremfxbmkxffbxoyyossgtbgdrykugjzzpvolwbxvyleluyeavjtxygiarxxpbbtffepiwzsgohalrhhprjwuralfxntmdcyggppdmfpvnwiofyapzqxuomoxebtauxqmllghrslymdkbskwatpfusyjjpeohseqizuzdnsubrjosztgujiojvpsnvxgyubancdevwrljesodfuxftppycvdwotkqshcofkwmhnmamabsdcuxgbaqgerkhuvrdllmlvkkhgffknpnlhdbfgtvkfumjyfcapgbeadybguwxgcuhwfvirybsyueqliskvekfapcumnkkkqvncjzffhjvlcgfhfvdjnmwvogxhqqaxbjfwkmbjhbdnwsdeecuhoybilbqzbnvfdnzlxlvsfutnqbddlubhlmasmukqgbypototabfenmbnrtyjdsfbghoqqicxxxlpdmmspzrykskisvucprelqrnhwrbclzbdbkqqdrdlfjqspalqvowqrspwyrqihcvrviltmntmqozghnhgosshzyademfkqoqwtetkmyenlkwykubumuenvsqplgdneexwxnppobrdsflfpyuzgfsjhtukbkxltfciqwvfwhbxwnvufcxridopcnzlbqjnxlvszahhirucyhuhybaubgnpqlmdlgmxnxcnjvpmhudaneucqlistxxxxvnosgoqqvbitrfublihhugrpukizejnifrnurntvhpcwpkeggljewllrmzlorrspfrthaldauaawqglnvxuiinhfewajlugovwofuzjmgaacyagjxaahtcyhpxwtilahnxbqtfrhyjznehmxbtxnfxlpjfwsuhiedpulcwqnnlljltnlbmfybvlrozeunlooxmwuwfyqlxprelqdxsjtndkfkwzhjpctscxcafnqqfalgttxzmzxsfavgsdxxjpemvexifgisdagskwytgrhxxsrafnkqcqflzyzmaftoakhsnqtiheodrowqyovnxgynzkhrggstbfchfkswsnoazcywxcrvmzcidwecacekyezamhnglejmfkelrjmvgcmuqslpibqicayeprjlaxrjjceagstjddmsmgacpxcvgujpqxxetuvuhwrdnysjkeuiumwhofryrfpeqzuiwdolahskxjwykmoeskookbswrovgdvtlgajhzfythkckamyavudnicgzmausgjpctseaybvvzhzvgyigodnlgwnrnfkoqsljtzlbftqmtyrlsztmvwvewudzefjeprkoxfhenhwzxndgbwcjsjabukfvirebfqhwmdyzblezzvwckqmeyprihxqexfmqlifrsudcxhwqpdcyztepdzxtcshjhuokcweiisydappcrqepvfagmodcsmtaqhygnudjzxbkpsybtajdgmyevlbqmvmggpzppeljmosulscfbodcfcqcezmffctengycejcdzbyikculifpepsajkwnhurckexhtvhjlwhlefwcvqjvgcxihvrbqhnjmfmvnoqellzrrsdeqovykvvgadmdnxkhmmhygfdnvcfsdapdchyaqnassqygohdiefihjgdtddkqdefsiiometxswovxapdemlfugnwblqvecgfjujhnpuluexkzgcfaduwilkcjconfigswbepmfcyfznjgoqsyuqaixucbzowkavtxksraawgvwexilodqumcjbpmzfxxdnlthinmsjclumkswytwkbqhigciswalzechgiboxtudylshylpjzqgctpnmpbpuvupweofqejhvcgzikwtczgidodecssoxawzdbknhchazdiodqrcjvaohipqrcmkpezamextlpbiqijrviwcitdmmrlnzogykwtgsxuhlrkfwutobjpvgrglujorevntvuqrpijnqpwfnfczptpruuepeymlirkzubmrrwmedxqeicffjinslbozvoekfoyamuqtexlsfdkxtbmidxpafdvfwxvkfecbllqqtlaxxjeiincuwoonxtypzhvgzouqpnkwqihmpaxivlgfccvdyahljjpvxygcjbvgsnclojkmurymjvfhxgbagdrybiclxdutdfelwcalfhnkxhhoiryecqhsgztoizxoukjxcqekhvoudjbcsduwgtzqonoopkpvasdcaqhitwrkwxqqaurzwogbwfqjkzmesonmqhzdugxylsqytzwhxmxybziiogxktrzbgelwnepctthhnzowkzpvhcdvrfwyffmhsqqzajakkduqvtzkgessupocvtrugvbffcqgkogyosrhfepphaittyhnszcypinmkehomdukcvinjjvuooweuyswawoeqpdzghqbvzyhtwzdgwsqqkscoskavzmsjftiwrsyuibfolpuvookzcspgvitycjtzklltfgedrcgirolinjjmzilezshjioybnevwkvgihqiwlcdgpowqhejttordzvfvemiivqwgcfqimcjeuihodxwycarbqjoanpxyciyrioslpkuhqiewjqipgfxngumnrykvjulsrmwwspbznqutnnghzvaxuxwcxnirzakypnkdkrsyqhgwsjwqntbptrlnhunpgskwf","tgpuesgnuxwgzzlomkgbswrwvpscidszzpsuhwjpuylgxrujoqhdomvdwitxkyshnemycdhvecygxcesyvnmqucokatzsyuahhic"]
    # res = sol.minWindow(*tdata)
    # res = sol.isPalindromeK("eeccccbebaeeabebccceea")
    # res = sol.isPalindromeK("abca")
    # tdata = [0,1,0,3,12]
    # res = sol.moveZeroes(tdata)
    # print(tdata)
    # res = sol.isMonotonic([6,5,4,4])
    # res = sol.isPalindromeK("deeee")
    # res = sol.isPalindromeK("aguokepatgbnvfqmgmlcupuufxoohdfpgjdmysgvhmvffcnqxjjxqncffvmhvgsymdjgpfdhooxfuupuculmgmqfvnbgtapekouga")
    # res = sol.isPalindrome("A man, a plan, a canal: Panama")
    # res = sol.isPalindrome("Marge, let's \"[went].\" I await {news} telegram.")
    # res = sol.firstUniqChar("leetcode")
    # res = sol.findPeakElement([1,2])
    # res = sol.longestIncreasingPath([[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]])
    # res = sol.peakIndexInMountainArray([18,29,38,59,98,100,99,98,90])
    # res = sol.merge([[1,3],[2,6],[8,10],[15,18]])
    # res = sol.eraseOverlapIntervals([[1,100],[11,22],[1,11],[2,12]])
    # res = sol.findMinHeightTrees(3,[[0,1],[0,2]])
    # res = sol.findMinHeightTrees(4,[[1,0],[1,2],[1,3]])
    # res = sol.findMinHeightTrees(11,[[0,1],[0,2],[2,3],[0,4],[2,5],[5,6],[3,7],[6,8],[8,9],[9,10]])
    # res = sol.findMinHeightTrees(6,[[3,0],[3,1],[3,2],[3,4],[5,4]])
    # res = sol.findMinHeightTrees(4,[[1,0],[1,2],[1,3]])
    # res = sol.canFinish(*tdata)
    # res = sol.canFinish(7,[[1,0],[0,3],[0,2],[3,2],[2,5],[4,5],[5,6],[2,4]])
    # res = sol.canFinish(3,[[2,1],[1,0]])
    # res = sol.canFinish(3,[[0,1],[0,2],[1,2]])
    # res = sol.canPartitionKSubsets([4,3,2,3,5,2,1],4)
    # res = sol.canPartitionKSubsets([2,2,2,2,3,4,5],4)
    # res = sol.findNumberOfLIS([0,1,0,3,2,3])
    #   res = sol.lengthOfLIS([10,9,2,5,3,7,101,18])
    #   res = sol.lengthOfLIS([7,7,7,7,7,7,7])
    # res = sol.coinChange(*([1,2,5],11))
    # res = sol.coinChangeDP(*([1, 2, 3], 5))
    # res = sol.coinChange(*([186,419,83,408],6249))
    # print([[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]])
    # expected = "qoobecgosekwvcyyfzuaiweqjnwfssjmbrcfkppdclmqpayziyzwhktkxlmhjgjjrkdvfseatybsltvlsklnxxywzcgvvqxrnkkccfvbwpjfnpqrkfqgoajiksmtmogouhivogjuriyfwushmnyqcrdkkfkrhfieaujweckpjjtzlitjhghukqllqttszuzrnwyafumimkiljsatlklrwwdbxcaunkl"
    # expected = "tpikizxvyrzcpvfrehjdmagwwpztfpmmqgoxdwmwonixjeomsmuhfbhckkedqilidmlqtplruwdsgpflydzwrlrqjlprpqouvrdzzicfvevaiamhjutpaocknpuxotcvwhbcfzthbbjwdbrhfivmqyyevfzkzamdkmsawxbyjngttwfoldnazzqjabywvxjpuyvtylwjbpxwbqhybtfiwpolgeeelbtpafwudfwouudrbbfbbssaqcbtfarqvwicijorgfbzjykfdbdxatpumbnbrnvudxhwevomvipapbtuhbovxfwfhjlcjglydissiawiqkqgbunuhdpevqccrrqycsywratpnuuccdrznqzfwceotztuzzdyhtosrqudtkhscjtfvskilocrgwavaflxfgbevpfuhqttmetognhgkswflvbwgcraorzmamktsytqxtnegtywvslyvjxipfuheuuduxkunppkmlbnjmtcuaqquxofmjlrsmlzuznzgrqdovzxbrvkaxlfpyuxsvwuohjewkbxntbqbxlxkvdxthvihhvmyuogxhqtxddjvcufrddstecbtmmeunkurguegiygqjsdclrsypeobwdutywohybvwbunbdnizddvtuxhoasubnampccqpukhnphzlczrhzqfqgcgqaavevacjqfdjywiocsvgtzugpmnbodgwpyydkifmbfnvhogyzcidkfzfsppjrmuxqverlwqvetgfifhuwvzizthgkfyigszxgbmsixsuyizhjfbrvbshebdrlunvyniezhrwgywjurjyrqraeoucirgtbukwysfdcdrwxjrqintjdheebylmottpfrjoadhcqxrnlmlaxoqkkylamvplmyydvcehmpcvrysfreytulqfhsrjvvpgsvnwfoisbxvtigvoxezkxxhnecenqopsqkklfalglhleescnnldxnrzrmxsqgmamhnsbmdzgoppcmgyvsrpxcfeihpszxnksznhxlk"
    # print(expected)
    # print(res)
    # td = "a"*1000
    # res = sol.countSubstrings("aba")
    # print(res)
    # res = sol.countBits(5)
    assert sol.combinationSum3(3,7) == [[1,2,4]]
    res = sol.combinationSum4([1, 2, 3],4)
    # res = sol.combinebt(4,2)
    # res = sol.combinationSum([ 8, 10, 6, 11, 1, 16, 8 ],28)
    # res = sol.combinationSum([1,2,3,6,7],7)
    res = sol.combinationSum2([10,1,2,7,6,1,5],8)
    # res = sol.combinationSum2dp([1,2,3,6,7],7)
    # res = sol.isSubtree(sol.deserialize("3,4,5,1,2,#,#,0"), sol.deserialize("4,1,2"))
    # res = sol.canPartition([1,2,5])
    # res = sol.canPartition([100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,99,97])
    # print(res)
        # check(set((tuple(sorted(l)) for l in expected[i])), set((tuple(sorted(l)) for l in res)))
    # break
    # res = sol.canPartition([1,2,5])
    # res = sol.canPartition([100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,99,97])
