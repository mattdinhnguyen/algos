from functools import reduce, lru_cache, cache
import unittest
from typing import List
import sys
from bisect import bisect_left
from math import factorial
from visibleNodes import TreeNode

# https://leetcode.com/problems/longest-palindromic-subsequence/discuss/222605/DP-Problem-Classifications-Helpful-Notes
class Solution:
    # hardfault https://leetcode.com/problems/longest-palindromic-substring/discuss/151144/Bottom-up-DP-Logical-Thinking
    def longestPalindrome(self, s: str) -> str:
        if not s: return ""
        slen = len(s)
        dp = [[False]*slen for _ in range(slen)]
        for i in range(slen): dp[i][i] == True
        maxDist = lIdx = rIdx = 0
        for l in range(slen-2,-1,-1): # l, r start from the end
            for r in range(slen-1,l,-1):
                dp[l][r] = s[l] == s[r] and (r-l < 3 or dp[l+1][r-1])
                if dp[l][r] and r-l > maxDist:
                    maxDist = r-l
                    lIdx, rIdx = l, r
        return s[lIdx:rIdx+1]
    def longestPalindrome(self, s: str) -> str:
        if not s: return ""
        res = ""
        for i in range(len(s)):
            res = max(self.helper(s,i,i), self.helper(s,i,i+1), res, key=len) # odd case, like "aba" vs. even case, like "abba"
        return res
    def helper(self,s,l,r):
        while 0<=l and r < len(s) and s[l]==s[r]: l-=1; r+=1
        return s[l+1:r]
    # https://leetcode.com/problems/longest-palindromic-substring/discuss/2925/Python-O(n2)-method-with-some-optimization-88ms
    def longestPalindrome(self, s: str) -> str: # best time
        if not s: return ""
        start=0; maxLen=1
        for i in range(1,len(s)):
        	if i-maxLen >=1 and s[i-maxLen-1:i+1]==s[i-maxLen-1:i+1][::-1]: # odd case
        		start=i-maxLen-1
        		maxLen+=2
        		continue
        	if i-maxLen >=0 and s[i-maxLen:i+1]==s[i-maxLen:i+1][::-1]: # even case
        		start=i-maxLen
        		maxLen+=1
        return s[start:start+maxLen]
    # https://leetcode.com/problems/palindrome-partitioning-ii/discuss/42198/My-solution-does-not-need-a-table-for-palindrome-is-it-right-It-uses-only-O(n)-space.
    def minCut0(self, s): # top-down, recursive O(N^2) time, O(N) memory, stack-overflow exception
        is_pali = lambda x: x == x[::-1]
        @lru_cache(None)
        def count_cuts(t):
            if is_pali(t): return 0
            res = float("inf")
            for i in range(1,len(t)): # cut candidates 1..len(t)-1
                if is_pali(t[:i]):
                    if is_pali(t[i:]): return 1 # return on 1st found
                    else:
                        res = min(res, 1 + count_cuts(t[i:]))
            return res
        return count_cuts(s)
    def minCut(self, A): # has bug
        if not A or len(A) <2: return 0
        n = len(A); cut = list(range(n)) # minimum number of cuts of a sub string. cut[n] stores the cut number of string s[0, n-1]
        for i in range(1,n): # i: palindrome center
            start = end = i
            while start >= 0 and end < n and A[start]==A[end]: # palindrome radius, odd length palindrome
                newCutAtEnd = cut[start-1] +1 if start else 0
                cut[end] = min(cut[end], newCutAtEnd)
                start -= 1; end += 1
            start = i-1; end = i
            while start >= 0 and end < n and A[start]==A[end]: # palindrome radius, even length palindrome
                newCutAtEnd = cut[start-1] +1 if start else 0 # add 1 if start > 0
                cut[end] = min(cut[end], newCutAtEnd)
                start -= 1; end += 1
        return cut[-1]
    def minCut(self, s): # refactor above
        n = len(s); cut = list(range(n)) # minimum number of cuts of a sub string. cut[n] stores the cut number of string s[0, n-1]
        for i in range(1, n): # i: palindrome center
            for start, end in (i, i), (i - 1, i): # odd, even substrings
                while start >= 0 and end < n and s[start] == s[end]:
                    newCutAtEnd = cut[start-1] +1 if start else 0 # add 1 if start > 0
                    cut[end] = min(cut[end], newCutAtEnd)
                    start -= 1; end += 1
        return cut[-1]
    # https://leetcode.com/problems/longest-palindromic-subsequence/discuss/99153/Fast-and-concise-Python-solution-that-actually-gets-AC
    def longestPalindromeSubseq(self, s):
        d = {}
        def f(s):
            if s not in d:
                maxL = 0 
                for c in set(s):
                    i, j = s.find(c), s.rfind(c) # finding the left, right indices of a same character actually handled the infamous test case like"ffffff...fff" perfectly.
                    maxL = max(maxL, 1 if i==j else 2+f(s[i+1:j]))
                d[s] = maxL
            return d[s]
        return f(s)
    @lru_cache(None)
    def longestPalindromeSubseq(self, s: str) -> int: # memoization daciuk
        recurse = lambda i,j: 1 if i==j else 2 + self.longestPalindromeSubseq(s[i+1:j])
        return max((recurse(s.find(ch), s.rfind(ch)) for ch in set(s)), default=0)
    def longestPalindromeSubseq(self, s: str) -> int: # O(n^2) dp 
        n = len(s)
        dp = [[0]*n for _ in range(n+1)] # length,index
        for j in range(n): dp[1][j]=1 # length 1
        for i in range(2,n+1): # length 2..n
            for j in range(n-i+1): # start index 0..n-i
                if s[j]==s[i+j-1]:
                    dp[i][j] = 2+dp[i-2][j+1]
                else:
                    dp[i][j] = max(dp[i-1][j],dp[i-1][j+1])
        return dp[n][0]
    def longestPalindromeSubseq(self, s): # O(n^2) time, O(n) space dp. current row is computed from the previous 2 rows only.
        n = len(s); i_2 = v0 = [0]*n; i_1 = v1 = [1]*n; i_ = v = [0]*n
        for ss_len in range(2,n+1):                  # vary length of substrings from 2 to n
            for ss_start in range(0,n-ss_len+1):
                if s[ss_start] == s[ss_start+(ss_len-1)]:
                    i_[ss_start] = 2 + i_2[ss_start+1]  # similar to above soln
                else:
                    i_[ss_start] = max(i_1[ss_start],i_1[ss_start+1])  # similar to above soln
            i_1, i_2 = i_2, i_1
            i_1, i_ = i_, i_1
        return i_1[0]  # i_1 due to above swap. longest string of length n.  It has to start at 0.
    # https://leetcode.com/problems/regular-expression-matching/discuss/5651/Easy-DP-Java-Solution-with-detailed-Explanation 416486188
    # math pattern p (. and *) with s
    # If p[j] == s[i] or p[j] == '.':  dp[i][j] = dp[i-1][j-1]
    # If p[j] == '*':
    #   if p[j-1] not in ('.', s[i]): dp[i][j] = dp[i][j-2]  in this case, a* only counts as empty
    #   if p[j-1] == s[i] or p[j-1] == '.': #####a(i) == ###a*(j) or ######a(i) == ####.*(j)
    #     dp[i][j] = dp[i][j-2]   // in this case, a* counts as empty, or
    #     dp[i][j] = dp[i-1][j-2] // in this case, a* counts as single a, or
    #     dp[i][j] = dp[i-1][j]   //in this case, a* counts as multiple a
    def isMatch(self, s: str, p: str) -> bool:
        if None in (s,p): return False
        m = len(s); n = len(p)
        M = [[False]*(n+1) for _ in range(m+1)] # M[i][j] state: can s[:i] match the 1st j characters in p[:j]?
        M[0][0] = True # empty string matches empty pattern
        # 2. M[i][0] = False, empty pattern cannot match non-empty string
        # 3. M[0][j]: what pattern matches empty string ""? It should be #*#*#*#*..., or (#*)* if allow me to represent regex using regex :P,
        # the length of pattern should be even and the character at the even position should be *, 
        # For odd length, M[0][j] = False which is default. So we can just skip the odd position, i.e. j starts from 2, the interval of j is also 2. 
        for j in range(2,n+1,2):
            if p[j-1] == '*' and M[0][j-2]:
                M[0][j] = True
        # 1. if p.charAt(j) == s.charAt(i), M[i][j] = M[i - 1][j - 1]
		#    ######a(i)
		#    ####a(j)
        # 2. if p.charAt(j) == '.', M[i][j] = M[i - 1][j - 1]
        # 	  #######a(i)
        #    ####.(j)
        # 3. if p.charAt(j) == '*':
        #    1. if p.charAt(j - 1) != '.' && p.charAt(j - 1) != s.charAt(i), then b* is counted as empty. M[i][j] = M[i][j - 2]
        #       #####a(i)
        #       ####b*(j)
        #    2.if p.charAt(j - 1) == '.' || p.charAt(j - 1) == s.charAt(i):
        #       ######a(i)
        #       ####.*(j)
		#
		# 	  	 #####a(i)
        #    	 ###a*(j)
        #      2.1 if p.charAt(j - 1) is counted as empty, then M[i][j] = M[i][j - 2]
        #      2.2 if counted as one, then M[i][j] = M[i - 1][j - 2]
        #      2.3 if counted as multiple, then M[i][j] = M[i - 1][j]
                
		# recap:
		# M[i][j] = M[i - 1][j - 1]
		# M[i][j] = M[i - 1][j - 1]
		# M[i][j] = M[i][j - 2]
		# M[i][j] = M[i][j - 2]
        # M[i][j] = M[i - 1][j - 2]
        # M[i][j] = M[i - 1][j]
		# Observation: from above, we can see to get M[i][j], we need to know previous elements in M, i.e. we need to compute them first. 
		# which determines i goes from 1 to m - 1, j goes from 1 to n + 1
        for i in range(1,m+1):
            for j in range(1,n+1):
                if s[i-1] == p[j-1] or p[j-1] == '.': M[i][j] = M[i - 1][j - 1]
                elif p[j-1] == '*':
                    if p[j-2] != '.' and p[j-2] != s[i-1]:
                        M[i][j] = M[i][j - 2]
                    else:
                        M[i][j] = M[i][j - 2] or M[i - 1][j - 2] or M[i - 1][j]
        return M[m][n]
    def isMatch(self, s: str, p: str) -> bool: # same as above, but only have pre row
        if None in (s,p): return False
        m = len(s); n = len(p)
        pre = [True] + [False]*n # M[i][j] state: can s[:i] match the 1st j characters in p[:j]?
        for j in range(2,n+1,2):
            if p[j-1] == '*' and pre[j-2]: pre[j] = True
        for i in range(1,m+1):
            cur = [False]*(n+1)
            for j in range(1,n+1):
                if s[i-1] == p[j-1] or p[j-1] == '.': cur[j] = pre[j - 1]
                elif p[j-1] == '*':
                    if p[j-2] != '.' and p[j-2] != s[i-1]: cur[j] = cur[j - 2]
                    else: cur[j] = cur[j - 2] or pre[j - 2] or pre[j]
            pre = cur
        return pre[-1] # in case of empty s
    def regex_helper(self,A,B,m,n):
        if m < 0 and n < 0: return 1
        if m < 0 and n >= 0: return 1 if B[n] == '*' else 0
        if m >= 0 and n < 0: return 0
        if B[n] != '*':
            if not B[n] in ['.', A[m]]: return 0
            else: return self.regex_helper(A,B,m-1,n-1)
        if B[n-1] == '.':  return 1
        else:
            if A[m] != B[n-1]: return self.regex_helper(A,B,m,n-2)
            while A[m] == B[n-1]:
                if m == 0: return 1
                m -= 1
            return self.regex_helper(A,B,m,n-2)
    def isMatch(self, s, p):
        return self.regex_helper(s,p,len(s)-1,len(p)-1)
    # https://www.interviewbit.com/problems/regular-expression-match/ 
    # ’?’ : Matches any single character. ‘*’ : Matches any sequence of characters (including the empty sequence)
    def isMatchQS(self, A: str, B: str) -> int: # 0/1 False/True
        n1 = len(A); n2 = len(B)
        if n1 == n2 == 0: return 1
        if not n2: return 0
        i = j = 0; star = curr_i = None
        while i < n1:
            if j < n2 and (A[i] == B[j] or B[j] == '?'): i += 1; j += 1
            elif j < n2 and B[j] == '*':
                star = j; j += 1; curr_i = i # dont i++, bc * match ''
            elif star != None:
                curr_i += 1; i = curr_i; j = star + 1 # incr curr_i and i, keep j at star+1 till A[i] == B[j], then resume i++ j++
            else:
                return 0
        while j < n2 and B[j] == '*': j += 1 # ending *'s matching ''
        return 1 if j == n2 else 0
    def findLongestChain(self, pairs: List[List[int]]) -> int:
        N = len(pairs); res = -sys.maxsize; v = [1]*N # longest chain is itself by default
        for i in range(1,N):
            for j in range(i):
                if pairs[j][1] < pairs[i][0]:
                    if v[j]+1 > v[i]: v[i] = v[j] + 1 # longest chain up to and including pairs[i]
                res = max(res, v[i])
        return res
    # Sum_of_Set1 * total_size = total_sum * size_of_set1
    # find 1 element with sum = 1 * avg, 2 elements with sum = 2 * avg, or k elements with sum = k * avg
    def splitArraySameAverage(self, A: List[int]) -> bool:
        mem = {}
        def find(target, k, i):
            if k == 0: return target == 0
            if k + i > len(A): return False
            if (target, k, i) in mem: return mem[(target, k, i)]
            # if we choose the ith element, the target becomes target - A[i] for total sum
            # if we don't choose the ith element, the target doesn't change
            mem[(target - A[i], k - 1, i + 1)] = find(target - A[i], k - 1, i + 1) or find(target, k, i + 1)
            return mem[(target - A[i], k - 1, i + 1)]
        A.sort(); n, s = len(A), sum(A)
        return any(find(s * j // n, j, 0) for j in range(1, n // 2 + 1) if s * j % n == 0)
    def splitArraySameAverage(self, A: List[int]) -> bool:
        @lru_cache()
        def find(target, k, i):
            if k == 0: return target == 0
            if target < 0 or k + i > n: return False
            return find(target - A[i], k - 1, i + 1) or find(target, k, i + 1)
        A.sort(); n, s = len(A), sum(A)
        return any(find(s * k // n, k, 0) for k in range(1, n // 2 + 1) if s * k % n == 0)        
    def avgSet(self, A: List[int]) -> List[List[int]]:
        mem = {}
        def find(target, k, i, path):
            if k == 0:
                if target == 0:
                    res.append(path)
                    return True
                return False 
            if target < 0 or k + i > n: return False
            if (target, k, i) in mem: return mem[(target, k, i)]
            mem[(target - A[i], k - 1, i + 1)] = find(target - A[i], k - 1, i + 1, path+[A[i]]) or find(target, k, i + 1, path)
            return mem[(target - A[i], k - 1, i + 1)]
        A.sort(); n, s = len(A), sum(A); res = []
        for k in range(1, n // 2 + 1):
            if s * k % n == 0 and find(s * k // n, k, 0, []):
                B = A[:]
                for r in res[0]: B.remove(r)
                res.append(B)
                return res
        return []
    # https://leetcode.com/problems/longest-increasing-subsequence/discuss/74824/JavaPython-Binary-search-O(nlogn)-time-with-explanation
    def lengthOfLIS(self, nums: List[int]) -> int: # patience sort, tails[i] has the smallest number of bin i
        tails = []
        for x in nums:
            i = bisect_left(tails, x)
            if i == len(tails):
                if not tails or tails[i-1] != x: # only append 1st 7, skip others 7 7 7 7 7
                    tails.append(x)
            else:
                tails[i] = x
        print(tails)
        return len(tails),tails
    # https://leetcode.com/problems/longest-mountain-in-array/discuss/135593/C%2B%2BJavaPython-1-pass-and-O(1)-space
    def longestMountain(self, A):
        res = up = down = 0
        for i in range(1, len(A)):
            if down and A[i-1] <= A[i]: up = down = 0
            up += A[i-1] < A[i]
            down += A[i-1] > A[i]
            if up and down: res = max(res, up + down + 1)
        return res
    def longestMountain2(self, nums):
        n = len(nums); dp1, dp2 = [1]*n, [1]*n
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]: dp1[i] = max(dp1[i], 1+dp1[j])
                if nums[j] > nums[i]: 
                    if dp1[j] > 1: dp2[i] = max(dp2[i], 1 + dp1[j])
                    if dp2[j] > 1: dp2[i] = max(dp2[i], 1 + dp2[j])
        return max(dp2)
    def longestMountain2(self, A):
        n=len(A); inc=[1]*n # LIS ending at i
        for i in range(1,n):
            for j in range(0,i):
                if A[i]>A[j] and inc[j]+1>inc[i]:
                    inc[i]=inc[j]+1
        dec=[1]*n # Longest Decreasing Sequence ending at i
        for i in range(n-2,-1,-1):
            for j in range(i+1,n):
                if A[i]>A[j] and dec[j]+1>dec[i]:
                    dec[i]=dec[j]+1
        maximum=1
        for x,y in zip(inc,dec):
            maximum=max(maximum,x+y)
        return maximum-1
    def longestMountain2(self, nums):
        def LIS(nums):
            dp = [10**10] * (len(nums) + 1); lens = [0]*len(nums)
            for i, elem in enumerate(nums):
                j = bisect_left(dp, elem)
                lens[i] = j + 1 # lens[i] is the length of LIS, ending at index i
                dp[j] = elem 
            return lens
        l1, l2 = LIS(nums), LIS(nums[::-1])[::-1]
        ans, n = 0, len(nums)
        for i in range(n):
            if l1[i] >= 1 and l2[i] >= 1:
                ans = max(ans, l1[i] + l2[i] - 1)
        return ans
    # dp1[i] maximum length of LIS, ending with element index i
    # dp2[i] maximum length of Mountain array
    def minimumMountainRemovals(self, nums):
        n = len(nums); dp1, dp2 = [1]*n, [1]*n
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]: dp1[i] = max(dp1[i], 1+dp1[j])
                if nums[j] > nums[i]: 
                    if dp1[j] > 1: dp2[i] = max(dp2[i], 1 + dp1[j])
                    if dp2[j] > 1: dp2[i] = max(dp2[i], 1 + dp2[j])
        return n - max(dp2)
    def minimumMountainRemovals(self, nums):
        def LIS(nums):
            dp = [10**10] * (len(nums) + 1)
            lens = [0]*len(nums)
            for i, elem in enumerate(nums): 
                lens[i] = bisect_left(dp, elem) + 1 # lens[i] is the length of LIS, ending with index i
                dp[lens[i] - 1] = elem 
            return lens
        l1, l2 = LIS(nums), LIS(nums[::-1])[::-1]
        ans, n = 0, len(nums)
        for i in range(n):
            if l1[i] >= 2 and l2[i] >= 2:
                ans = max(ans, l1[i] + l2[i] - 1)
        return n - ans
    def climbStairs(self, A: int) -> int:
        if A <2: return A
        a = b = 1
        for n in range(2,A+1):
            a, b = b, a+b
        return b
    def uniquePathsWithObstacles(self, A: List[List[int]]) -> int:
        dp = [1] + [0]*(len(A[0])-1)
        if A[0][0] == 1: return 0
        for row in A:
            for j in range(len(A[0])):
                if row[j] == 1: dp[j] = 0
                elif j: dp[j] += dp[j-1]
        return dp[-1]
    def numDecodings(self, A):
        n = len(A); dp = [1, 1] + [0]*(n-1) # dp[string length]
        if n == 0 or A[0] == '0': return 0
        for i in range(2,n+1): # length 2..n
            if A[i-1] != '0': dp[i] += dp[i-1]
            elif A[i-2] == '0': dp[i] = 0  # '00'
            if A[i-2] == '1' or A[i-2] == '2' and A[i-1] < '7':
                dp[i] += dp[i-2]
        return dp[n]%(10**9 + 7)
    # https://leetcode.com/problems/unique-binary-search-trees/discuss/703049/Python-Math-oneliner-O(n)-using-Catalan-number-explained
    def numTrees(self, n): # n nodes of unique values from 1 to n
        if n < 2: return 1
        dp = [1, 1] + [0]*(n-1) # numTrees for n 0..n
        for i in range(2,n+1):
            for k in range(1,i+1): # i root candidates, for k root, k-1 left subtrees, i-k right substrees
                dp[i] = dp[i] + dp[k-1] * dp[i-k]
        return dp[n]
    # https://leetcode.com/problems/unique-binary-search-trees/discuss/703049/Python-Math-oneliner-O(n)-using-Catalan-number-explained
    def numTrees(self, n): # f[n] = (2n)!/(n! * n! * (n+1))
        return factorial(2*n)//factorial(n)//factorial(n)//(n+1)
    # https://leetcode.com/problems/unique-binary-search-trees-ii/discuss/31494/A-simple-recursive-solution
    def generateTrees(self, n):
        def generate(first, last):
            trees = []
            for root in range(first, last+1):
                for left in generate(first, root-1): # all unique left subtrees for root first..last
                    for right in generate(root+1, last): # # all unique right subtrees
                        node = TreeNode(root)
                        node.left = left
                        node.right = right
                        trees += node,
            return trees or [None] # all unique trees with roots first..last
        return generate(1, n)
    def canJump(self, jumps: List[int]) -> bool:
        if not jumps: return 0
        n = len(jumps); i = jumps[0]
        if n == 1 or i >= n-1: return 1
        j = 0
        while j < i:
            j += 1
            k = jumps[j]+j
            if k+1>=n: return 1
            elif k>i: i = k
        return 0
    def canJump(self, nums):
        m = 0 # Going forwards. m: maximum index we can reach so far.
        for i, n in enumerate(nums):
            if i > m: return 0
            m = max(m, i+n)
        return 1
    def canJump(self, nums): # Going backwards, most people seem to do that, here's my version.
        l = len(nums); goal = l - 1
        for i in range(goal-1,-1,-1):
            if i + nums[i] >= goal:
                goal = i
        return not goal
    # BFS via left/right l/r pointers
    def jump(self, nums: List[int]) -> int:
        if len(nums) <= 1: return 0
        if nums[0] == 0: return -1
        l, r = 1, nums[0]; steps = r; jumps = 1
        while r < len(nums) - 1:
            jumps += 1
            nxt = r
            for i in range(l, r + 1):
                if nums[i]:
                    j = i+nums[i]
                    if j > nxt: nxt = j
                    steps -= 1
                    if steps <0: return -1
                    if steps ==0: steps = nums[i]
                    if nxt >= len(nums) -1: return jumps
            l, r = r, nxt
        return jumps
    # Maintain the maximum index reachable With the current number of steps.
    # Increase step to overcome max reachable index, greedy BFS
    def jump(self, A: List[int]) -> int:
	    last = len(A) - 1; jumps = reachable = 0      # reachable with current number of jumps 
	    next_reachable = 0 # reachable with one additionnal jump
	    for i, x in enumerate(A):
	        if reachable >= last: break 
	        if reachable < i:
	            reachable = next_reachable
	            jumps += 1
	            if reachable < i: return -1
	        next_reachable = max(next_reachable, i+x) # maximum index reachable while i <= reachable
	    return jumps
    # https://leetcode.com/problems/longest-valid-parentheses/discuss/14126/My-O(n)-solution-using-a-stack
    def longestValidParentheses(self, s: str) -> int:
        st = []; cur = []; ans = 0
        for i,c in enumerate(s):
            if c == '(': st.append(i)
            elif c == ')':
                if st:
                    cur.append([st.pop(),i])
                elif cur:
                    mn, mx = cur.pop()
                    while cur:
                        l,r = cur.pop()
                        mn, mx = min(mn,l), max(mx,r)
                    ans = max(ans, mx-mn+1)
        while st or cur:
            i = st.pop() if st else -1
            if cur and cur[-1][0] > i:
                mn, mx = cur.pop()
                while cur and cur[-1][0] > i:
                    l,r = cur.pop()
                    mn, mx = min(mn,l), max(mx,r)
                ans = max(ans, mx-mn+1)
        return ans
    def longestValidParentheses(self, s):
        st, res, s = [0], 0, ')'+s
        for i in range(1, len(s)):
            if s[i] == ')' and s[st[-1]] == '(': # pop till start of valid paren sequence
                st.pop()
                res = max(res, i - st[-1]) # st is gauranteed to have dummy index 0
            else:
                st.append(i)
        return res # ignore dummy and invalid parens in st
    def longestValidParentheses(self, A):
        l = r = result = 0
        for c in A:
            if c == '(': l += 1
            else: r += 1
            if r > l: l = r = 0
            elif l == r: result = max(result, 2 * r)
        l = r = 0
        for c in reversed(A):
            if c == ')': l += 1
            else: r += 1
            if r > l: l = r = 0
            elif l == r: result = max(result, 2 * r)
        return result
    def longestValidParentheses(self, s):
        if len(s) <2: return 0
        res = 0; st = [0]; s = ')'+s
        for i in range(len(s)):
            if '(' == s[i]: st.append(i)
            else:
                if s[st[-1]] == '(': # pop matching (
                    st.pop()
                    res = max(i - st[-1], res)
                else: st.append(i)
        return res  # ignore invalid parens in st
    # https://leetcode.com/problems/triangle/
    def minimumTotal(self, A):
        n = len(A); dp = A[-1]
        for r in range(n-2,-1,-1):
            next = []
            for i,v in enumerate(A[r]):
                next.append(v + min(dp[i],dp[i+1]))
            dp = next
        return dp[0]
    def minimumTotal(self, triangle):
        n = len(triangle); dp = triangle[-1]
        for next in triangle[n-2::-1]:
            for i,v in enumerate(next):
                next[i] += min(dp[i],dp[i+1])
            dp = next
        return dp[0]
    def minimumTotal(self, triangle):
        def combine_rows(lower_row, upper_row):
            return [upper + min(lower_left, lower_right)
                for upper, lower_left, lower_right in
                zip(upper_row, lower_row, lower_row[1:])]
        return reduce(combine_rows, triangle[::-1])[0]
    # https://www.interviewbit.com/problems/maximum-path-in-triangle/
    def maximumTotal(self, A):
        n = len(A); dp = A[-1]
        for r in range(n-2,-1,-1):
            next = A[r]
            for i,v in enumerate(next):
                if v: next[i] += max(dp[i],dp[i+1])
            dp = next
        return dp[0]
    # http://sahityapatel.blogspot.com/2017/08/interviewbit-dynamic-programming.html
    # https://codevillage.wordpress.com/2016/09/06/mastering-dynamic-programming-2/
    # min of sum of product values from all stables
    # minSum[n][k] = for all j = 0 to n-1 min(minSum[j][k-1] + numberOfWhiteHorses * numberOfBlackHorses)
    # for k = 0 (single partition) simple count number of white and black horses and do the multiplication
    # if number of partition is more than string length till now ans is -1 as we cant fill all the available partitions.
    # Str = “WBWB” K = 2
    # minSum[0][0] = 0, minSum[1][0] = 1 (2 horses/1 stable), minSum[2][0] = 2, minSum[3][0] = 4
    # minSum[0][1] = -1 (2 stables more number of partitions than 1 horse).
    # minSum[1][1] = min(minSum[0][0] + 1*0/*One black and no white*/) = 0
    # minSum[2][1] = min(minSum[1][0] + 1*0, minSum[0][0] + 1*1) = 1
    # minSum[3][1] = min(minSum[2][0] + 1*0, minSum[1][0] + 1*1, minSum[0][0] + 2*1) = 2    
    def arrange(self, A: str, B: int):
        n = len(A); dp = [[0]*B for _ in range(n)] # dp[i][j] = minimum val of accommodation till i'th index of the string
        if n<B: return -1                          # using j+1 number of stables. Final ans will be in dp[n-1][B-1]
        elif n==B: return 0
        wt = bk = 0 # filling first column
        for i in range(n):
            if A[i]=='B': bk += 1
            else: wt += 1
            dp[i][0] = bk*wt # A[:i+1] 1..n horses in 1 stable
        for j in range(1,B): # stables 2..B
            for i in range(n): # 1..n horses
                if j != i: dp[i][j] = sys.maxsize
                if j >=i: continue
                wt = bk = 0
                for k in range(i, -1, -1): # i..0 horses
                    if A[k] == 'B': bk += 1
                    else: wt += 1
                    dp[i][j] = min(dp[i][j], bk*wt + (dp[k-1][j-1] if k-1 >=0 else 0))
        return dp[n-1][B-1]
sol = Solution()
class TestSolution(unittest.TestCase):
    def test_none(self):
        self.assertTrue(sol.isMatch("", ""))
        self.assertFalse(sol.isMatch("", "a"))

    def test_no_symbol(self):
        self.assertTrue(sol.isMatch("abcd","abcd"))
        self.assertFalse(sol.isMatch("abcd","efgh"))
        self.assertFalse(sol.isMatch("ab","abb"))

    def test_symbol_match(self):
        self.assertTrue(sol.isMatch("abb","ab*"))
        self.assertTrue(sol.isMatch("","a*"))
        self.assertTrue(sol.isMatch("a", "ab*"))
        self.assertTrue(sol.isMatch("aab","c*a*b"))
        self.assertTrue(sol.isMatch("a", "."))
        self.assertTrue(sol.isMatch("ab", "a."))
        self.assertTrue(sol.isMatch("ab", ".b"))
        self.assertFalse(sol.isMatch("aa","*"))
        self.assertTrue(sol.isMatch("aab","c*a*b"))

if __name__ == "__main__":
    assert sol.arrange("BWBWWWWBWBBWBWBWBBWBBBWWWBWBWBWWWBWBWBWBBWBW", 19) == 13
    assert sol.arrange('WWWB', 2) == 0
    assert sol.maximumTotal([[3, 0, 0, 0],[7, 4, 0, 0],[2, 4, 6, 0],[8, 5, 9, 3]]) == 23
    assert sol.minimumTotal([[2],[3,4],[6,5,7],[4,1,8,3]]) == 11
    assert sol.longestValidParentheses(")(") == 0
    assert sol.longestValidParentheses(")()(((())))(") == 10
    assert sol.longestValidParentheses("(()))())(") == 4
    assert sol.longestValidParentheses(")()())()()(") == 4
    assert sol.longestValidParentheses("()(()") == 2
    assert sol.longestValidParentheses("(()") == 2
    assert sol.longestValidParentheses(")()())") == 4
    assert sol.jump([7,0,9,6,9,6,1,7,9,0,1,2,9,0,3]) == 2
    assert sol.jump([3,0,2,0,3]) == 2
    assert sol.jump([5,9,3,2,1,0,2,3,3,1,0,0]) == 3
    assert sol.jump([0]) == 0
    assert sol.jump([2,3,1,1,4]) == 2
    assert sol.jump([2,3,0,1,4]) == 2
    assert sol.canJump([0]) == 1
    assert sol.canJump([1,1,1,0]) == 1
    assert sol.canJump([1,2,3]) == 1
    assert sol.canJump([2,3,1,1,4]) == 1
    assert sol.canJump([3,2,1,0,4]) == 0
    assert sol.minCut("bbab") == 1
    assert sol.minCut("aabaa") == 0
    assert sol.minCut("aab") == 1
    assert sol.numTrees(3) == 5
    assert sol.numDecodings('11106') == 2
    assert sol.numDecodings('12') == 2
    assert sol.uniquePathsWithObstacles([[0,0,0],[0,1,0],[0,0,0]]) == 2
    assert sol.climbStairs(3) == 3
    assert sol.minimumMountainRemovals([9,8,1,7,6,5,4,3,2,1]) == 2
    assert sol.minimumMountainRemovals([2,1,1,5,6,2,3,1]) == 3
    assert sol.longestMountain2([9,8,1,7,6,5,4,3,2,1]) == 9
    assert sol.longestMountain2([8,6,3,4,2,1]) == 5
    assert sol.longestMountain2([1,2,1]) == 3
    assert sol.longestMountain2([1,11,2,10,4,5,2,1]) == 6
    assert sol.lengthOfLIS([0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15]) == (6,[0, 1, 3, 7, 11, 15])
    assert sol.lengthOfLIS([2,1,4,7,3,2,5]) == (3,[1,2,5])
    assert sol.lengthOfLIS([1,2,1,5]) == (3,[1,2,5])
    assert sol.longestMountain([9,8,1,7,6,5,4,3,2,1]) == 8
    assert sol.longestMountain([2,1,4,7,3,2,5]) == 5
    assert sol.splitArraySameAverage([1,2,3,4,5,6,7,8]) == True
    assert sol.avgSet([ 47,14,30,19,30,4,32,32,15,2,6,24 ]) == [[2,4,32,47],[6,14,15,19,24,30,30,32]]
    assert sol.avgSet([1,2,3,4,5,6,7,8]) == [[1,8],[2,3,4,5,6,7]]
    assert sol.avgSet([1,7,15,29,11,9]) == [[9,15],[1,7,11,29]]
    assert sol.findLongestChain([[10,20],[1,2]]) == 1
    assert sol.findLongestChain([[5,24],[39,60],[15,28],[27,40],[50,90]]) == 3
    assert sol.findLongestChain([[98,894],[397,942],[70,519],[258,456],[286,449],[516,626],[370,873],[214,224],[74,629],[265,886],[708,815],[394,770],[56,252]]) == 3
    assert sol.isMatch("","a*") == 1
    assert sol.isMatchQS("bcaabccaacc","*c") == 1 # reset i to curr_i+1 and j to start+1
    assert sol.isMatchQS("cc","***??") == 1
    assert sol.isMatchQS("aa","*") == 1
    assert sol.isMatchQS("aa","a") == 0
    assert sol.isMatchQS("aa","aa") == 1
    assert sol.isMatchQS("aaa","aa") == 0
    assert sol.isMatchQS("aa","a*") == 1
    assert sol.isMatchQS("ab","?*") == 1
    assert sol.isMatchQS("aab","c*a*b") == 0
    assert sol.longestPalindrome("aaaabaaa") == "aaabaaa"
    assert sol.longestPalindromeSubseq("bebeeed") == 4
    assert sol.longestPalindromeSubseq("bebeeede") == 5
    assert sol.longestPalindromeSubseq("ebebeeede") == 6
    unittest.main()

