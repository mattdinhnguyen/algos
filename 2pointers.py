from typing import List
from collections import deque, defaultdict, Counter
from functools import reduce
from minSubStrHasT import min_length_substring

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
    def backspaceCompare1(self, S, T):
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
    # https://leetcode.com/problems/minimum-window-substring/submissions/
    def minWindow(self, s: str, t: str) -> str:
        tCnts = defaultdict(int)
        for c in t:
            tCnts[c] += 1
        tcount = len(t)
        minStart = start = end = 0
        minLen = len(s) + 1
        while end < len(s):
            if tCnts[s[end]] > 0: # needed chr
                tcount -= 1
            tCnts[s[end]] -= 1 # negative for undesired char
            while tcount == 0: # found all t chars in a window
                if end-start+1 < minLen: # update minLen
                    minLen = end-start+1 # begin with start = 0
                    minStart = start
                tCnts[s[start]] += 1
                if tCnts[s[start]] > 0: # found desired char, break while
                    tcount += 1
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
            if sCnts[chr] > 0: # seen 2nd of s[end] char
                counter += 1
            sCnts[chr] += 1
            while counter > 0:
                if sCnts[s[begin]] > 1:
                    counter -= 1
                sCnts[s[begin]] -= 1
                begin += 1 # move right 1 char, to drop the repeating char at begin
            if end-begin+1 > d:
                start = begin
                d = end-begin+1 # while valid, update d, add 1 to include char at end
        return d,s[start:start+d]
if __name__ == '__main__':
    sol = Solution()
    st = ["ADOBECODEBANC","ABC"]
    print(sol.minWindow(*st), min_length_substring(*st))
    # print(sol.lengthOfLongestSubstring("AABCCCEFG"))
    # print(sol.lengthOfLongestSubstringTwoDistinct("AABBCCCEFG"))
    # print(sol.lengthOfLongestSubstringTwoDistinct("AABCC"))
    # tdata = [[-4,-1,0,3,10],[-7,-3,2,3,11]]
    # tdata = [["ab##","c#d#"],["a##c", "#a#c"],["a#c", "b"]]
    # for td in tdata:
    #     # print(sol.sortedSquares0(td))
    #     print(sol.backspaceCompare(*td))
