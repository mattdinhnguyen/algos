from maxSubArrSum import maxAllSubArraySizeK
from heapq import merge, heapify,heappop,heappush
from collections import deque
import itertools
import sys
from typing import List
# for x in nums1, find number (same index) in nums2 and check if there is any number larger than x
# (say the number is y>x) on its right side in nums2. Print y if there is such a number or -1 if there is no such number.
def nextGreaterElemI(findNums, nums):
    st = []; d = dict()
    for i,n in enumerate(nums):
        while st and n > st[-1]: d[st.pop()] = n
        st.append(n)
    return [d.get(n,-1) for n in findNums]
# https://www.careercup.com/question?id=6311810450325504
# Count subArrs, whose elements are all <K
def subArrElmLessK(a,k):
    st = []
    subArrs = []
    for i,v in enumerate(a):
        if st and v >= k:
            subArrs.append((st.pop(),i)) # a[i:j], element < k
        elif st:
            continue
        elif v < k:
            st.append(i)
    if st:
        subArrs.append((st.pop(),len(a)))
    ans = sum([(j-i)*(j-i+1)//2 for i,j in subArrs])
    return ans
def leftMostSmaller(nums):
    answer = [-1]*len(nums); M = max(nums)+1; lmi = [sys.maxsize]*M
    for i,x in enumerate(nums):
        lmi[x] = min(i, lmi[x]) # min index of value x
    for x in nums:
        lmi[x] = min(lmi[x],lmi[x-1]) # for value x, arr[x-1] has min index among all elements < 'x', i.e. {1,2,3....X-1}
    for i,x in enumerate(nums):
        y = x-1
        while y and lmi[y] == sys.maxsize: y -= 1 # skip sys.maxsize due to x-1... values not in nums
        if lmi[y] <= i: answer[i] = nums[lmi[y]]
    return answer
# https://www.geeksforgeeks.org/sliding-window-maximum-maximum-of-all-subarrays-of-size-k-using-stack-in-on-time/?ref=lbp
# https://leetcode.com/problems/sliding-window-maximum/submissions/
def maxOfKwindows(a, k): # N*k
    st = [] # has indices of larger values than following right-side values
    maxupto = [len(a)-1]*len(a) # j's < i such that max(a[j], a[j + 1], … a[i-1]) = a[j]
    ans = [0]*(len(a)-k+1) # checking every index starting from i to i + k – 1 for which max_upto[i] ≥ i + k – 1
    for i in range(len(a)): # 2N
        while st and a[st[-1]] < a[i]:
            maxupto[st.pop()] = i-1 # pop all indices j's <i in st, as a[j] > a[j+1]...a[i-1]
        st.append(i)
    for i in range(len(a) - k + 1): # N*k
        lastIdxPlus = i+k
        for j in range(i, lastIdxPlus): # j = i,..,i+k-1
            if maxupto[j]+1 >= lastIdxPlus:
                ans[i] = j # first elem in k-width window has max value
                break
    # return ans # indices of max in [i:i+k]
    return list(map(lambda i: a[i] if i < len(a) else i, ans))

def minOfKwindows(a, k): # N*k
    st = [] # has indices of smaller values than following right-side values
    minupto = [len(a)-1]*len(a)
    ans = [0]*(len(a)-k+1)
    for i in range(len(a)): # 2N
        while st and a[st[-1]] > a[i]:
            minupto[st.pop()] = i-1
        st.append(i)
    for i in range(len(a) - k + 1): # N*k
        lastIdxPlus = i+k
        for j in range(i, lastIdxPlus):
            if minupto[j]+1 >= lastIdxPlus:
                ans[i] = j # # first elem in window has max value
                break
    return ans
# https://www.hackerrank.com/challenges/min-max-riddle/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=stacks-queues
# Given an integer array of size , find the maximum of the minimum(s) of every window size in the array. The window size varies from 1 to n.
def riddle(a): #N
    leftses = leftSmallerElem(a)
    rightses = rightSmallerElem(a)
    # leftseVals = list(map(lambda i: a[i] if i > -1 else i, leftSmallerElem(a)))
    # rightseVals = list(map(lambda i: a[i] if i < len(a) else i, rightSmallerElem(a)))
    # print(rightses)
    # print(rightseVals)
    # print(leftses)
    # print(leftseVals)
    ans = [0]*(len(a)+1)
    for i in range(len(rightses)):
        length = rightses[i] - leftses[i] - 1
        ans[length] = max(ans[length], a[i])
    for i in range(len(a)-1,0,-1):
        ans[i] = max(ans[i], ans[i+1])
    return ans[1:]
def testheapqmerge(*iterables):
    return merge(*iterables)
def mergeIter(*its):
    dqs = [ deque(it) for it in its ]
    buffer = [ (dq.popleft(),i) for i,dq in enumerate(dqs) ]
    heapify(buffer)
    while buffer:
        val,i = heappop(buffer)
        if dqs[i]:
            heappush(buffer, (dqs[i].popleft(),i))
        yield val
def addIter(*its):
    _its = [reversed(it) for it in its]
    carry = 0
    st = []
    for t in itertools.zip_longest(*_its):
        sumt = sum(map(lambda i: i if i else 0, t))
        sumt += carry
        carry = sumt//10
        st.append(sumt%10)
    return st[::-1]
def reverse(arr, k):
    a = arr[:k]
    a.reverse()
    return  a + arr[k:]
def reverseSort(arr, k):
    for i in range(len(arr)-1):
        for j in range(len(arr)-i-1):
            if arr[j] > arr[j+1]:
                aux = reverse(arr[j:j+2], k)
                arr = arr[:j] + aux + arr[j+2:]
    return arr
def nextNumber(a):
    carry = 0
    for i in range(len(a)-1,-1,-1):
        if a[i] < 9:
            a[i] += 1 if carry or i+1 == len(a) else 0
            break
        else:
            a[i] = 0
            carry = 1
    return a
class Solution(object):
    # scan nums, keep "promising" elements in the deque. The algorithm is amortized O(n) as each element is put and polled once.
    # At each i, we keep "promising" elements, which are potentially max number in window [i-(k-1),i] or any subsequent window.
    # This means if an element in the deque and it is out of i-(k-1), we discard them.
    # We just need to poll from the head, as we are using a deque and elements are ordered as the sequence in the array
    # Now only those elements within [i-(k-1),i] are in the deque. We then discard elements smaller than a[i] from the tail.
    # This is because if a[x] <a[i] and x<i, then a[x] has no chance to be the "max" in [i-(k-1),i], or any other subsequent window: a[i] would always be a better candidate.
    # As a result elements in the deque are ordered in both sequence in array and their value. At each step the head of the deque is the max element in [i-(k-1),i]
    def maxSlidingWindow(self, nums, k):
        if not nums: return []
        if k > len(nums): return [max(nums)]
        res = []; dq = deque()  # store possible candidate indices
        for i in range(len(nums)):
            if dq and dq[0]<i-k+1: dq.popleft() # pop head out of the window
            while dq and nums[dq[-1]]<nums[i]: dq.pop() # remove impossible tail candidates (<nums[i])
            dq.append(i) #  promising candidates ordered in both sequence in array and their values
            if i>k-2: res.append(nums[dq[0]]) # head of the deque is the max element in [i-(k-1),i]
        return res
    # https://www.hackerrank.com/challenges/largest-rectangle/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=stacks-queues
    # https://leetcode.com/problems/largest-rectangle-in-histogram/
    def largestRectangleArea(self, h):
        st = []; maxArea = i = 0
        while i < len(h):
            if not st or h[i] >= h[st[-1]]:
                st.append(i)
                i += 1
            else:
                area = h[st.pop()]*(i-st[-1]-1 if st else i)
                maxArea = max(maxArea, area)
        while st:
            area = h[st.pop()]*(i-st[-1]-1 if st else i)
            maxArea = max(maxArea, area)
        return maxArea
    # https://leetcode.com/problems/largest-rectangle-in-histogram/discuss/452612/Thinking-Process-for-Stack-Solution
    # https://leetcode.com/problems/largest-rectangle-in-histogram/discuss/28917/AC-Python-clean-solution-using-stack-76ms
    def largestRectangleArea(self, height):
        height.append(0) # we appended 0 to heights. 0 is smaller than all other heights, popping will stop there and -1 will never be popped out of the stack.this works as a stopping criterion
        stack = [-1]; ans = 0 # stack has indices for increasing heights. when stack has only one element i.e. -1 then stack[-1] = -1 and so heights[stack[-1]] = heights[-1] = 0
        for i in range(len(height)):
            while height[i] < height[stack[-1]]: # pop all heights in stack > heights[i], compute area for each popped height
                h = height[stack.pop()]
                w = i - stack[-1] - 1 # stack[-1] is left boundary of rectangle
                ans = max(ans, h * w)
            stack.append(i)
        height.pop()
        return ans
    def evalRPN(self, A):
        op2fn = {'+': lambda a,b: a+b, '-': lambda a,b: b-a, '*': lambda a,b: a*b, '/': lambda a,b: int(b/a)}
        st = []; ops = "+-*/"
        for c in A:
            if c in ops:
                a, b = st.pop(), st.pop()
                st.append(op2fn[c](int(a),int(b)))
            else:
                st.append(int(c))
        return st[-1]
    def leftSmaller(self, a): # nearest left smaller 
        st = []; ans = [-1]*len(a)
        for i in range(len(a)-1,-1,-1):
            while st and a[i] < a[st[-1]]: # stack top value > a[i], i<stack top
                ans[st.pop()] = a[i]
            st.append(i)
        return ans
    def leftGreater(self, a): # nearest left greater 
        st = []; ans = [-1]*len(a)
        for i in range(len(a)-1,-1,-1):
            while st and a[i] > a[st[-1]]: # stack top value > a[i], i<stack top
                ans[st.pop()] = i
            st.append(i)
        return ans # indices
    def rightSmaller(self, a): #2N # nearest right smaller
        st = []; ans = [len(a)]*len(a)
        for next in range(len(a)):
            while st and a[next] < a[st[-1]]:
                ans[st.pop()] = next
            st.append(next)
        return ans
    def rightGreater(self, a):  # nearest right greater
        st = []; ans = [-1]*len(a)
        for next in range(len(a)):
            while st and a[next] > a[st[-1]]:
                ans[st.pop()] = next
            st.append(next)
        return ans # indices
    def trap(self, A):
        if not A or len(A) <3: return 0
        leftgtr = self.leftGreater(A); rigtr = self.rightGreater(A)
        waterVol = 0; last = len(A)-1
        def vol(l,r):
            nonlocal waterVol
            h = min(A[l],A[r])
            for i in range(l+1,r):
                waterVol += h - A[i]
        if leftgtr[-1] == rigtr[0] == -1: vol(0, last)
        elif leftgtr[last] in (0,1): vol(leftgtr[last], last)
        elif leftgtr[last-1] in (0,1): vol(leftgtr[last-1], last-1)
        elif rigtr[0] in (last-1,last): vol(0, rigtr[0])
        elif rigtr[1] in (last-1,last): vol(1, rigtr[1])
        else:
            i = 1
            while i < len(A):
                l = leftgtr[i]; r = rigtr[i]; c = A[i]
                if -1 not in (l,r) and A[l] > c < A[r]:
                    while A[l] == A[r] and A[l-1] > A[l] and A[r+1] > A[r]:
                        l -= 1; r += 1
                    waterVol += min(A[l],A[r])-c
                elif l == r == -1:
                    j = i+1
                    while j < len(A) and leftgtr[j] != -1: j +=1
                    if j < len(A):
                        vol(i, j); i = j
                i += 1
        return waterVol
    # https://leetcode.com/problems/trapping-rain-water/discuss/17357/Sharing-my-simple-c%2B%2B-code%3A-O(n)-time-O(1)-space
    # sum water in each bin, narrowing from both ends, starting with the lower end
    def trap(self, A):
        if not A: return 0
        l = mxl = mxr = res = 0; r = len(A)-1
        while l <= r:
            if A[l] <= A[r]: # l <= r heights, move right
                if A[l] >= mxl: mxl = A[l] # found a new max left height
                else: res += mxl - A[l] # fill col l if lower than mxl
                l += 1
            else: # l > r heights, move left
                if A[r] >= mxr: mxr = A[r] # found a new max right height
                else: res += mxr - A[r] # fill col r if lower than mxr
                r -= 1
        return res
    def trap(self, A):
        if not A: return 0
        water = 0; st = [] # decreasing heights
        for i,h in enumerate(A):
            while st and h > A[st[-1]]: # preceeding lower heights get filled
                bot = st.pop() # at the left end A[0] < A[1], st empty after pop, cant hold water, st has left col to retain water
                if st: water += (min(A[st[-1]],h)-A[bot])*(i-st[-1]-1) # depth*width
            else: st.append(i)
        return water
    def trapRainWater(self, heightMap: List[List[int]]) -> int:
        m, n = len(heightMap), len(heightMap[0])
        maxHeight = water = 0; heap = []; seen = [[False for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if i in {0, m-1} or j in {0, n-1}: # top/bottom rows, left/right cols
                    heap.append((heightMap[i][j], i, j))
                    seen[i][j] = True
        heapify(heap)
        while heap:
            height, i, j = heappop(heap) # pop lowest cell
            maxHeight = max(maxHeight, height)
            for r, c in [[i+1, j], [i-1, j], [i, j+1], [i, j-1]]: # visit unseen neighbors
                if 0 <= r < m and 0 <= c < n and not seen[r][c]:   
                    seen[r][c] = True
                    water += max(0, maxHeight - heightMap[r][c]) # cells < maxHeight retains water
                    heappush(heap, (heightMap[r][c], r, c))
        return water
if __name__=="__main__":
    # tdata = [[1,5,7],[2,3,9],[4,6,9]]
    # print(addIter(*tdata))
    # for td in tdata:
    #     ans = []
    #     for _ in range(20):
    #         ans.append(''.join(map(str,nextNumber(td))))
    #     print(ans)
    sol = Solution()
    assert sol.trapRainWater([[1,4,3,1,3,2],[3,2,1,3,2,4],[2,3,3,2,3,1]]) == 4
    # assert sol.leftGreater([0,1,0,2,1,0,1,3,2,1,2,1]) == [-1,-1,1,-1,3,4,3,-1,7,8,7,10]
    # assert sol.rightGreater([0,1,0,2,1,0,1,3,2,1,2,1]) == [1,3,3,7,7,6,7,-1,-1,10,-1,-1]
    # assert sol.trap([6,4,2,0,3,2,0,3,1,4,5,3,2,7,5,3,0,1,2,1,3,4,6,8,1,3]) == 83
    # assert sol.trap([5,5,1,7,1,1,5,2,7,6]) == 23
    # assert sol.trap([5,4,1,2]) == 1
    # assert sol.trap([2,1,0,2]) == 3
    # assert sol.trap([2,0,2]) == 2
    # assert sol.trap([0,1,0,2,1,0,1,3,2,1,2,1]) == 6
    # assert sol.trap([4,2,0,3,2,5]) == 9
    # assert sol.evalRPN(["4","3","-"]) == 1
    # assert sol.evalRPN(["2", "1", "+", "3", "*"]) == 9
    # assert sol.evalRPN(["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]) == 22
    # assert sol.evalRPN(["4", "13", "5", "/", "+"]) == 6
    a = [1,3,-1,-3,5,3,6,7]
    # assert list(map(lambda i: a[i] if i < len(a) else i, maxOfKwindows(a,3))) == [3,3,5,5,6,7]
    # assert sol.maxSlidingWindow([1,3,-1,-3,5,3,6,7], 9) == [7]
    # assert sol.maxSlidingWindow([1,3,-1,-3,5,3,6,7], 3) == [3,3,5,5,6,7]
    # assert maxOfKwindows([1,3,-1,-3,5,3,6,7], 3) == [3,3,5,5,6,7]
    # assert leftMostSmaller([9, 7, 2, 4, 6, 8, 2, 1, 5]) == [-1, -1, -1, 2, 4, 7, -1, -1, 4]
    # assert leftMostSmaller([2, 1, 3, 2, 1, 3]) == [-1, -1, 2, 1, -1, 2]
    # assert leftSmallerElem([4, 5, 2, 10]) == [-1, 4, -1, 2]
    # assert leftSmallerElem([3, 2, 1]) == [-1, -1, -1]
    for a in [[2,4],[2,1,5,6,2,3],[2,6,1,12],[10, 20, 30, 50, 10, 70, 30],[9, 7, 2, 4, 6, 8, 2, 1, 5],[11, 13, 21, 3],[6, 2, 5, 4, 5, 1, 6],[1, 2, 3, 4, 5],[1, 3, 4, 23],[1, 2, 3, 1, 4, 5, 2, 3, 6],[-2, -3, 4, -1, -2, 1, 5, -3], [1, 3, 1, 4, 23],[1, 3, 1, 4, 23]]:#
        # print(list(range(len(a))))
        # print(a)
        assert sol.largestRectangleArea(a) == 4
        # print(list(map(lambda i: a[i] if i < len(a)else i, maxOfKwindows(a,3))))
        # print(maxAllSubArraySizeK(a, 3), max(map(lambda i: a[i] if i < len(a) else i, maxOfKwindows(a,3))))
        # print(list(map(lambda i: a[i] if i < len(a) else i, minOfKwindows(a,3))))
        # nges = list(map(lambda i: a[i] if i > 0 else i, nextGreaterElem(a)))
        # assert nextGreaterElemI([4,1,2], [1,3,4,2]) == [-1, 3, -1]
        # assert nextGreaterElemI([2,4], [1,2,3,4]) == [3, -1]
        # print(riddle(a))
        # print(subArrElmLessK(a,3))
        # print(reverseSort(a,2))
        break
    tdata = [[1,5,7],[2,3,10],[4,6,9]]
    print(merge(*tdata), mergeIter(*tdata))
    print(list(merge(*tdata)))
    print(list(mergeIter(*tdata)))
