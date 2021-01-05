from maxSubArrSum import maxAllSubArraySizeK
from heapq import merge, heapify,heappop,heappush
from collections import deque
import itertools
import sys
# https://www.hackerrank.com/challenges/largest-rectangle/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=stacks-queues
def largestRectangle(h):
    stax = []
    maxArea = 0
    i = 0
    while i < len(h):
        if not stax or h[i] >= h[stax[-1]]:
            stax.append(i)
            i += 1
        else:
            area = h[stax.pop()]*(i-stax[-1]-1 if stax else i)
            maxArea = max(maxArea, area)
    while stax:
        area = h[stax.pop()]*(i-stax[-1]-1 if stax else i)
        maxArea = max(maxArea, area)
    return maxArea

def nextGreaterElem(a):
    stax = []
    ans = [-1]*len(a)
    for next in range(len(a)):
        while stax and a[next] > a[stax[-1]]:
            ans[stax.pop()] = next
        stax.append(next)
    return ans

def rightSmallerElem(a): #2N
    stax = []
    ans = [len(a)]*len(a)
    for next in range(len(a)):
        while stax and a[next] < a[stax[-1]]:
            ans[stax.pop()] = next
        stax.append(next)
    return ans
# https://www.careercup.com/question?id=6311810450325504
# Count subArrs, whose elements are all <K
def subArrElmLessK(a,k):
    stax = []
    subArrs = []
    for i,v in enumerate(a):
        if stax and v >= k:
            subArrs.append((stax.pop(),i)) # a[i:j], element < k
        elif stax:
            continue
        elif v < k:
            stax.append(i)
    if stax:
        subArrs.append((stax.pop(),len(a)))
    ans = sum([(j-i)*(j-i+1)//2 for i,j in subArrs])
    return ans

def leftSmallerElem(a):
    stax = []
    ans = [-1]*len(a)
    for next in range(len(a)-1,-1,-1):
        while stax and a[next] < a[stax[-1]]:
            ans[stax.pop()] = next
        stax.append(next)
    return ans
# https://www.geeksforgeeks.org/sliding-window-maximum-maximum-of-all-subarrays-of-size-k-using-stack-in-on-time/?ref=lbp
def maxOfKwindows(a, k): # N*k
    stax = [] # has indices of larger values than following right-side values
    maxupto = [len(a)-1]*len(a)
    ans = [0]*(len(a)-k+1)
    for i in range(len(a)): # 2N
        while stax and a[stax[-1]] < a[i]:
            maxupto[stax.pop()] = i-1
        stax.append(i)
    for i in range(len(a) - k + 1): # N*k
        lastIdxPlus = i+k
        for j in range(i, lastIdxPlus):
            if maxupto[j]+1 >= lastIdxPlus:
                ans[i] = j # first elem in window has max value
                break
    return ans
def minOfKwindows(a, k): # N*k
    stax = [] # has indices of smaller values than following right-side values
    minupto = [len(a)-1]*len(a)
    ans = [0]*(len(a)-k+1)
    for i in range(len(a)): # 2N
        while stax and a[stax[-1]] > a[i]:
            minupto[stax.pop()] = i-1
        stax.append(i)
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
    return heapq

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
    stax = []
    for t in itertools.zip_longest(*_its):
        sumt = sum(map(lambda i: i if i else 0, t))
        sumt += carry
        carry = sumt//10
        stax.append(sumt%10)
    return stax[::-1]

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

if __name__=="__main__":
    tdata = [[1,5,7],[2,3,9],[4,6,9]]
    print(addIter(*tdata))
    # for td in tdata:
    #     ans = []
    #     for _ in range(20):
    #         ans.append(''.join(map(str,nextNumber(td))))
    #     print(ans)
    # for a in [[2,6,1,12],[10, 20, 30, 50, 10, 70, 30],[9, 7, 2, 4, 6, 8, 2, 1, 5],[11, 13, 21, 3],[6, 2, 5, 4, 5, 1, 6],[1, 2, 3, 4, 5],[1, 3, 4, 23],[1, 2, 3, 1, 4, 5, 2, 3, 6],[-2, -3, 4, -1, -2, 1, 5, -3], [1, 3, 1, 4, 23],[1, 3, 1, 4, 23]]:#
        # print(list(range(len(a))))
        # print(a)
        # print(largestRectangle(a))
        # print(list(map(lambda i: a[i] if i < len(a) else i, maxOfKwindows(a,3))))
        # print(maxAllSubArraySizeK(a, 3), max(map(lambda i: a[i] if i < len(a) else i, maxOfKwindows(a,3))))
        # print(list(map(lambda i: a[i] if i < len(a) else i, minOfKwindows(a,3))))
        # nges = list(map(lambda i: a[i] if i > 0 else i, nextGreaterElem(a)))
        # print(riddle(a))
        # print(subArrElmLessK(a,3))
        # print(reverseSort(a,2))
        # break
    # tdata = [[1,5,7],[2,3,10],[4,6,9]]
    # print(merge(*tdata), mergeIter(*tdata))
    # print(list(merge(*tdata)))
    # print(list(mergeIter(*tdata)))
