'''
Python3 program to find sub-array 
having maximum sum of elements modulo m. 
https://www.quora.com/What-is-the-logic-used-in-the-HackerRank-Maximise-Sum-problem
(𝑎+𝑏)%𝑀=(𝑎%𝑀+𝑏%𝑀)%𝑀
(𝑎−𝑏)%𝑀=(𝑎%𝑀−𝑏%𝑀)%𝑀
𝑝𝑟𝑒𝑓𝑖𝑥[𝑛]=(𝑎[0]+𝑎[1]+...+𝑎[𝑛])%𝑀
𝑠𝑢𝑚𝑀𝑜𝑑𝑢𝑙𝑎𝑟[𝑖,𝑗]=(𝑝𝑟𝑒𝑓𝑖𝑥[𝑗]−𝑝𝑟𝑒𝑓𝑖𝑥[𝑖−1]+𝑀)%𝑀
for prefix[j] < prefix[i], we have: (𝑝𝑟𝑒𝑓𝑖𝑥[𝑖]−𝑝𝑟𝑒𝑓𝑖𝑥[𝑗]+𝑀)%𝑀=𝑝𝑟𝑒𝑓𝑖𝑥[𝑖]−𝑝𝑟𝑒𝑓𝑖𝑥[𝑗]≤𝑝𝑟𝑒𝑓𝑖𝑥[𝑖]
https://www.youtube.com/watch?v=u_ft5jCDZXk&feature=youtu.be
'''
from timeit import timeit
from sortedCollection import SortedCollection
from bisect import insort, bisect_left, bisect
import sys
# Return the maximum sum subarray mod m.
@timeit
def maximumSum0(arr, m):

    prefix = 0
    maxim = 0
    S = SortedCollection()

    for i in range(n):
        # Finding prefix sum. 
        prefix = (prefix + arr[i]) % m 
        maxim = max(maxim, prefix)
        if S and prefix < S[-1]:
            it = S[-1] if prefix+1 == S[-1] else S.find_ge(prefix+1)
            maxim = max(maxim, prefix - it + m)

        # adding prefix in the set. 
        S.insert(prefix)

    return maxim 
def find_ge(a, x):
    'Find leftmost item greater than or equal to x'
    i = bisect_left(a, x)
    if i != len(a):
        return a[i]
    raise ValueError
def maximumSum1(arr, m):

    prefix = 0
    maxim = 0
    S = []

    for i in range(n):
        # Finding prefix sum. 
        prefix = (prefix + arr[i]) % m
        maxim = max(maxim, prefix)
        if S and prefix < S[-1]:
            it = S[bisect(S,prefix)]
            maxim = max(maxim, prefix - it + m)

        # adding prefix in the set. 
        insort(S, prefix)

    return maxim 
@timeit
def maximumSum(a, m):
    maxSum = 0
    S = []
    preSum = [0]*(len(a)+1)
    for i,v in enumerate(a,1): #N
        preSum[i] = (preSum[i-1] + a[i-1]%m)%m
        maxSum = max(maxSum, preSum[i])
        if S and preSum[i] < S[-1]:
            it = find_ge(S,preSum[i]+1) #logN Remove optimization at line #71
            maxSum = max(maxSum, preSum[i] - it + m) # max of sums of subArrays between 0 to i
        insort(S, preSum[i]) #N
    return maxSum

# Driver Code 
# arr = [3, 3, 9, 9, 5] 
# n = 5
# m = 7
# print(maximumSum(arr, m)) 

fptr = open("maxSubArraySum.ut")
fptro = open("maxSubArraySum.uto")
ans = list(map(int,fptro.readlines()))
q = int(fptr.readline())
for i in range(q):
    nm = fptr.readline().split()
    n = int(nm[0])
    m = int(nm[1])
    a = list(map(int, fptr.readline().rstrip().split()))
    res = maximumSum(a, m)
    print(res,ans[i],maximumSum1(a, n))
    assert(res==ans[i])

