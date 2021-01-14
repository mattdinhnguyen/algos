# count a's, 1 b, 2 c's string of length N
from itertools import permutations, combinations
from collections import deque
from functools import reduce
from math import factorial
from heapq import heappush, heappop, merge
import random
from math import log
# https://leetcode.com/problems/single-number/submissions/
def singleNumber(A):
    val = 0
    for n in A:
        val ^= n
    return val
def singleNumber(A):
    return reduce(lambda a,b: a^b, A, 0)
# print(singleNumber([1,1,2,2,3,4,4]))
def count_abc0(n, nb, nc):
    if n == 1: # 1,0,0    1,1,0   1,1,1      1,1,2
        return 1 + (1 if nb else 0) + (1 if nc else 0)
    
    ret = count_abc(n-1, nb, nc)   # 2, x, x
    if nb > 0:
        c2 = count_abc(n-1, nb-1, nc)
        ret += c2
    if nc > 0:
        c3 = count_abc(n-1, nb, nc-1)
        ret += c3
    return ret

def count_abc1(n, nb, nc):
    return (1 + (n * 2) + 
                (n * ((n * n) - 1) // 2))

def count_abc(n, nb, nc):
    dp = [[[1] * (n+1) for _ in [0,1]] for _ in [0,1,2]]
    for l in range(1,n+1):
        dp[2][1][l] = dp[2][1][l-1] + dp[2][0][l-1] + dp[1][1][l-1]
        dp[1][1][l] = dp[1][1][l-1] + dp[1][0][l-1] + dp[0][1][l-1]
        dp[0][1][l] = dp[0][1][l-1] + dp[0][0][l-1]
        dp[2][0][l] = dp[2][0][l-1] + dp[1][0][l-1]
        dp[1][0][l] = dp[1][0][l-1] + dp[0][0][l-1]
    
    return dp[2][1][n]

def findPermsGaps(n):
    a = [0]*2*n
    i = j = 0
    dque = deque(range(1,n+1))
    while dque:
        while a[j]:
            j += 1
            continue
        a[j] = dque.popleft() if i%2 else dque.pop()
        k = j + a[j] + 1
        while a[k]:
            k += 1
        else:
            a[k] = a[j]
        i += 1

    return a

def call_counter(func):
    def helper(*args):
        helper.calls += 1
        return func(*args)
    helper.calls = 0
    return helper

@call_counter
def fill_i(l, n, i):
    #print(l)
    for ind in range(0, 2*n-(i+1)):
        if l[ind] == 0 and l[ind+i+1] == 0:
            l[ind] = i
            l[ind+i+1] = i
            if i == 1:
                print('found:', l)
            else:
                fill_i(l, n, i-1)
            l[ind] = 0
            l[ind+i+1] = 0


def permu(n):
    l = [0] * 2 * n
    fill_i(l, n, n)


def back_track(tmp, n):
    if not n:
        # print(tmp)
        return 0
    N = len(tmp)
    for i in range(N - n):
        j = i + n + 1 
        if tmp[i] != 0 or tmp[j] != 0:
            continue
    tmp_next = list(tmp) # deep copy 
    tmp_next[i] = tmp_next[j] = n 
    back_track( tmp_next, n-1 )
    return 0

def do_fb(n):
    buf = merge(*([i]*2 for i in range(1,n+1)))
    back_track(buf, n)

def solution(n: int) -> list:
    arr = [y for x in range(1, n + 1) for y in (x, ) * 2]
    for perm in permutations(arr):
        for i in range(1, n + 1):
            if not (n*2 - perm[::-1].index(i) - 1) - perm.index(i) - 1 == i:
               break
        else:
            return list(perm)
    return []

# find all combinations where every element appears twice and distance 
# between appearances is equal to the value 
def allCombinationsRec(arr, elem, n): 

	# if all elements are filled, print the solution 
	if (elem > n):	
		for i in (arr): 
			print(i, end = " ")	
		print("")
		return
	# Try all possible combinations for element elem 
	for i in range(0, 2 * n):
		# if position i and (i+elem+1) are 
		# not occupied in the vector 
		if (arr[i] == -1 and (i + elem + 1) < 2*n and arr[i + elem + 1] == -1):
			# place elem at position i and (i+elem+1) 
			arr[i] = arr[i + elem + 1] = elem 

			# recurse for next element 
			allCombinationsRec(arr, elem + 1, n) 

			# backtrack (remove elem from 
			# position i and (i+elem+1) ) 
			arr[i] = -1
			arr[i + elem + 1] = -1
		
def allCombinations(n): 
	arr = [-1] * (2 * n) 
	allCombinationsRec(arr, 1, n) # start with elem 1
# This code is contributed by Smitha Dinesh Semwal.

def randomMaxIdx(a):
    maxIndices = []
    for i,v in enumerate(a):
        if maxIndices:
            if v > a[maxIndices[-1]]:
                maxIndices = [i]
            elif v == a[maxIndices[-1]]:
                maxIndices.append(i)
        else:
            maxIndices.append(i)
        yield random.choice(maxIndices)
# https://leetcode.com/problems/single-number-ii/discuss/43295/Detailed-explanation-and-generalization-of-the-bitwise-operation-method-for-single-numbers
class Solution(object):
    def singleNumber(self, nums):
        x1 = x2 = mask = 0
        for i in nums:
            x2 ^= x1 & i
            x1 ^= i
            mask = ~(x1 & x2)
            x2 &= mask
            x1 &= mask
        return x1
    # https://leetcode.com/problems/single-number-ii/discuss/43294/Challenge-me-thx
    # First time number appear -> save it in "ones"
    # Second time -> clear "ones" but save it in "twos" for later check
    # Third time -> try to save in "ones" but value saved in "twos" clear i
    def singleNumber(self, A):
        ones = twos = 0
        for n in A:
            ones = (ones ^ n) & ~twos
            twos = (twos ^ n) & ~ones
        return ones
    def numSetBits(self, n):
        c = 0
        while n: # Brian Kernighan’s Algorithm: iterates only over set bits
            n &= n-1
            c += 1
        return c
    ''' If we observe bits from rightmost side at distance i than bits get inverted after 2i position in vertical sequence.
For example A = 5;
0 = 0000
1 = 0001
2 = 0010
3 = 0011
4 = 0100
5 = 0101
Iterate till the number of bits in the number. And we don’t have to iterate every single number in the range from 1 to A.
We will keep one variable which is a power of 2, in order to keep track of bit we are computing.
We will iterate till the power of 2 becomes greater than A.
We can get the number of pairs of 0s and 1s in the current bit for all the numbers by dividing A by current power of 2.
Now we have to add the bits in the set bits count. We can do this by dividing the number of pairs of 0s and 1s by 2 which will give us the number of pairs of 1s only and after that, we will multiply that with the current power of 2 to get the count of ones in the groups.
Now there may be a chance that we get a number as number of pairs, which is somewhere in the middle of the group i.e. the number of 1s are less than the current power of 2 in that particular group. So, we will find modulus and add that to the count of set bits.
Time Complexity : O(logA) '''
    def totalSetBits0(self, A):
        ret = 0; power = 2; # power tracks col being counted
        ret = A//2 + A%2 # count bits in 2**0 col
        for _ in range(1,round(log(A,2))+2): # interate thru the bits starting at 2**1
            ret += (A//(power*2)) * power
            remdr = A%(power*2)
            if remdr//power: ret += (remdr%power) + 1
            power *= 2
            ret = int(ret%(1e9+7))
        return int(ret%(1e9+7))
    def totalSetBits(self, A):
        A += 1 # include A value
        mod = 10**9+7
        ans = A//2 # the number of set bits encountered till now
        curr = 2 # start at 2
        while A >= curr:
            tot_pair = A//curr
            ans = (ans + ((tot_pair//2)*curr)%mod)%mod
            if tot_pair & 1:
                ans= (ans + A %curr)%mod
            curr = curr*2
        return ans%mod
    def totalSetBits(self, A):
        count = 0
        A = A+1 # as we would require 0 to A to get the results instead of 1 to A
        for i in range(32):
            # for each bit, calculating the total number of set bits for numbers uptil A;
            # that is number of sets of 2^(i+1), each set consisting 2^(i) set bits and
            # then adding the remainder of bits by getting the modulus with 2^(i+1)
            # and subtracting the first 2^(i) 0's from the remainder if this result is negative just take zero
            count = (count%1000000007+(A//(2**(i+1)))*(2**i)%1000000007+max(A%(2**(i+1))-(2**i),0)%1000000007)%1000000007
        return count
if __name__ == '__main__':
    sol = Solution()
    print(sum([sol.numSetBits(i) for i in range(1,6)])%1000000007)
    print(sol.totalSetBits0(5))
    # print(sol.singleNumber([2,2,3,2]))
    # print(sol.singleNumber([0,1,0,1,0,1,99]))
    # print(count_abc(1, 1, 2),factorial(4)/(factorial(2)* factorial(1)*factorial(1)))
    # print(count_abc(2, 1, 2))
    # print(count_abc(3, 1, 2))
    # print(count_abc(4, 1, 2))
    # print(count_abc(5, 1, 2))
    # print(count_abc(6, 1, 2))
    # print(count_abc(7, 1, 2))
    # print(count_abc(100, 1, 2))
    # for n in range(4,7):
    #     permsGap4 = findPermsGaps(n) # 4 1 3 1 2 4 3 2   3 1 2 1 3 2   5 1 4 1 3 2 5 4 3 2
    #     print(permsGap4)
    #     print(allCombinations(n))
        # print(solution(4))
    # permsGap4.reverse()
    # print(permsGap4)
    # print( "== back tracking == ")
    # do_fb(7)
    # permu(7)
    # print('total attempts:', fill_i.calls) # [4, 1, 3, 1, 2, 4, 3, 2]
    # tdata = [11, 30, 2, 30, 30, 30, 6, 2, 62, 62]
    # for rMax in randomMaxIdx(tdata):
    #     print(rMax,tdata[rMax])
