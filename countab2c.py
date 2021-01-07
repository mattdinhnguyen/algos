# count a's, 1 b, 2 c's string of length N
from itertools import permutations, combinations
from collections import deque
from math import factorial
from heapq import heappush, heappop, merge
import random

<<<<<<< HEAD
def singleNumber(A):
    count = [2]*(len(A)//2+1)
    for val in A:
        count[val] -= 1
    for val,c in enumerate(count):
        if c:
            return val
print(singleNumber([1,1,2,2,3,4,4]))
=======
>>>>>>> c3286385401cd4d18b10ab49d997c344606626b2
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

if __name__ == '__main__':
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
    tdata = [11, 30, 2, 30, 30, 30, 6, 2, 62, 62]
    for rMax in randomMaxIdx(tdata):
        print(rMax,tdata[rMax])
