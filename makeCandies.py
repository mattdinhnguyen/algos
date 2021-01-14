from math import ceil, sqrt
from sys import maxsize
# All prime factors of n 
def primeFactors(n): 
    ret = []
    while n % 2 == 0: # number of two's that divide n 
        ret.append(2)
        n = n // 2
    for i in range(3,int(sqrt(n))+1,2): # n must be odd at this point, kip of 2 ( i = i + 2)
        while n % i== 0:
            ret.append(i)
            n = n // i
    if n > 2: ret.append(n) # if n is a prime, n > 2 
    return ret

def computePasses(m, w, p, n, m1, w1):
    candies = passes = 0
    while candies < n:
        passes += 1
        candies += m*w
        while candies >= p and (w < w1 or m < m1):
            candies -= p
            if w < m:
                w += 1
            else:
                m += 1

    return passes

# https://allhackerranksolutions.blogspot.com/2019/02/making-candies-hacker-rank-solution.html
def minimumPasses_1(m, w, p, n):
    sumPrimes = sum(primeFactors(n))
    w1 = sumPrimes//2
    m1 = sumPrimes - w1
    candies = 0
    investPasses = ceil(n/(m*w))
    mwTups = []
    for mm  in range(m,m1+1):
        for ww in range(w,w1+1):
            mwTups.append((mm,ww))
    mwTupSorted = sorted(mwTups, key = lambda x: abs(x[0]-x[1]))
    for mm,ww in mwTupSorted:
        passes = computePasses(m, w, p, n, mm, ww)
        print(passes)
        investPasses = min(investPasses,passes)
    return investPasses

def minimumPasses0(m, w, p, n):
    factors = primeFactors(n)
    preSum = [0]*(len(factors)+1)
    for i,v in enumerate(factors,1):
        preSum[i] = preSum[i-1] + factors[i-1]
    w1 = preSum[-1]//2
    m1 = preSum[-1] - w1
    spendPasses = ceil(n/(m*w))
    investPasses = ceil((n+p*(w1-w+m1-m))/m1*w1)
    minPasses = min(spendPasses,investPasses)
    
    return min(computePasses(m, w, p, n, m1, w1), computePasses(m, w, p, n, m1-1, w1), computePasses(m, w, p, n, m1-1, w1-1))

def minimumPasses(m, w, p, n):
    candy = 0
    invest = 0
    spend = ceil(n/(m*w)) # no investing m*w number of candies produced in 1 pass
    while candy < n:
        # investing
        passes = (p - candy) // (m * w) # passes need to get 1 machine or worker, when it is > m*w
        if passes <= 0:
            mw = (candy // p) + m + w # get as many as could if existing candies. new + existing production unit
            half = ceil(mw / 2)
            if m > w:
                m = max(m, half)
                w = mw - m
            else:
                w = max(w, half)
                m = mw - w
            candy %= p
            passes = 1
        # production
        candy += passes * m * w
        invest += passes # accumulate # of passes for investment
        spend = min(spend, invest + ceil((n - candy) / (m * w))) # candies yet to make divide to compute # of passes to reach goal n
    return min(invest, spend)

m, w, p, n = [3, 1, 2, 12]
# m, w, p, n = [1, 1, 6, 45]
# m, w, p, n = map(int,input().split())

print(minimumPasses(m, w, p, n))
for i in range(2,21): print(i,i//2,i%2)