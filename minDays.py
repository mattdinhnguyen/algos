from math import ceil, floor
from collections import Counter
from timeit import timeit
# https://www.hackerrank.com/challenges/minimum-time-required/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=search
@timeit
def minTime(machines, goal):
    c = Counter(machines)
    fastest = min(c)
    slowest = max(c)
    min_days = ceil(fastest*goal/len(machines))
    max_days = floor(slowest*goal/len(machines))
    while max_days-min_days>1:
        mid = (min_days+max_days)//2
        output = sum((mid//x)*y for x,y in c.items())
        if output<goal:
            min_days = mid
        else:
            max_days = mid
    return max_days

@timeit
def minTime0(machines, goal):
    c = Counter(machines)
    fastest = min(c)
    min_days = 1
    max_days = ceil((fastest*goal)/c[fastest])
    while max_days-min_days>1:
        mid = (min_days+max_days)//2
        output = sum((mid//x)*y for x,y in c.items())
        if output<goal:
            min_days = mid
        else:
            max_days = mid
    return max_days

# n,goal = map(int,input().split())
# machines = list(map(int,input().split()))
# print(minTime(machines, goal))
print(minTime0([2,3], 5))
print(minTime([2,3], 5))