from bisect import bisect, bisect_left
from collections import Counter
from timeit import timeit
# https://www.hackerrank.com/challenges/triple-sum/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=search
@timeit
def triplets(a, b, c): # AlogA + ClogC
    sa = sorted(set(a))
    sc = sorted(set(c))
    bC = Counter(b)
    triplets = 0
    for bVal in list(bC):
        triplets += bisect(sa,bVal)*bisect(sc,bVal)
    return triplets

@timeit
def triplets1(a, b, c):
    sa = sorted(set(a))
    sc = sorted(set(c))
    cnt = 0
    for _b in set(b):
        cnt += bisect(sa,_b)*bisect(sc,_b)
    return cnt

print(triplets([1, 4, 5],[2, 3, 3],[1, 2, 3]))
print(triplets1([1, 4, 5],[2, 3, 3],[1, 2, 3]))
print(triplets([1, 3, 5], [2, 3], [1, 2, 3]))
print(triplets1([1, 3, 5], [2, 3], [1, 2, 3]))
