from typing import List
from orderedset import OrderedSet
import sys
# https://www.careercup.com/question?id=5638939143045120
# Given coins in l (sorted), print combination sums of coins in increasing order up to k = 1000
if __name__ == '__main__':
    def sumCoins(l: List[int], k: int) -> List[int]: # from smallest sum to target
        l.sort()
        s = OrderedSet([0]) # start with no coin
        inc = sys.maxsize # min increment based on the coin values
        for i in range(1,len(l)):
            inc = min(inc,l[i]-l[i-1])
        for i in range(l[0],k+1,inc): # potential sum i's in min increment
            for c in l:
                if i-c in s: # potential (sum i reduced by each coin value) were already in s, becomes 1 comb sum
                    s.add(i)
                    break
        s.remove(0)
        return list(s)
    tdata = [[10,15,55],1000]
    tdata = [[1,2,3],4]
    sc = sumCoins(*tdata)
    print(sc)
