import numpy as np

class Solution:
    def diffBitSumPairwise(self, A):
        l = np.array(A,dtype=np.uint32).reshape(-1,1)
        l = (np.unpackbits(l.view(np.uint8),axis=1))
        ans = (np.count_nonzero(l==0,axis=0)*np.count_nonzero(l==1,axis=0))
        return int(2*ans.sum())%(10**9+7)

if __name__ == '__main__':
    sol = Solution()
    print(sol.diffBitSumPairwise([1,3,5]))
