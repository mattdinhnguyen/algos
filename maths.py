#!/usr/local/bin/python
import numpy as np
class Solution:
    # For M = {{1, 1},{1, 0}} power(M, n-1), then we get the n-th Fibonacci number as the element at row and column (0, 0) in the resultant matrix.
    # Time Complexity: O(LogA), Extra Space: O(LogA) if we consider the function call stack size, otherwise O(1).
    def kthFibonacci(self, A: int) -> int:
        if A <3: return 1
        if A <5: return A-1
        a, b = 2, 3; by = 10**9+7
        for i in range(5,A+1):
            a, b = b, a+b
        return b%by
    def kthFibonacci(self, A: int) -> int:
        by = 10**9+7
        if A in [1,2]: return 1
        M = np.matrix([[1,1],[1,0]],dtype=int)
        def ret(N):
            if N == 1 : return M
            M1 = ret(N//2)
            return (M1*M1*M)%by if N%2 else (M1*M1)%by
        return np.array(ret(A-1))[0][0]
    def power(self, x, y): # x: int/float, y pos/neg
        if y == 0: return 1
        tmp = self.power(x, int(y / 2))
        if y % 2 == 0: return tmp * tmp
        else:
            if y > 0: return x * tmp * tmp # x* for odd
            else: return (tmp * tmp) / x

if __name__ == "__main__":
    sol = Solution()
    assert sol.kthFibonacci(1000) == 517691607
    print('%.6f' %sol.power(2, -3))
