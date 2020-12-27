from typing import List
from heapq import heappush, heappop, heapify
# | Item | Weight | Value |
# |------|--------|-------|
# | 1    | 2      | 1     |
# | 2    | 10     | 20    |
# | 3    | 3      | 3     |
# | 4    | 6      | 14    |
# | 5    | 18     | 100   |

class Solution:
    def knapSack(self, items_weights, items_values, capacity):
        n = len(item_weights)
        K = [0] * (capacity + 1)

        for w in range(1, capacity + 1):
            max_sub_result = 0
            for i in range(1, n):
                wi = item_weights[i]
                vi = item_values[i]
                if wi <= w:
                    subproblem_value = K[w - wi] + vi
                    if subproblem_value > max_sub_result:
                        max_sub_result = subproblem_value 
            K[w] = max_sub_result
        return K

if __name__ == "__main__":
    sol = Solution()
    item_weights = [0, 2, 10, 3, 6, 18]
    item_values = [0, 1, 20, 3, 14, 100]
    W = 15 # total weight capacity

    print(sol.knapSack(item_weights, item_values, W))
