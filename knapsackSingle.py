# | Item | Weight | Value |
# |------|--------|-------|
# | 1    | 2      | 1     |
# | 2    | 10     | 20    |
# | 3    | 3      | 3     |
# | 4    | 6      | 14    |
# | 5    | 18     | 100   |

# Put a placeholder 0 weight, 0 value item to max
# these line up better with the 1D memoization table K
class Solution:
  # Recurrence
   def knapSack(self, weights, values, capacity):
        n = len(weights) # number of items
        K = [[0 for w in range(capacity + 1)] for i in range(n+1)] # Space(N*capacity) item i value + values up to i-1 at weight limit w
        for i in range(1, n+1): # Time(N*capacity)
            for w in range(1, capacity+1):
                wi, vi = weights[i-1], values[i-1]
                if wi <= w:
                    K[i][w] = max([K[i - 1][w - wi] + vi, K[i - 1][w]]) # max(value(i + those at w-wi excluding i), value(items up to i-1 at w limit))
                else:
                    K[i][w] = K[i - 1][w] # excluding wi > w
        w, items, res = capacity, [], K[n - 1][capacity]
        for i in range(n, 0, -1):
            if res <= 0: break
            if res != K[i-1][w]: # skip if res == K value (without i) at capacity w
                items.append(weights[i-1])
                res -= values[i-1]
                w -= weights[i-1]   
        return K[n - 1][capacity], items
   def knapSack(self, weights, values, capacity):
        n = len(weights) # number of items
        dp = [0]*(capacity+1) # value at capacity j Space(capacity)
        items = set() # weights contribute to dpJnWi > dp[j]
        for i in range(n): # Time(N*capacity) for each item weight, iterate from capacity to this item weight, find the max value including this item
            for j in range(capacity, weights[i]-1, -1): # -1 to include weights[0]
                dpJnWi = values[i] + dp[j-weights[i]] # value of i + value at capacity j less weights[i]
                if dpJnWi > dp[j]: # before: value at capacity j without i because "for j loop" starting at full capacity
                    dp[j] = dpJnWi # after: including values[i]
                    if sum(items) + weights[i] <= capacity:
                        items.add(weights[i])
        return dp[capacity], items
if __name__ == "__main__":
    item_weights = [0, 2, 10, 3, 6, 18]
    item_values = [0, 1, 20, 3, 14, 100]
    W = 15 # total weight capacity
    sol = Solution()
    assert sol.knapSack(item_weights,item_values, W) == (24, {3, 2, 10})
    vals, weights, cap = [60, 100, 120], [10, 20, 30], 50
    assert sol.knapSack(weights,vals, cap)[0] == 220
    vals, weights, cap = [10, 20, 30, 40],[12, 13, 15, 19],10
    assert sol.knapSack(weights,vals, cap)[0] == 0