# Dont need dp[i-2]
# https://www.hackerrank.com/challenges/max-array-sum/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=dynamic-programming
def maxSubsetSum(arr):
    dp = {} # key : max index of subarray, value = sum
    dp[0], dp[1] = arr[0], max(arr[0], arr[1])
    for i, num in enumerate(arr[2:], start=2):
        dp[i] = max(dp[i-1], dp[i-2]+num, dp[i-2], num)
    return dp[len(arr)-1]

>>> [ db_value(x) for x in range(0,10) ]
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
>>> [ db_value(x) for x in range(10,20) ]
[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
>>> [ db_value(x) for x in range(20,30) ]
[4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
>>> [ db_value(x) for x in range(90,100) ]
[18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
>>> [ db_value(x) for x in range(100,110) ]
[4, 5, 6, 7, 8, 9, 10, 11, 12, 13]