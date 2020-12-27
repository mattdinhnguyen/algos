# Given a sequence of integers and an integer total target, return whether a contiguous sequence of integers sums up to target.
from collections import defaultdict
# FB screen interview
def hasSubArrSumX(a, k): #N
  prefixSum = [0]*(len(a)+1)
  preSumCnts = defaultdict(int)
  for i in range(1,len(a)+1):
    prefixSum[i] = prefixSum[i-1] + a[i-1]
    preSumCnts[prefixSum[i]] += 1
  for v in preSumCnts:
    if k + v in preSumCnts:
      return True
  return False
# https://www.geeksforgeeks.org/number-subarrays-sum-exactly-equal-k/
def NumSubArrSumX(a, k): #N
  prefixSum = 0
  preSumCnts = defaultdict(int)
  numSubArrSumX = 0
  for i in range(len(a)):
    prefixSum += a[i]
    # if prefixSum == k: handled in 24
    #   numSubArrSumX += preSumCnts[0] # Add number of subarrays previously having a sum equal to 0
    # if a[i] == k: handled in 24
    #   numSubArrSumX += 1 
    if prefixSum-k in preSumCnts:
      numSubArrSumX += preSumCnts[prefixSum-k] # Add number of subarrays previously found having sum equal to currsum-sum
    preSumCnts[prefixSum] += 1 # Count number of subarrays having sum equal to prefixSum[i]
  numSubArrSumX += preSumCnts[k]
  return numSubArrSumX
# find the number of subarrays with sum exactly equal to k.
# Does not handle 1-element subarrays with value k
def findSubarraySum(arr, n, Sum): 

	# Dictionary to store number of subarrays starting from index zero having particular value of sum. 
	prevSum = defaultdict(lambda : 0)
	res = 0 
	currsum = 0

	for i in range(0, n): 

		# Add current element to sum so far. 
		currsum += arr[i]

		# If currsum is equal to desired sum, 
		# then a new subarray is found. So 
		# increase count of subarrays. 
		if currsum == Sum: 
			res += 1

		# currsum exceeds given sum by currsum - sum.
		# Find number of subarrays having 
		# this sum and exclude those subarrays 
		# from currsum by increasing count by 
		# same amount. 
		if (currsum - Sum) in prevSum:
			res += prevSum[currsum - Sum] 
		

		# Add currsum value to count of 
		# different values of sum. 
		prevSum[currsum] += 1
	
	return res 

if __name__ == "__main__":

  for a,k in [([10, 2, -2, -20, 10] ,-10),([0, 1, 3, 4, 8, 23], 8),([1, 0, 3, 1, 4, 23], 8),([1, 3, 1, 4, 23], 7),([9, 4, 20, 3, 10, 5],33)]:
    print(hasSubArrSumX(a, k), NumSubArrSumX(a, k),findSubarraySum(a, len(a), k)) 
	

# [1, 3, 1, 4, 23], 8 : True (because 3 + 1 + 4 = 8)  1, 4, 5, 9, 8, 32
# [1, 3, 1, 4, 23], 7 : False
# [1, 3, 4, 23] 8
