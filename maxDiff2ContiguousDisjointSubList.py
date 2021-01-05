# https://www.careercup.com/question?id=19286747
from timeit import timeit
# @timeit
def max_contiguous_subarray_abs_diff(A):
  n = len(A)
  assert n > 0
  left_min, left_max = left_min_max(A)
  right_min, right_max = left_min_max(A[::-1])
  max_diff = 0
  for i in range(n - 1):
    smallest = min(left_min[i], right_min[n - 2 - i])
    largest = max(left_max[i], right_max[n - 2 - i])
    diff = abs(largest - smallest)
    if max_diff < diff:
      max_diff = diff
  return max_diff


def left_min_max(A):
  n = len(A)
  pre_min = [0] * (n + 1)
  pre_max = [0] * (n + 1)
  pre = [0] * (n + 1)  # :i
  for i in range(0, n):
    pre[i + 1] = pre[i] + A[i]
    pre_min[i + 1] = pre_min[i]
    if pre_min[i + 1] > pre[i + 1]:
      pre_min[i + 1] = pre[i + 1]
    pre_max[i + 1] = pre_max[i]
    if pre_max[i + 1] < pre[i + 1]:
      pre_max[i + 1] = pre[i + 1]
  left_min = [None] * (n - 1)
  left_max = [None] * (n - 1)
  for i in range(0, n - 1):
    left_min[i] = pre[i + 1] - pre_max[i]
    left_max[i] = pre[i + 1] - pre_min[i]
  # print(A, pre, pre_min, pre_max, left_min, left_max)
  return (left_min, left_max)

# find 2 disjoint contiguous subarrays of 1 array, the absolute difference between the sum of these 2 subarray is max
def maxMin_subarray(A):
  index = len(A) - 1
  curMin = curMax = A[index]
  diffMax = 0
  max_map = {index: curMax}
  min_map = {index: curMin}
  index -= 1
  while index > -1:
    if A[index] > A[index] + max_map[index+1]:
      max_map[index] = A[index]
    else:
      max_map[index] = A[index] + max_map[index+1]
    if max_map[index] > curMax:
      curMax = max_map[index]
    if A[index] < A[index] + min_map[index+1]:
      min_map[index] = A[index]
    else:
      min_map[index] = A[index] + min_map[index+1]
    if min_map[index] < curMin:
      curMin = min_map[index]
    index -= 1
  return(curMin,curMax)

def _maxDiff2ContiguousDisjointSubLists(A):
  maxDiff = 0
  for i in range(1,len(A)):
    left_min, left_max = maxMin_subarray(A[:i])
    right_min, right_max = maxMin_subarray(A[i:])
    diff = max(abs(left_max-right_min),abs(right_max-left_min))
    if diff > maxDiff:
      maxDiff = diff
  return maxDiff
# @timeit
def maxDiff2ContiguousDisjointSubLists(A):
  assert len(A) > 0
  if len(A) == 1:
    return A[0]
  return max(_maxDiff2ContiguousDisjointSubLists(A), _maxDiff2ContiguousDisjointSubLists(A[::-1]))

if __name__=="__main__":
  for A in [[2, -1, -2, 1, -4, 2, 8], [8, 1, 1, 1], [4, -1, 7], [-1, -2, -3], [-10, 7, -3, 2, 2, -20, 1]]:
    print(A, max_contiguous_subarray_abs_diff(A), maxDiff2ContiguousDisjointSubLists(A), left_min_max(A))
