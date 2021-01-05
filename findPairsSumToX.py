import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from typing import List
# https://www.hackerrank.com/challenges/icecream-parlor/problem
# Given a list of prices for the flavors of ice cream, select the two that will cost all of the money they have.
def icecreamParlor0(x, lst):
    slst = sorted([(v,i+1) for i, v in enumerate(lst)])
    head = 0
    tail = len(slst) - 1
    while head < tail and slst[head][0] < x:
        while slst[tail][0] + slst[head][0] > x:
            tail -= 1
        while slst[tail][0] + slst[head][0] < x:
            head += 1
        if slst[tail][0] + slst[head][0] == x:
            i1, i2 = slst[tail][1], slst[head][1]
            result = [i1,i2] if i1<i2 else [i2,i1]
            break
    return result
# https://www.hackerrank.com/challenges/ctci-ice-cream-parlor/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=search
def icecreamParlor(k, arr):
    valIdx = dict()
    for i,val in enumerate(arr):
        valIdx.setdefault(val,[]).append(i+1)
    for val,lst in valIdx.items():
        if k-val in valIdx:
            if k == 2*val and len(valIdx[k-val]) < 2:
                continue
            res = lst if k == 2*val else [lst[0],valIdx[k-val][0]]
            return res

def findLessThan_k(arr, k, head, tail):
  mid = (head+tail)//2
  if head == mid:
      return mid
  if arr[mid] > k:
    return findLessThan_k(arr, k, head, mid)
  else:
    return findLessThan_k(arr, k, mid, tail)

def numberOfWays0(arr , k: int) -> int:
  pairCount: int = 0
  al = len(arr)
  s = sorted(arr)
  head = 0
  if k <= s[head]:
      return pairCount
  tail = findLessThan_k(s, k, 0, al-1)
  while head < tail:
    sum = s[head] + s[tail]
    if sum < k:
        head += 1
    elif sum > k:
        tail -= 1
    elif s[head] == s[tail]:
      pairCount += (tail-head)*(tail-head+1)/2
      break
    elif s[head] == s[head+1]:
      pairCount += 1
      head += 1
    elif s[tail] == s[tail-1]:
      pairCount += 1
      tail -= 1
    else:
      pairCount += 1
      head += 1

  return pairCount
# https://www.careercup.com/question?id=5081274989936640
def pairsSumK(arr, k):
  ans = 0
  valCounts = defaultdict(int)
  for i,v in enumerate(arr):
    if k > v and k-v in valCounts:
      ans += valCounts[k-v]
    else:
      valCounts[v] += 1
  return ans
# https://leetcode.com/problems/two-sum/submissions/
def twoSum(nums: List[int], target: int) -> List[int]:
  valIdxMap = dict()
  for i,val in enumerate(nums):
    if target-val in valIdxMap:
      return [valIdxMap[target-val],i]
    valIdxMap[val] = i
  return []


def pairsSumK0(arr , k):
  ways = 0
  aC = Counter(arr)
  visited = set()
  for v in aC:
    if v < k and v not in visited and k-v in aC:
      ways += aC[v]*(aC[v]-1)/2 if k == 2*v else aC[v] * aC[k-v]
      visited.add(k-v)
  return ways

# https://www.hackerrank.com/challenges/pairs/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=search
def pairsDiffK(arr , k): # arr[i] is unique
  sA = set(arr)
  return sum(1 for v in sA if k+v in sA)


# These are the tests we use to determine if the solution is correct.
# You can add your own at the bottom, but they are otherwise not editable!

def printInteger(n):
  print('[', n, ']', sep='', end='')

test_case_number = 1

def check(expected, output):
  global test_case_number
  result = False
  if expected == output:
    result = True
  rightTick = '\u2713'
  wrongTick = '\u2717'
  if result:
    print(rightTick, 'Test #', test_case_number, sep='')
  else:
    print(wrongTick, 'Test #', test_case_number, ': Expected ', sep='', end='')
    printInteger(expected)
    print(' Your output: ', end='')
    printInteger(output)
    print()
  test_case_number += 1


if __name__ == "__main__":
  # for k,arr in [(8,[4, 3, 2, 5, 7]),(4, [1, 4, 5, 3, 2]),(4, [2, 2, 4, 3])]:
  #   out = [icecreamParlor(k,arr),icecreamParlor0(k,arr)]
  #   print(*out)
  #   check(*out)
    # out = [pairsDiffK(arr,k),pairsSumK(arr,k)]
    # print(*out)
    # check(*out)
  # print(pairsDiffK([1, 5, 3, 4, 2],2))
  # print(pairsDiffK([1, 2, 3, 4, 3, 8, 10],6))
  target = 9
  arr = [2,7,11,15]
  expected = [0,1]
  output = twoSum(arr,target)
  check(expected,output)
  target = 0
  arr = [-3,4,3,90]
  expected = [0,2]
  output = twoSum(arr,target)
  check(expected,output)
  target = 6
  arr = [3,2,4]
  expected = [1,2]
  output = twoSum(arr,target)
  check(expected,output)
  k_1 = 6
  arr_1 = [1, 2, 3, 4, 3, 8, 10]
  expected_1 = 2
  output_1 = pairsSumK(arr_1, k_1)
  check(expected_1, output_1)

  # k_2 = 6
  # arr_2 = [1, 5, 3, 3, 3, 6, 8, 10]
  # expected_2 = 4
  # output_2 = pairsSumK(arr_2, k_2)
  # check(expected_2, output_2)

  # k_3 = 1000000000
  # arr_3 = [1, 5, 3, 3, 3,250000000,250000000,750000000,750000000]
  # expected_3 = 4
  # output_3 = pairsSumK(arr_3, k_3)
  # check(expected_3, output_3)

  # k_4 = 1000
  # arr_4 = [500,500,500,500,500]
  # expected_4 = 10
  # output_4 = pairsSumK(arr_4, k_4)
  # check(expected_4, output_4)

  