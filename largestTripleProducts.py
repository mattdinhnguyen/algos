from numpy import product
from heapq import heappush, heappop, nlargest

# Add any helper functions you may need here


def findMaxProduct0(arr):
  
  res = [-1, -1, product(arr[:3])]
  sa = sorted(arr[:3])
  for i in range(3,len(arr)):
    if arr[i] > sa[0]:
      sa[0] = arr[i]
      sa = sorted(sa)
    res.append(product(sa))

  return res

def findMaxProduct(arr):
  
  res = [-1, -1, product(arr[:3])]
  h = arr[:3]
  for i in range(3,len(arr)):
    heappush(h, arr[i])
    res.append(product(nlargest(3, h)))

  return res


def printInteger(n):
  print('[', n, ']', sep='', end='')

def printIntegerList(array):
  size = len(array)
  print('[', end='')
  for i in range(size):
    if i != 0:
      print(', ', end='')
    print(array[i], end='')
  print(']', end='')

test_case_number = 1

def check(expected, output):
  global test_case_number
  expected_size = len(expected)
  output_size = len(output)
  result = True
  if expected_size != output_size:
    result = False
  for i in range(min(expected_size, output_size)):
    result &= (output[i] == expected[i])
  rightTick = '\u2713'
  wrongTick = '\u2717'
  if result:
    print(rightTick, 'Test #', test_case_number, sep='')
  else:
    print(wrongTick, 'Test #', test_case_number, ': Expected ', sep='', end='')
    printIntegerList(expected)
    print(' Your output: ', end='')
    printIntegerList(output)
    print()
  test_case_number += 1

if __name__ == "__main__":
  arr_1 = [1, 2, 3, 4, 5]
  expected_1 = [-1, -1, 6, 24, 60]
  output_1 = findMaxProduct(arr_1)
  check(expected_1, output_1)

  arr_2 = [2, 4, 7, 1, 5, 3]
  expected_2 = [-1, -1, 56, 56, 140, 140]
  output_2 = findMaxProduct(arr_2)
  check(expected_2, output_2)
