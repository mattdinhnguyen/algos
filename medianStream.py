import math
import bisect

def median(slst):
  lenSl = len(slst)
  if lenSl % 2:
    result = slst[lenSl//2]
  else:
    e = lenSl//2
    o = e - 1
    result = (slst[o] + slst[e])//2

  return result

def myinsert(slst, v):
  if v >= slst[len(slst) - 1]:
    slst.append(v)
    ret = slst
  elif v <= slst[0]:
    ret = [v] + slst
  else:
    mid = len(slst)//2
    if v <= slst[mid]:
        ret = myinsert(slst[:mid], v) + slst[mid:]
    else:
        ret = slst[:mid+1] + myinsert(slst[mid+1:], v)
  return ret

def insert(slst, v):
  bisect.insort(slst, v)
  return slst
# Add any helper functions you may need here


def findMedian(arr):
  print(arr)
  result = [arr[0]]
  sortedList = [arr[0]]

  for i in range(1,len(arr)):
    sortedList = insert(sortedList,arr[i])
    result.append(median(sortedList))
  return result

	









# These are the tests we use to determine if the solution is correct.
# You can add your own at the bottom, but they are otherwise not editable!

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
  arr_1 = [5, 15, 1, 3]
  expected_1 = [5, 10, 5, 4]
  output_1 = findMedian(arr_1)
  check(expected_1, output_1)

  arr_2 = [2, 4, 7, 1, 5, 3]
  expected_2 = [2, 3, 4, 3, 4, 3]
  output_2 = findMedian(arr_2)
  check(expected_2, output_2)
