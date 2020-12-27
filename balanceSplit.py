from sorts import quickSort, quickSort2
# https://helloacm.com/algorithm-to-split-a-number-array-into-two-balanced-parts-by-using-sorting-and-prefix-sum/
# Given an array of integers (which may include repeated integers), determine if there’s a way to split the array into two subarrays A and B such that the sum of the integers in both arrays is the same,
# and all of the integers in A are strictly smaller than all of the integers in B. Note: Strictly smaller denotes that every integer in A must be less than, and not equal to, every integer in B
def balancedSplitExists0(arr):
  a = quickSort(arr)
  mid = len(a)//2
  while mid < len(a) - 1 and sum(a[:mid]) < sum(a[mid:]):
    mid += 1
    while mid < len(a)-2 and a[mid] == a[mid+1]:
        mid += 1
  while mid > 0 and sum(a[:mid]) > sum(a[mid:]):
    mid -= 1
    while mid > 1 and a[mid] == a[mid-1]:
        mid -= 1

  return sum(a[:mid]) == sum(a[mid:]) and a[mid-1] < a[mid]

def balancedSplitExists(arr):
  quickSort2(arr, 0, len(arr)-1)
  aSum = sum(arr)
  prefix = 0
  for i in range(len(arr)-1):
    prefix += arr[i]
    if arr[i] != arr[i + 1] and aSum == 2*prefix:
      return True
  return False

# These are the tests we use to determine if the solution is correct.
# You can add your own at the bottom, but they are otherwise not editable!

def printString(string):
  print('[\"', string, '\"]', sep='', end='')

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
    printString(expected)
    print(' Your output: ', end='')
    printString(output)
    print()
  test_case_number += 1

if __name__ == "__main__":
  arr_1 = [2, 1, 2, 5]
  expected_1 = True
  output_1 = balancedSplitExists(arr_1)
  check(expected_1, output_1)

  arr_2 = [3, 6, 3, 4, 4]
  expected_2 = False
  output_2 = balancedSplitExists(arr_2)
  check(expected_2, output_2)

  # Add your own test cases here