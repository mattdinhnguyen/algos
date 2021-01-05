import random

def countSubArr(arr):
  subArrCount = 0
  val = arr[0]
  for v in arr:
    if v <= val:
      subArrCount += 1
    else:
      break
  return subArrCount




def count_subarrays0(arr):
  result = [countSubArr(arr)]

  for i in range(1, len(arr)-1):
    result.append(countSubArr(arr[i:]) + countSubArr(arr[0:i+1][::-1]) - 1)

  result.append(countSubArr(arr[::-1]))
  return result

def count_subarrays(arr):
  L = {0:1}
  R = {len(arr)-1: 1}
  result = []

  maxIdx = 0
  for i in range(1, len(arr)):
    if arr[i] > arr[maxIdx]:
      L[i] = i+1
      maxIdx = i
    else:
      for j in range(i-1,maxIdx-1,-1):
        if arr[i] < arr[j]:
          L[i] = i-j
          break
      else:
          L[i] = i - maxIdx
  maxIdx = len(arr) - 1
  for i in range(len(arr)-2,-1,-1):
    if arr[i] > arr[maxIdx]:
      R[i] = len(arr) - i 
      maxIdx = i
    else:
      for j in range(i+1,maxIdx+1):
          if arr[i] < arr[j]:
              R[i] = j-i
              break
      else:
          R[i] = j - maxIdx
  for i in range(len(arr)):
    result.append(L[i]+R[i]-1)
  print(arr)
  print(result)
  return result
                    
def bruteforce_solution(arr): #O(N^2)
    # Starts from each index,
    # expand towards both directions looking for a larger element.
    n = len(arr)
    result = [1] * n
    st = []
    for i, x in enumerate(arr):
        for di in [1, -1]:
            step = 1
            while 0 <= i+ di*step < n and  arr[i+ di*step] < x:
                result[i] += 1
                step += 1

    return result

def count_subarrays2(arr): #O(N)
    # this solution uses Stacks. Every index starts with n possibilities.
    # Using stack, going from left to right, we remove the subarrays that
    # doesn't satisify the problem condition at this line:
    # 'result[st.pop()] -= n-i'
    # Then we do it again from right to left.
    n = len(arr)
    result = [n] * n
    st = []
    for i, x in enumerate(arr):
        while st and x >= arr[st[-1]]:
            result[st.pop()] -= n-i
        st.append(i)
    st.clear()
    for i, x in reversed(list(enumerate(arr))):
        while st and x >= arr[st[-1]]:
            result[st.pop()] -= i+1
        st.append(i)
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

#   arr = [random.randint(1, 100) for _ in range(10)]
#   print(arr)
#   print(count_subarrays(arr))
#   print(bruteforce_solution(arr))
#   assert(count_subarrays(arr) == bruteforce_solution(arr))
  test_1 = [3, 4, 1, 6, 2]
  expected_1 = [1, 3, 1, 5, 1]
  output_1 = count_subarrays(test_1)
  check(expected_1, output_1)
  
  test_2 = [2, 4, 7, 1, 5, 3]
  expected_2 = [1, 2, 6, 1, 3, 1]
  output_2 = count_subarrays(test_2)
  check(expected_2, output_2)
