from heapq import heappop, heappush
# https://www.teamblind.com/post/Who-can-leetcode-hard-q1vFGBam
# you are given an integer N, and a permutation, P of the integers from 1 to N, denoted as (a_1, a_2, ..., a_N).
# You want to rearrange the elements of the permutation into increasing order, repeatedly making the following operation:
# Select a sub-portion of the permutation, (a_i, ..., a_j), and reverse its order.
# Your goal is to compute the minimum number of such operations required to return the permutation to increasing order.
def is_sorted(arr): # N
    for i in range(1, len(arr)):
        if arr[i] < arr[i-1]:
            return False
    return True

def utility(arr): # N
    cnt = 0
    for i in range(1, len(arr)):
        if arr[i] == arr[i-1]+1 or arr[i] == arr[i-1]-1:
            cnt += 1
    return cnt
        
def minOperations(arr):
    u = utility(arr)
    heap = [(-u, 0, arr[:])]
    min_cost = float("Inf")
    cache = {}
    
    while len(heap) > 0:
        v, cost, a = heappop(heap) # highest utility value
        
        if is_sorted(a):
        # if a == sorted(a):
            min_cost = min(min_cost, cost)
            if cost <= 1:
                break
            
        elif cost < min_cost:
            for i in range(len(a)-1):
                for j in range(i+1, len(a)):
                    b = a[:]
                    b[i:j+1] = b[i:j+1][::-1]
                    
                    if tuple(b) not in cache or cache[tuple(b)] > cost+1:
                        u = utility(b)
                        heappush(heap, (-u, cost+1, b))
                        cache[tuple(b)] = cost+1
    return min_cost

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
  n_1 = 5
  arr_1 = [1, 2, 5, 4, 3]
  expected_1 = 1
  output_1 = minOperations(arr_1)
  check(expected_1, output_1)

  n_2 = 3
  arr_2 = [3, 1, 2]
  expected_2 = 2
  output_2 = minOperations(arr_2)
  check(expected_2, output_2)
  