from collections import defaultdict

def indexMap(s):
  d = defaultdict(list)
  for i,c in enumerate(s):
    d[c].append(i)
  return d

def min_length_substring(s, t):
  minIdx = len(s) - 1
  maxIdx = 0
  sIdxMap = indexMap(s)
  tIdxMap = indexMap(t)
  for c,v in tIdxMap.items():
    if len(v) > len(sIdxMap[c]):
      return -1
    sl = sorted(sIdxMap[c])[:len(v)]
    if sl[0] < minIdx:
      minIdx = sl[0]
    if sl[-1] > maxIdx:
      maxIdx = sl[-1]

  return maxIdx - minIdx + 1

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
  s1 = "dcbefebce"
  t1 = "fd"
  expected_1 = 5
  output_1 = min_length_substring(s1, t1)
  check(expected_1, output_1)

  s2 = "bfbeadbcbcbfeaaeefcddcccbbbfaaafdbebedddf"
  t2 = "cbccfafebccdccebdd"
  expected_2 = -1
  output_2 = min_length_substring(s2, t2)
  check(expected_2, output_2)