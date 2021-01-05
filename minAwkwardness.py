def computeMaxAwkwardness(short, tall):
  awkw = max(tall[0]-short[0],tall[-1]-short[-1])
  for l in [short,tall]:
    for i in range(len(l) - 2):
      aw = l[i+1] - l[i]
      if aw > awkw:
        awkw = aw
  return awkw

def minOverallAwkwardness(arr):
  sArr = sorted(arr)
  mid = len(arr)//2
  awkw = []
  for m in [mid,mid+1]:
    awkw.append(computeMaxAwkwardness(sArr[:m], sArr[m:]))
  return min(*awkw)

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
  arr_1 = [5, 10, 6, 8]
  expected_1 = 4
  output_1 = minOverallAwkwardness(arr_1)
  check(expected_1, output_1)

  arr_2 = [1, 2, 5, 3, 7]
  expected_2 = 4
  output_2 = minOverallAwkwardness(arr_2)
  check(expected_2, output_2)
