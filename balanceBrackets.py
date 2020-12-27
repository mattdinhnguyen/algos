from heapq import heappush, heappop
from typing import List

def findEncryptedWord(s):
  if len(s) < 3:
    return s
  mid = len(s)//2
  if len(s) % 2 == 0:
    mid -= 1
  r = s[mid] + findEncryptedWord(s[:mid]) + findEncryptedWord(s[mid+1:])
  return r
	
oBrackets = {'{': '}', '(': ')', '[': ']'}
cBrackets = {'}': '{', ')': '(', ']': '['}

# space/time N/N

def isBalanced0(s):
  stack = []
  for c in s:
    if c in oBrackets:
      heappush(stack, c)
    elif c in cBrackets:
      if len(stack) == 0:
        return False
      ob = heappop(stack)
      if ob != cBrackets[c]:
        return False
  return True if len(stack) == 0 else False

def isBalanced(s):
  stack = []
  for c in s:
    if c in oBrackets:
      stack.append(c)
    elif c in cBrackets:
      if len(stack) == 0:
        return False
      ob = stack.pop()
      if ob != cBrackets[c]:
        return False
  return True if len(stack) == 0 else False

class Solution0(object):
    def __init__(self):
      pass

    def removeInvalidParentheses(self, s):
        result = []
        self.remove(s, result, 0, 0, ('(', ')'))
        return result
    
    def remove(self, s, result, last_i, last_j, par):
        count = 0
        for i in range(last_i, len(s)):
            count += (s[i] == par[0]) - (s[i] == par[1])
            if count >= 0:
                continue
            for j in range(last_j, i + 1):
                if s[j] == par[1] and (j == last_j or s[j - 1] != par[1]):
                    self.remove(s[:j] + s[j + 1:], result, i, j, par)
            return
        reversed_s = s[::-1]
        if par[0] == '(':
            self.remove(reversed_s, result, 0, 0, (')', '('))
        else:
            result.append(reversed_s)

class Solution:
    def __init__(self):
      pass

    def removeInvalidParentheses(self, s: str) -> List[str]:
      ans: List[str] = []
      self.remove(s, ans, 0, 0, ['(', ')'])
      return ans

    def remove(self, s: str, ans: List[str], last_i: int, last_j: int,  par: List[str]) -> None:
        count = 0
        for i in range(last_i, len(s)):
            chr = s[i]
            if chr == par[0]:
                count += 1
            elif chr == par[1]:
                count -= 1
            if count >= 0:
                continue
            for j in range(last_j, i+1):
                if s[j] == par[1] and (j == last_j or s[j-1] != par[1]):
                    self.remove(s[:j] + s[j+1:], ans, i, j, par)
            return
        # No invalid closed parenthesis detected. Now check opposite direction to detect extra '(', or reverse back to original direction.
        reversed_s = s[::-1]
        if par[0] == '(': # Now check opposite direction to detect extra '('
            self.remove(reversed_s, ans, 0, 0, [')','('])
        else: # reverse back to original direction
            ans.append(reversed_s)

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
  sol = Solution()
  s1 = "()())()"
  expected_1 = ["()()()", "(())()"]
  output_1 = sol.removeInvalidParentheses(s1)
  check(expected_1, output_1)

  # s1 = "abc"
  # expected_1 = "bac"
  # output_1 = findEncryptedWord(s1)
  # check(expected_1, output_1)

  # s2 = "abcd"
  # expected_2 = "bacd"
  # output_2 = findEncryptedWord(s2)
  # check(expected_2, output_2)

  # s1 = "{[(])}"
  # expected_1 = False
  # output_1 = isBalanced(s1)
  # check(expected_1, output_1)

  # s2 = "{{[[(())]]}}"
  # expected_2 = True
  # output_2 = isBalanced(s2)
  # check(expected_2, output_2)
