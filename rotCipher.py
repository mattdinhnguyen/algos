import string
# Add any extra import statements you may need here


# Add any helper functions you may need here



orda = ord('a')
ordA = ord('A')
lenAlpha = len(string.ascii_lowercase)
def rotationalCipher(input, rotation_factor):
  l = list()
  for a in input:
    if a.islower():
      idx = (ord(a) - orda + rotation_factor) % lenAlpha
      l.append(string.ascii_lowercase[idx])
    elif a.isupper():
      idx = (ord(a) - ordA + rotation_factor) % lenAlpha
      l.append(string.ascii_uppercase[idx])
    elif a.isdigit():
      idx = (int(a) + rotation_factor) % 10
      l.append(string.digits[idx])
    else:
      l.append(a)

  return "".join(l)


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
  input_1 = "All-convoYs-9-be:Alert1."
  rotation_factor_1 = 4
  expected_1 = "Epp-gsrzsCw-3-fi:Epivx5."
  output_1 = rotationalCipher(input_1, rotation_factor_1)
  check(expected_1, output_1)

  input_2 = "abcdZXYzxy-999.@"
  rotation_factor_2 = 200
  expected_2 = "stuvRPQrpq-999.@"
  output_2 = rotationalCipher(input_2, rotation_factor_2)
  check(expected_2, output_2)
