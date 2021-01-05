'''1) Store the count of each character (a to z) in S.

2) Update the count of characters required for A by dividing by 2.

3) Select each character for A by parsing S from right to left.

4) One "can" afford to "not select" a character, if its required-count for A will be fulfilled even after leaving it.

5) Considering point 4, always select smallest character if possible.

6) If a character can't be left (point 4 invalid), select the smallest character seen so far (even if it is not smallest for the entire string).

7) Ignore characters not required for A anymore.
Built frequency hash-map for char and occurences. Original word has half of the occurence of every char
Reverse the original string and try to construct the smallest lexicographic one
While iterating, use a stack (the idea somehow resembles the largest rectangle problem, from stack category).
If the new char occurs already n/2 times, where n = frequency of the original / 2, then skip this char, since we can not, nor we should push it to the stack
else pop chars from the stack, if: stack is not empty, top char stack is greater than the new char and by removing the top char we still can build the word with desired frequency'''
from collections import Counter

def merge(a,b):
    i = j = 0
    m = ""
    while i < len(a) or j < len(b):
        if i == len(a):
            m += b[j:]
            j = len(b)
        elif j == len(b):
            m += a[i:]
            i = len(a)
        elif a[i] <= b[j]:
            m += a[i]
            i += 1
        else:
            m += b[j]
            j += 1
    return m

# Complete the reverseShuffleMerge function below.
def reverseShuffleMerge(s):
    remain_chars = Counter(s)
    needed_chars = Counter(dict((k,v//2) for k,v in remain_chars.items()))
    used_chars = Counter()
    res = []
    
    def can_use(char):
        return (needed_chars[char] - used_chars[char]) > 0
    
    def can_pop(char):
        return used_chars[char] + remain_chars[char] - 1 >= needed_chars[char]
    
    for i in range(len(s)-1,-1,-1):
        char = s[i]
        if can_use(char):
            while res and res[-1] > char and can_pop(res[-1]):
                removed_char = res.pop()
                used_chars[removed_char] -= 1
            
            used_chars[char] += 1
            res.append(char)
        
        remain_chars[char] -= 1
    
    return "".join(res)

print(reverseShuffleMerge("abcdefgabcdefg"))