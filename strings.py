from typing import List
from bisect import bisect
import string
import json
from dp import Solution as vSol
from collections import deque, OrderedDict
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        for i,c in enumerate(strs[0]):
            for j in range(1,len(strs)):
                s = strs[j]
                if i == len(s):
                    return s
                elif s[i] != c:
                    return s[:i]
        return strs[0]
    def countAndSay(self, n: int) -> str:
        def _countSay(s):
            _s = ""
            i = j = 0
            while j <= len(s):
                if j == len(s):
                    _s += str(j-i)+s[i]
                elif s[j] != s[i]:
                    _s += str(j-i)+s[i]
                    i = j
                j += 1
            return _s
        countSay = ["0", "1", "11", "21", "1211"]
        for i in range(len(countSay),n+1):
            countSay.append(_countSay(countSay[i-1]))
        return countSay[n]
    # version1 > version2 return 1, version1 < version2 return -1
    def compareVersion(self, A, B): # 0.1 < 1.1 < 1.2 < 1.13 < 1.13.4
        a = list(map(int, A.split("."))); b = list(map(int, B.split(".")))
        for i,n in enumerate(a):
            if i == len(b):
                return 1 if sum(a[i:]) else 0
            if n > b[i]: return 1
            elif n < b[i]: return -1
        return -1 if len(b) > len(a) and sum(b[len(a):]) else 0
    # Q1. Does string contain whitespace characters before the number? Yes
    # Q2. Can the string have garbage characters after the number? Yes. Ignore it.
    # Q3. If no numeric character is found before encountering garbage characters, what should I do? Return 0.
    # Q4. What if the integer overflows? Return INT_MAX 2147483647 if the number is positive, INT_MIN -2147483648 otherwise. 
    def atoi(self, S):
        s = S.strip().split()
        if not s: return 0
        s = s[0]
        sign = 1
        if s[0] in ('+','-'):
            if s[0] == '-': sign = -1
            s = s[1:]
        n = 0
        for i,d in enumerate(s):
            if '0' <= d <= '9':
                n = n*10+ ord(d) - ord('0')
            elif n: break
            else:
                return 0
        return max(-2**31, min(sign * n,2**31-1))
    # 3-loop divides the string s into 4 substring: s1, s2, s3, s4. Check if each substring is valid.
    # recursive DFS time: three loops O(3^4)
    def restoreIpAddresses(self, s: str) -> List[str]: # each integer is between 0 and 255
        lenS = len(s); res = []
        def isValid(s): # invalid cases: length >3 or ==0, length >1 and s[0]=='0', s > '255'
            if len(s) >3 or len(s) ==0 or len(s) >1 and s[0]=='0' or int(s) > 255:
                return False
            return True
        for i in range(1,4): # s[:1] to s[:3]
            if i < (lenS-2):
                for j in range(i+1,i+4): # s[1:2] to s[3:6]
                    if j < lenS-1:
                        for k in range(j+1,j+4): # s[2:3] to s[6:9]
                            if k <lenS:
                                ls = [s[:i],s[i:j],s[j:k],s[k:]]
                                if all(isValid(_) for _ in ls):
                                    res.append(".".join(ls))
        return res
    def restoreIpAddresses(self, ip: str) -> List[str]: # each integer is between 0 and 255
        def restoreIp(idx, restored, count):
            if len(ip) - idx > 3 * (4 - count): return # remaining string is longer than needed digits
            for i in range(1,4):
                if idx+i > len(ip): break
                s = ip[idx:idx+i] # 1 to 3 digits
                if s[0] == "0" and len(s) >1 or i ==3 and int(s) > 255: continue # invalid ip subnet parts
                if count <3: restoreIp(idx+i, restored+s+".", count+1)
                elif count == 3 and idx+i == len(ip): solutions.append(restored+s) # got 1 candidate
        solutions = []
        restoreIp(0, "", 0)
        return solutions
    # https://leetcode.com/problems/implement-strstr/discuss/12814/My-answer-by-Python
    def strStr(self, haystack: str, needle: str) -> int:
        if not needle: return 0
        if not haystack or len(haystack) < len(needle): return -1
        i = j = 0
        while i <len(haystack):
            if haystack[i] == needle[j]: # iterate haystack to match needle
                if j == len(needle)-1: return i-j # matched all needle chars
                else: j += 1 # see if next char matches
            elif j:     # previous chars matched but now different
                i = i-j # back to the first same character
                j = 0
            i += 1
        return -1
    # https://old.blog.phusion.nl/2010/12/06/efficient-substring-searching/
    def strStr(self, haystack: str, needle: str) -> int:
        return haystack.find(needle)
    def strStr(self, haystack: str, needle: str) -> int:
        if not needle: return 0
        if len(haystack) < len(needle):  return -1 # early termination
        i = j = 0
        while j < len(needle) and i < len(haystack): 
            if haystack[i] != needle[j]: # restart matching search at i-j+1
                i = i - j + 1
                j = 0
                continue
            i += 1; j += 1
        return i-j if j == len(needle) else -1
    # KMP algorithm: jBoxer's blog http://jakeboxer.com/blog/2009/12/13/the-knuth-morris-pratt-algorithm-in-my-own-words/ 
    # https://www.geeksforgeeks.org/kmp-algorithm-for-pattern-searching/
    def genLPS(self, pattern: str) -> List[int]: # https://dev.to/girish3/string-matching-kmp-algorithm-cie
        i = 1; lps = [0]*len(pattern); imm = 0 # lps (index of first mismatch)
        while i < len(pattern):
            if pattern[i] == pattern[imm]: imm += 1; lps[i] = imm; i += 1
            elif imm: imm = lps[imm-1] # repeat above line: rematch i with new imm 
            else: lps[i] = 0; i += 1 # no prefix == suffix for i, move to next char
        return lps
    def strStr(self, haystack: str, needle: str) -> int:
        m = len(haystack); n = len(needle)
        if not n: return 0
        i = j = 0
        lps = self.genLPS(needle)
        while i < m:
            if haystack[i] == needle[j]: i += 1; j += 1 # matching
            if j == n: return i - j # matched all chars of needle
            if i < m and haystack[i] != needle[j]: # prev matched, but now not
                if j: j = lps[j-1] # reset j to lsp
                else: i += 1 # prev not matched
        return -1
    def strStr(self, haystack: str, needle: str) -> int:
        def kmp(str_):
            b, prefix = 0, [0]
            for i in range(1, len(str_)):
                while b > 0 and str_[i] != str_[b]:
                    b = prefix[b - 1]
                if str_[b] == str_[i]:
                    b += 1
                else:
                    b = 0
                prefix.append(b)
            return prefix

        str_ = kmp(needle + '#' + haystack)
        n = len(needle)
        if n == 0:
            return n
        for i in range(n + 1, len(str_)):
            if str_[i] == n:
                return i - 2 * n
        return -1
    def reverseWords(self, A):
        rev = A.split()
        rev.reverse()
        s = ' '
        s = s.join(rev)
        return s
    def minPrefix4Palindrome(self, S: str) -> int:
        lps = self.genLPS(S + "$" + S[::-1]) # generate LPS
        return len(S) - lps[-1] # return minimum prefix to make a palindrome
    def requiredAppends(self, S: str) -> int:
        def isPalin(s, l, r):
            while l < r and s[l] == s[r]: l += 1; r -= 1
            return True if l >= r else False
        i = 0; r = len(S)-1
        for i in range(r+1):
            if isPalin(S, i, r):
                break
        return i
    def requiredAppends(self, S: str) -> int:
        lps = self.genLPS(S[::-1] + "$" + S) # generate LPS
        return len(S) - lps[-1] # return minimum prefix (suffix of S[::-1]) to make a palindrome
    # https://leetcode.com/problems/add-binary/
    def addBinary(self, a: str, b: str) -> str:
        ahi = len(a)-1; bhi = len(b)-1
        dl = abs(ahi-bhi)
        s2i = {'00':0,'01':1,'10':1,'11':2}
        i2c = {0:'0',1:'1',2:'0',3:'1'}
        if dl: # prepend shorter string with 0's
            if ahi < bhi: a = '0'*dl + a
            else: b = '0'*dl + b
        out, c = '', 0
        for i in range(len(a)-1,-1,-1):
            add = s2i[a[i]+b[i]] + c
            out = i2c[add] + out
            c = add//2
        if c: out = '1'+ out # prepend carry
        return out
    def addBinary(self, a: str, b: str) -> str:
        res, carry = '', 0; ord0 = ord('0')
        i, j = len(a) - 1, len(b) - 1
        while i >= 0 or j >= 0 or carry:
            curval = (i >= 0 and a[i] == '1') + (j >= 0 and b[j] == '1')
            carry, rem = divmod(curval + carry, 2)
            res = chr(ord0+rem) + res
            i -= 1; j -= 1
        return res
    # https://leetcode.com/problems/power-of-two/discuss/676737/Python-Oneliner-O(1)-bit-manipulation-trick-explained
    # https://leetcode.com/problems/power-of-two/discuss/63966/4-different-ways-to-solve-Iterative-Recursive-Bit-operation-Math
    def isPowerOfTwo(self, n: int) -> bool:
        return n > 0 and (1<<32) % n == 0
    def isPowerOfTwo(self, n):
        return n > 0 and not (n & n-1)
    def multiply(self, num1: str, num2: str) -> str: # Time complexity: O(n * m). Space complexity: O(n + m)
        if num1 == '0' or num2 == '0': return 0
        res = [0] * (len(num1)+len(num2)); ord0 = ord('0')
        for i in range(len(num1)-1, -1, -1):
            carry = 0
            for j in range(len(num2)-1, -1, -1):
                tmp = (ord(num1[i])-ord0)*(ord(num2[j])-ord0) + carry
                carry = (res[i+j+1]+tmp) // 10
                res[i+j+1] = (res[i+j+1]+tmp) % 10
            res[i] += carry # carry becomes the most significant digit
        res = ''.join(map(str, res))
        return res.lstrip('0')
    def multiply(self, num1: str, num2: str) -> str: # Time complexity: O(n * m). Space complexity: O(n + m)
        len1, len2 = len(num1), len(num2)
        num1 = list(map(int, reversed(num1))) # reverse the numbers and convert each digit to integer
        num2 = list(map(int, reversed(num2)))
        result = [0 for i in range(len1 + len2)] # digits of resulting number in reversed order
        for j in range(len2):
            for i in range(len1):
                result[i + j] += num1[i] * num2[j] # multiple least significant num2 digits (one at a time) to all num1 digits
                result[i + j + 1] += result[i + j] // 10  # carry
                result[i + j] %= 10  # 1-digit
        i = len(result) - 1
        while result[i] == 0 and i > 0: # get rid of leading zeros, find index of first non-zero digit
            i -= 1
        return "".join(map(str, result[:i + 1][::-1])) # convert digits to strings, reverse and concatenate
    # https://towardsdatascience.com/10-algorithms-to-solve-before-your-python-coding-interview-feb74fb9bc27
    def addNumStr(self, num1: str, num2:str):
        n1 = n2 = 0; ord0 = ord("0")
        m1, m2 = 10**(len(num1)-1), 10**(len(num2)-1)
        for i in num1:
            n1 += (ord(i) - ord0) * m1
            m1 = m1//10
        for i in num2:
            n2 += (ord(i) - ord0) * m2
            m2 = m2//10
        return str(n1 + n2)
    # https://leetcode.com/problems/first-unique-character-in-a-string/
    def firstUniqChar(self, s: str) -> int:
        orda = ord('a')
        letters: List[List[int]] = [[] for _ in range(26)]
        for i,chr in enumerate(s): letters[ord(chr)-orda].append(i)
        fuchrIdx = len(s)
        for aIndices in letters:
            if len(aIndices) == 1 and aIndices[0] < fuchrIdx:
                fuchrIdx = aIndices[0]
        return fuchrIdx
    def firstUniqChar(self, s: str) -> int:
        seen = set()
        for idx, c in enumerate(s):
            if c not in seen:
                if s.count(c) == 1: return idx
                seen.add(c)
        return -1
    def firstUniqChar(self, s: str) -> int: # best time, because s.find implemented in C
        return min([s.find(c) for c in string.ascii_lowercase if s.count(c)==1] or [-1])
    def firstNonRepeatCh(self, A):
        B = ""
        seen = set()
        q = deque()
        for c in A:
            if c not in seen:
                seen.add(c)
                q.append(c)
            elif c in q: q.remove(c)
            B += q[0] if q else '#'
        return B


    # https://leetcode.com/problems/number-of-1-bits/submissions/
    def hammingWeight(self, n: int) -> int:
        count = 0
        while n:
            count += n & 1
            n >>= 1
        return count # bin(n).count('1)
    # https://leetcode.com/problems/hamming-distance/submissions/
    def hammingDistance(self, x: int, y: int) -> int:
        xor = x^y
        count = 0
        while xor:
            count += xor&1
            xor >>= 1
        return count
    # https://leetcode.com/problems/counting-bits/discuss/270693/Intermediate-processsolution-for-the-most-voted-solution-i.e.-no-simplificationtrick-hidden
    def countBits(self, num: int) -> List[int]:
        dp = [0] # dp[0] number of 1's in binary 0
        for i in range(1,num+1):
            dp.append(dp[i >> 1] + i&1)
        return dp
    # https://leetcode.com/discuss/interview-question/124708/print-json-format-string
    def prettyJson(self, A):
        result = [""]; i = r = 0; brace = 1 # vector rows, A and row indices, brace count for indent
        while i < len(A):
            c = A[i]
            if c in '{[':
                if brace==1 and r==0: result[r] += c # first brace
                else:
                    result.append(""); r += 1 # next brace starts a new string with proper indent by itself
                    result[r] += brace*'\t' + c
                    brace += 1
                result.append(""); r += 1 # prep next string/line with proper indent
                result[r] += brace*'\t'
            elif c in '}]':
                if brace > 1:
                    result.append(""); r += 1; brace -= 1 # closing brace starts a new string with proper indent
                    result[r] += brace*'\t' + c
                else:
                    result.append(""); r += 1 # last closing brace, no indent
                    result[r] += c; brace -= 1
            elif c == ',':
                result[r] += c
                if A[i+1] not in '{[':
                    result.append(""); r += 1 # start a new string/line with proper indent
                    result[r] += brace*'\t'
            elif c == ':':
                result[r] += c
                if A[i+1] in '{[': # start a new string/line with openning brace after :
                    result.append(""); r += 1; i += 1
                    result[r] += brace*'\t' + A[i]
                    brace += 1
                    if A[i+1] not in '{[':
                        result.append(""); r += 1; i += 1 # start a new string/line for property/value after brace
                        result[r] += brace*'\t' + A[i]
            elif c != ' ':
                result[r] += A[i] # property/value
            i += 1
        return result
    def prettyJson(self, A):
        data = json.loads(A)
        ret = json.dumps(data, indent = 4).split('\n')
        return ret
    # https://leetcode.com/problems/text-justification/discuss/191149/Readable-Python-Solution-which-beats-99.65
    def fullJustify(self, words, maxWidth):
        ans = []; n = len(words); L = maxWidth; i = 0 # the index of the current word
        def getKwords(i): # start with words[i]
            cur_len = len(words[i]); k = 1
            while i + k < n:
                next_len = len(words[i+k]) + 1 # next word + space 
                if cur_len + next_len <= L: k += 1; cur_len += next_len
                else: break   
            return k # k words <= L for a line
        def insertSpace(i, k): # concatenate words[i:i+k] into one line
            l = ' '.join(words[i:i+k])       
            if k == 1 or i + k == n:        # if the line contains only one word or it is the last line  
                spaces = L - len(l)         # we just need to left assigned it
                line = l + ' ' * spaces 
            else:                           
                spaces = L - len(l) + (k-1) # total number of spaces we need insert  
                space = spaces // (k-1)     # average number of spaces we should insert between two words
                left = spaces % (k-1)       # number of 'left' words, i.e. words that have 1 more space than the other words on the right side
                if left > 0:
                    line = ( " " * (space + 1) ).join(words[i:i+left])  # left words
                    line += " " * (space + 1)                           # spaces between left words & right words
                    line += (" " * space).join(words[i+left:i+k])       # right woreds
                else: 
                    line = (" " * space).join(words[i:i+k])
            return line
        while i < n: 
            k = getKwords(i) # get k words for line
            line = insertSpace(i, k) # create a line which contains words from words[i] to words[i+k-1]
            ans.append(line) 
            i += k 
        return ans
    # https://leetcode.com/problems/text-justification/discuss/24891/Concise-python-solution-10-lines.
    # Round robin: only k words can fit on a given line, total length of those words is num_of_letters.
    # The rest are spaces, and there are (maxWidth - num_of_letters) of spaces.
    # The "or 1" part is for dealing with the edge case len(cur) == 1
    def fullJustify(self, words, maxWidth):
        res, cur, num_of_letters = [], [], 0
        for w in words:
            if num_of_letters + len(w) + len(cur) > maxWidth: # add words to cur line till exceeding maxWidth
                for i in range(maxWidth - num_of_letters): # round robin add 1 space to each word on line till ran out
                    cur[i%(len(cur)-1 or 1)] += ' '
                res.append(''.join(cur)) # got 1 line
                cur, num_of_letters = [], 0
            cur += [w] # words on current line
            num_of_letters += len(w) # for words fit on current line 
        return res + [' '.join(cur).ljust(maxWidth)] # add sapces to last line up to maxwidth
if __name__ == '__main__':
    vsol = vSol()
    assert vsol.longestPalindrome("aaaabaaa") == "aaabaaa"
    sol = Solution()
    assert sol.reverseWords("Good morning!") == "morning! Good"
    assert sol.firstNonRepeatCh("hspkzrqozquywqsnumncuclkrrwsormkfprzotxrcotbnteiizlvt") == "hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh"
    assert sol.firstNonRepeatCh("abadbc") == "aabbdd"
    assert sol.firstNonRepeatCh("abcabc") == "aaabc#"
    # assert sol.firstUniqChar("loveleetcode") == 2
    # assert sol.fullJustify(["This", "is", "an", "example", "of", "text", "justification."], 16) == ["This    is    an", "example  of text", "justification.  "]
    # assert sol.fullJustify(["What","must","be","acknowledgment","shall","be"], 16) == ["What   must   be","acknowledgment  ","shall be        "]
    # assert sol.fullJustify(["Science","is","what","we","understand","well","enough","to","explain","to","a","computer.","Art","is","everything","else","we","do"], 20) == ["Science  is  what we","understand      well","enough to explain to","a  computer.  Art is","everything  else  we","do                  "]
    print(sol.prettyJson('{"id": "0001", "type": "donut","name": "Cake","ppu": 0.55, "batters":{"batter":[{ "id": "1001", "type": "Regular" },{ "id": "1002", "type": "Chocolate" }]},"topping":[{ "id": "5001", "type": "None" },{ "id": "5002", "type": "Glazed" }]}'))
    print(sol.prettyJson('{"id":100,"firstName":"Jack","lastName":"Jones","age":12}'))
    print(sol.prettyJson('["foo", {"bar":["baz",null,1.0,2]}]'))
    # assert sol.multiply("12", "10") == "120"
    # assert sol.multiply("123", "456") == "56088"
    # assert sol.addNumStr("12", "10") == "22"
    # assert sol.addBinary('111','1') == '1000'
    # res = sol.addBinary('1011','1010') == '10101'
    # res = sol.hammingWeight(7)
    assert sol.requiredAppends("aabb") == 2
    assert sol.requiredAppends("abede") == 2
    assert sol.requiredAppends("ABC") == 2
    assert sol.requiredAppends("abcac") == 2
    assert sol.minPrefix4Palindrome("ABC") == 2
    assert sol.minPrefix4Palindrome("AACECAAAA") == 2
    assert sol.strStr("AAAAAAAAAAAAAAAAAB", "AAAAB") == 13
    assert sol.strStr("ABABABCABABABCABABABC", "ABABAC") == -1
    assert sol.strStr("mississippi","issip") == 4
    assert sol.strStr("hello", "ll") == 2
    assert sol.strStr("aaaaa", "bba") == -1
    assert sol.strStr("", "") == 0
    # assert sol.restoreIpAddresses("255255111135") == ["255.255.111.135"]
    # assert sol.restoreIpAddresses("25525511135") == ["255.255.11.135","255.255.111.35"]
    # assert sol.restoreIpAddresses("0000") == ["0.0.0.0"]
    # assert sol.restoreIpAddresses("1111") == ["1.1.1.1"]
    # assert sol.restoreIpAddresses("010010") == ["0.10.0.10","0.100.1.0"]
    # assert sol.restoreIpAddresses("101023") == ["1.0.10.23","1.0.102.3","10.1.0.23","10.10.2.3","101.0.2.3"]
    # assert sol.atoi("") == 0
    # assert sol.atoi("  ") == 0
    # assert sol.atoi("4193 with words") == 4193
    # assert sol.atoi("words and 987") == 0
    # assert sol.atoi(" 9 2704") == 9
    # assert sol.atoi(" -9 2704") == -9
    # assert sol.atoi(" -901 2704") == -901
    # assert sol.compareVersion("1","1.1") == -1
    # assert sol.compareVersion("1.01", "1.001") == 0
    # assert sol.compareVersion("1.0", "1.0.0") == 0
    # assert sol.compareVersion("0.1", "1.1") == -1
    # assert sol.compareVersion("1.0.1", "1") == 1
    # assert sol.compareVersion("7.5.2.4", "7.5.3") == -1
    # assert sol.countAndSay(5) == "111221"
    # assert sol.countAndSay(6) == "312211"
    # assert sol.longestCommonPrefix(["abcdefgh", "aefghijk", "abcefgh"]) == "a"
    # assert sol.longestCommonPrefix(["ab", "a"]) == "a"
    # assert sol.longestCommonPrefix(["abab", "ab", "abcd"]) == "ab"
    # assert sol.longestCommonPrefix(["flower","flow","flight"]) == "fl"
    # assert sol.longestCommonPrefix(["dog","racecar","car"]) == ""