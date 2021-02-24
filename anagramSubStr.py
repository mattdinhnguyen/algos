import numpy
from collections import Counter, defaultdict
from timeit import timeit
# does b has any anagram of a
# 1- Sort characters of a
# 2- For each contiguous-character substring S of a.Length in b
# 2.1- Sort the characters of S
# 2.2- if S == a, return true
# 3- return false

def hasAnagram0(a, b):
    if len(b) < len(a):
        return False
    if len(a) == 0 or len(b) == 0:
        return True
    sa = "".join(sorted(a))
    for i in range(len(b)-len(a)+1):
        sb = "".join(numpy.unique(numpy.array(sorted(b[i:i+len(a)]))))
        if sa == sb:
            return True
    return False

def hasAnagram(a, b):
    aC = Counter(a)
    for i in range(len(b)-len(a)+1):
        sb = b[i:i+len(a)]
        if Counter(sb) == aC:
            return True
    return False

testVector = [
("xyz", "afdgzyxksldfm"),
("abcdefg", "abcfedgsfdaf"),
("a", "cdsgsdgsa"),
("abc", "ccccccabbbbbbb")
]
for tdata in testVector:
    assert(hasAnagram(*tdata) == hasAnagram0(*tdata))
'''
https://www.hackerrank.com/challenges/sherlock-and-anagrams/problem
Given a string, find the number of pairs of substrings of the string that are anagrams of each other.
The order of letters in a pair of anagrams doesn't matter, only what the letters are.
1) Start by iterating over the string to extract substrings
2) Instead of storing every substring to later compare it against every other single substring: Sort substring alphabetically before storing
3) Store the sorted string in a map of string to number of occurences
4) Every time you find a new substring, sort it, if there is already a match or several in your hashmap, add the number of already existing occurences to your anagram counter, increment in hashmap
'''
def sherlockAndAnagrams(s):
    ssFreqs = Counter(s)

    for l in range(2, len(s)): # N**2
        for i in range(0,len(s)-l+1):
            key = frozenset(Counter(s[i:i+l]).items())
            ssFreqs[key] += 1
    return sum(map(lambda x: x*(x-1)//2, ssFreqs.values())) # number of pairs of substrings that are anagrams of each other

def sherlockAndAnagrams1(string):
    buckets = Counter()
    for i in range(len(string)):
        for j in range(1, len(string) - i + 1):
            key = frozenset(Counter(string[i:i+j]).items()) # O(N) time key extract
            buckets[key] = buckets[key] + 1
    count = 0
    for key in buckets:
        count += buckets[key] * (buckets[key]-1) // 2
    return count

# return an integer representing the minimum total characters that must be deleted to make the strings anagrams.
def makeAnagram(a, b): #N
    aC = Counter(a)
    bC = Counter(b)
    aC.subtract(bC)
    minDels = 0
    for k,val in aC.items():
        minDels += val if val >= 0 else -val

    return minDels

# valid if he can remove just 1 character at 1 index in the string, and the remaining characters will occur the same number of times.
def isValid(s):
    chrCount = Counter(s)
    chrCvS = sorted(chrCount.values())
    maxF = chrCvS[-1]
    minF = chrCvS[0]
    if 0 < len(s) <= 3 or minF == maxF or maxF-1 == minF and chrCvS[-2] == minF or minF == 1 and chrCvS[1] == maxF:
         return "YES"
    else:
        return "NO"


#{ a: 11111, b: 11111, c: 11111, d: 11111, e: 11111, f: 11111, g: 11111, h: 11111, i: 11111, p: 1 }
if __name__ == "__main__":
    ss = "ibfdgaeadiaefgbhbdghhhbgdfgeiccbiehhfcggchgghadhdhagfbahhddgghbdehidbibaeaagaeeigffcebfbaieggabcfbiiedcabfihchdfabifahcbhagccbdfifhghcadfiadeeaheeddddiecaicbgigccageicehfdhdgafaddhffadigfhhcaedcedecafeacbdacgfgfeeibgaiffdehigebhhehiaahfidibccdcdagifgaihacihadecgifihbebffebdfbchbgigeccahgihbcbcaggebaaafgfedbfgagfediddghdgbgehhhifhgcedechahidcbchebheihaadbbbiaiccededchdagfhccfdefigfibifabeiaccghcegfbcghaefifbachebaacbhbfgfddeceababbacgffbagidebeadfihaefefegbghgddbbgddeehgfbhafbccidebgehifafgbghafacgfdccgifdcbbbidfifhdaibgigebigaedeaaiadegfefbhacgddhchgcbgcaeaieiegiffchbgbebgbehbbfcebciiagacaiechdigbgbghefcahgbhfibhedaeeiffebdiabcifgccdefabccdghehfibfiifdaicfedagahhdcbhbicdgibgcedieihcichadgchgbdcdagaihebbabhibcihicadgadfcihdheefbhffiageddhgahaidfdhhdbgciiaciegchiiebfbcbhaeagccfhbfhaddagnfieihghfbaggiffbbfbecgaiiidccdceadbbdfgigibgcgchafccdchgifdeieicbaididhfcfdedbhaadedfageigfdehgcdaecaebebebfcieaecfagfdieaefdiedbcadchabhebgehiidfcgahcdhcdhgchhiiheffiifeegcfdgbdeffhgeghdfhbfbifgidcafbfcd"
    for s in ["aaaaabc","aabbcd", ss, "a"*111 + "b"*111 + "c"*111 + "p"*1, "aaabbcc","abcdefghhgfedecba","aabbcd"]:
        print(isValid(s))

    # print(makeAnagram("cde","abc"))

    for s in ["ifailuhkqq","kkkk","abba","abcd"]:
        assert sherlockAndAnagrams(s) == sherlockAndAnagrams1(s)
