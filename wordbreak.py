from words import WordDictionary, WordSearch
from timeit import timeit

class Solution:
    # https://leetcode.com/problems/word-break/discuss/43788/4-lines-in-Python
    @timeit
    def wordBreak0(self, s, wordDict):
        words = [k for k,_ in wordDict.items()]
        ok = [True]
        max_len = max(map(len,words+['']))
        words = set(words)
        for i in range(1, len(s)+1):
            ok += any(ok[j] and s[j:i] in words for j in range(max(0, i-max_len),i)),
        return ok[-1]
    # https://leetcode.com/problems/word-break/
    @timeit
    def wordBreak1(self, s, wordDict): # n + n-1 + n-2 + ... = n(n+1)/2
        wordCache = dict()
        wordSet = set(wordDict)
        def dfs(s, wordSet):
            if not s:
                return True
            elif s in wordCache:
                return wordCache[s]
            for i in range(1,len(s)+1): # match with words in increasing length
                if s[:i] in wordSet and dfs(s[i:], wordSet): # match s prefix with shortest words, repeat this step with remaining of s
                    wordCache[s[i:]] = True
                    return True # return at 1st successful attempt
            wordCache[s] = False
            return False
        return dfs(s, wordSet)
    # https://leetcode.com/problems/word-break/discuss/43808/Simple-DP-solution-in-Python-with-description
    # d is an array of booleans.
    # d[i] is True if there is a word in the dictionary that ends at ith index of s AND d is also True at the beginning of the word
    # s = "leetcode" words = ["leet", "code"]
    # d[3] is True because there is "leet" in the dictionary that ends at 3rd index of "leetcode"
    # d[7] is True because there is "code" in the dictionary that ends at the 7th index of "leetcode" AND d[3] is True
    @timeit
    def wordBreak(self, s, wordDict): #  O(s * k *2s) => O(s^2 *k) . String loop + word loop + slicing/comparison (takes s time each)
        dp = [False]*(len(s)+1)        # s = len(s) and k = len(wordDict)
        dp[0] = True
        for i in range(1,len(s)+1): # s
            for w in wordDict: # k
                if s[i-len(w):i] == w and dp[i-len(w)]: # s[i-len(w):i] == w takes 2s
                    dp[i] = True # w ends at i-1, and substring ending at i-len(w)-1 is wordbreak valid
                    break
        return dp[-1] # whole s is wordbreak valid
    def wordcheck(self,str,hash): # count number of words in hash len(hash)*len(str)
        result = {}        
        for k,v in hash.items(): # hash length
            x = len(k)
            word_count = i = 0
            while i < len(str): # str length
                if str[i:x+i] == k:
                    word_count +=1
                    i += x
                    if word_count == v:
                        result[k] = word_count
                        break
                else:
                    i += 1
            if word_count: # when i == len(str), update word_count in result[k]
                result[k] = word_count
        return (result == hash, result) # confirm str has the number of words in hash

def factors(n):
    f = []
    for i in range(1, n//2+1):
        if n % i == 0:
            f.append(i)
    return f

def periodic(s):
    f = factors(len(s))
    for i in f:
        m = len(s)/i
        if s[:i] * int(m) == s:
            print(i, s[:i], s)
            return True
    return False

def repeatedSubstringPattern(s):
    slen = len(s)
    for i in range(2,slen//2+1):
        j = sublen = slen//i
        while j <= sublen*(i-1) and s[j-sublen:j] == s[j:j+sublen]:
            if j == sublen*(i-1) and slen == sublen*i:
                return True
            j += sublen
    return True if slen > 1 and len(set(s)) == 1  else False

if __name__ == "__main__":
    sol = Solution()
    tdata = [({'apple':1,'pear':1,'pie':1},'applepie'),({'leet':1,'code':1}, 'leetcode'),({'abc':3, 'ab':2, 'abca':1}, 'abcabcabca'),({'abc':3, 'ab':2, 'abca':1}, 'abcx'),({'abc':3, 'ab':2},'abcabab')]
    for wordhash, word in tdata:
        print(sol.wordcheck(word, wordhash)),
        print(sol.wordBreak(word, wordhash),sol.wordBreak0(word, wordhash),sol.wordBreak1(word, wordhash))
    # tdata = ["xxxxxx","a","aabaaba","bb","abab","ababab","abcabcabcabc","abcabcabc"]
    # for s in tdata:
    #     print(repeatedSubstringPattern(s),periodic(s))
