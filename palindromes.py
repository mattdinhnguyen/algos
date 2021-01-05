def ispldromes(s):
    if len(s)%2:
        mid = len(s)//2
        return 1 if set(s[0]) == set(s[:mid]) == set(s[mid+1:]) else 0
    else:
        return 1 if set(s) == set(s[0]) else 0
    
def substrCount0(n, s):
    pldromes = len(s)
    if ispldromes(s):
        pldromes += 1
    for l in range(2,n):
        for j in range(0,n-l+1):
            pldromes += ispldromes(s[j:j+l])
    return pldromes
'''1. Build a list of tuples such that the string "aaabbc" can be squashed down to [("a", 3), ("b", 2), ("c", 1)].
2. add to answer all combinations of substrings from these tuples which would represent palindromes which have all same letters.
3. traverse this list to specifically find the second case mentioned in probelm
'''
def substrCount(n, s):
    cnt = 1
    cur = s[0]
    tl = []
    # step 1
    for i in range(1,n):
        if s[i] == cur:
            cnt += 1
        else:
            tl.append((cur,cnt))
            cur = s[i]
            cnt = 1
    tl.append((cur,cnt))
    # step 2
    ssCnt = sum([cnt*(cnt+1)//2 for _, cnt in tl])
    # step 3
    for i in range(1, len(tl) - 1):
        if tl[i - 1][0] == tl[i + 1][0] and tl[i][1] == 1:
            ssCnt += min(tl[i - 1][1], tl[i + 1][1])

    return ssCnt


def substrCount0(n, s): # N
    l = []
    count = 0
    cur = None

    # 1st pass N
    for i,c in enumerate(s):
        if c == cur:
            count += 1
        else:
            if cur is not None:
                l.append((cur, count))
            cur = c
            count = 1
    l.append((cur, count))

    ans = 0
		
    # 2nd pass N
    for _,count in l:
        ans += count * (count + 1) // 2

    # 3rd pass N
    for i in range(1, len(l) - 1):
        if l[i - 1][0] == l[i + 1][0] and l[i][1] == 1:
            ans += min(l[i - 1][1], l[i + 1][1])

    return ans

tdata = ["asasd"]
for d in tdata:
    print(d, substrCount0(len(d), d), substrCount(len(d), d))
