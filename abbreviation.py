# https://www.hackerrank.com/challenges/abbr/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=dynamic-programming

def abbreviation0(a, b): # failed 12,13,14
    if len(a) < len(b):
        return("NO")
    j=len(b)-1
    res=[]
    n=-1*len(a)
    for i in range(-1,n-1,-1):
        if(a[i].upper()==b[j] and j>(len(b)*-1)):
            if a[i].islower():
                res.append(a[i])
            j-=1
        elif a[i].isupper() and a[i].lower() in res:
            res.remove(a[i].lower())
        elif a[i].isupper():
            return("NO")
    return("YES")

def abbreviation(a, b):
    m, n = len(a), len(b)
    dp = [[False]*(m+1) for _ in range(n+1)]
    dp[0][0] = True
    for i in range(n+1):
        for j in range(m+1):
            if i == 0 and j != 0:
                dp[i][j] = a[j-1].islower() and dp[i][j-1]
            elif i != 0 and j != 0:
                if a[j-1] == b[i-1]:
                    dp[i][j] = dp[i-1][j-1]
                elif a[j-1].upper() == b[i-1]:
                    dp[i][j] = dp[i-1][j-1] or dp[i][j-1]
                elif a[j-1].islower():
                    dp[i][j] = dp[i][j-1]
    return "YES" if dp[n][m] else "NO"

if __name__ == '__main__':
    fptr = open("abbreviation.ut", 'r')
    fptro = open("abbreviation.uto", 'r')
    q = int(fptr.readline())
    for q_itr in range(q):
        a = fptr.readline().strip()
        b = fptr.readline().strip()
        expected = fptro.readline().strip()
        ans = abbreviation(a, b)
        print(ans,expected,abbreviation0(a, b))
        assert(ans == expected)
