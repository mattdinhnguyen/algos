# Complete the getMinimumCost function below.
def getMinimumCost0(k, c):
    round = 0
    minCost = 0
    j = k
    sc = sorted(c)
    for i in range(len(c)-1,-1,-1):
        minCost += sc[i]*(round+1)
        j -= 1
        if j == 0:
            j = k
            round += 1
    return minCost

def getMinimumCost(k, c):
    round = 0
    minCost = 0
    sc = sorted(c, reverse=True)
    for i in range(0,len(c),k):
        minCost += sum(p*(round+1) for p in sc[i:i+k])
        round += 1
    return minCost

        
if __name__ == '__main__':

    for c,k in [([2, 5, 6],2),([1, 3, 5, 7, 9],3)]:
        print(getMinimumCost(k, c),getMinimumCost0(k, c))
   