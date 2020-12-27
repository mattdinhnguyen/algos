from collections import defaultdict, Counter
from copy import copy

# Driver program to test above function 
arr = [-5, -10, 0, -3, 8, 5, -1, 10] 
# ans = count_sort(arr) 
# print("Sorted character array is " + str(ans))

def lb2xMedian(countArr, d):
    sArr = []
    for v,t in countArr.items():
        sArr.extend([v]*t)
    med = d//2
    return sArr[med]*2 if d%2 else sArr[med-1]+sArr[med]


def activityNotifications(expenditure, d):
    notifies = 0
    valCounts = Counter(expenditure[:d])
    for i in range(d,len(expenditure)):
        median2x = lb2xMedian(valCounts, d)
        if expenditure[i] >= median2x:
            notifies += 1
        valCounts[expenditure[i-d]] -= 1
        valCounts[expenditure[i]] += 1
       
    return notifies

if __name__ == '__main__':
    fptr = open("fraudNotify.ut", "r")
    n, d = list(map(int, fptr.readline().rstrip().split()))

    expenditures = list(map(int, fptr.readline().rstrip().split()))

    print(activityNotifications(expenditures, d)) # 633
    print(activityNotifications([1, 2, 3, 4, 4], 4))
    print(activityNotifications([2,3,4,2,3,6,8,4,5], 5))
