def merge(l,r):
    lIdx = rIdx = 0
    result = []
    while lIdx < len(l) or rIdx < len(r):
        if lIdx == len(l):
            return result + r[rIdx:]
        if rIdx == len(r):
            return result + l[lIdx:]
        if l[lIdx] < r[rIdx]:
            result.append(l[lIdx])
            lIdx += 1
        else:
            result.append(r[rIdx])
            rIdx += 1
    return result

def mergeSort(a):
    lenA = len(a)
    if lenA == 1:
        return a
    elif lenA == 2:
        return a if a[0] < a[1] else a[::-1]

    m = len(a)//2
    left = mergeSort(a[:m])
    right = mergeSort(a[m:])
    return merge(left,right)

print(mergeSort([12, 11, 13, 5, 6, 7]))
