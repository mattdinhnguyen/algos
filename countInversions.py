from bestFirstSearchDP import is_sorted
# space/time N/N^2
def countInversions0(arr):
    inverts = 0
    for i in range(len(arr)-1):
        for j in range(i+1,len(arr)):
            if arr[i] > arr[j]:
                arr[i],arr[j] = arr[j],arr[i]
                inverts += 1
    assert(is_sorted(arr))
    return inverts

def sort_pair(arr0, arr1):
    if len(arr0) > len(arr1):
        return arr1, arr0
    else:
        return arr0, arr1

def merge(arr0, arr1):
    inversions = 0
    result = []
    # two indices to keep track of where we are in the array
    i0 = 0
    i1 = 0
    # probably doesn't really save much time but neater than calling len() everywhere
    len0 = len(arr0)
    len1 = len(arr1)
    while len0 > i0 and len1 > i1:
        if arr0[i0] <= arr1[i1]:
            result.append(arr0[i0])
            i0 += 1
        else:
            # count the inversion right here: add the length of left array
            inversions += len0 - i0
            result.append(arr1[i1])
            i1 += 1

    if len0 == i0:
        result += arr1[i1:]
    elif len1 == i1:
        result += arr0[i0:]

    return result, inversions 
   
def merge0(arr0, arr1):
    inversions = 0
    result = []
    while len(arr0) > 0 and len(arr1) > 0:
        if arr0[0] <= arr1[0]:
            result.append(arr0.pop(0))
        else:
            # count the inversion right here: add the length of left array
            inversions += len(arr0)
            result.append(arr1.pop(0))
            
    if len(arr0) == 0:
        result += arr1
    elif len(arr1) == 0:
        result += arr0
        
    return result, inversions
# space/time N/NlogN
def sort(arr):
    length = len(arr)
    mid = length//2
    if length > 2:
        sorted_0, counts_0 = sort(arr[:mid])
        sorted_1, counts_1 = sort(arr[mid:])
        result, counts = merge(sorted_0, sorted_1)
        return result, counts + counts_0 + counts_1
    else:
        return (arr[::-1], 1) if (length == 2 and arr[0] > arr[1]) else (arr, 0)

def countInversions(a):
    final_array, inversions = sort(a)
    # print(final_array)
    print(countInversions0(a))
    for i in range(len(a)):
        if a[i] != final_array[i]:
            print(i,a[i],final_array[i])
    assert(a == final_array)
    return inversions
      
# 38046 78149 97560 174498 3083
if __name__ == '__main__':
    # print(countInversions([3,2,1]))
    fptr = open("countInversions.ut")

    for t in range(int(fptr.readline().rstrip())):
        n = fptr.readline().rstrip()
        arr = list(map(int, fptr.readline().rstrip().split()))
        ans = countInversions(arr)
        print(str(ans) + '\n')

    fptr.close()

