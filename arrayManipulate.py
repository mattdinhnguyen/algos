from timeit import timeit
# https://www.hackerrank.com/challenges/crush/problem
# Starting with a 1-indexed array of zeros and a list of operations, for each operation add a value to each of the array element between two given indices, inclusive.
# Once all operations have been performed, return the maximum value in the array.
@timeit
def arrayManipulationBrute(n, queries):
    arr = [0]*n
    for a,b,k in queries:
        for i in range(a-1,b):
            arr[i] += k
    return max(arr)

@timeit
def arrayManipulation(n, fptr, m):
    arr = [0]*(n+2)
    # for a,b,k in generator_queries(fptr,m):
    for a,b,k in (map(int,fptr.readline().rstrip().split()) for _ in range(m)):
        arr[a] += k
        arr[b+1] -= k
    maxVal = sum = 0
    for val in arr:
        sum += val
        maxVal = max(maxVal, sum)
    return maxVal

def generator_queries(fptr, queries):
    init = 0
    index = 0
    while index <  queries:
        yield map(int, fptr.readline().rstrip().split())
        index += 1

if __name__ == '__main__':
    fptr = open("arrayManipulate.ut7", 'r')

    nm = fptr.readline().split()

    n = int(nm[0])

    m = int(nm[1])

    # queries = []

    # for _ in range(m):
    #     queries.append(list(map(int, fptr.readline().rstrip().split())))

    # result = arrayManipulationBrute(n, queries)
    result = arrayManipulation(n, fptr, m)
    print(result)

    fptr.close()
