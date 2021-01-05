import multiprocessing
import numpy
import time

def arrayManipulation(n, queries):
    arr = [0]*(n+2)
    for a,b,k in queries:
        arr[a] += k
        arr[b+1] -= k
    for i in range(1,n+2):
        arr[i] += arr[i-1]
    return arr

if __name__ == '__main__':
    fptr = open("arrayManipulate.ut7", 'r')

    nm = fptr.readline().split()

    n = int(nm[0])

    m = int(nm[1])

    queries = []
    ts = time.time()
    for _ in range(m):
        queries.append(list(map(int, fptr.readline().rstrip().split())))

    q = len(queries)//8
    data = [[n, queries[q*i:q*(i+1)]] for i in range(8)]
    results = multiprocessing.Pool(8).starmap(arrayManipulation, data)
    result = max(numpy.sum(results, axis=0))

    print(result)
    te = time.time()
    print(f'{te-ts:2.2f} sec')
    fptr.close()