from collections import defaultdict
# https://www.hackerrank.com/challenges/candies/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=dynamic-programming

def candies(n, a):
    c, desc_buf = [1]*n, []

    for i in range(1, n):
        if a[i] < a[i-1]:
            if not desc_buf: #descending buffer
                desc_buf = [i-1]
            desc_buf.append(i)
            if not i == n-1:
                continue
        if a[i] > a[i-1]:
            c[i] = c[i-1] + 1
        if desc_buf:
            for extra, idx in enumerate(desc_buf[::-1]):
                c[idx] = max(c[idx], extra + 1)
            del desc_buf[:]

    return sum(c)

if __name__ == '__main__':
    fptr = open("candies.ut")

    n = int(fptr.readline())

    arr = []

    for _ in range(n):
        arr_item = int(fptr.readline())
        arr.append(arr_item)

    result = candies(n, arr)

    print(161518, result)

    fptr.close()
