def hammingDistance(x, y):
    """
    :type x: int
    :type y: int
    :rtype: int
    """
    x = x ^ y
    y = 0
    while x:
        y += 1  # reuse y to save mem allocation to cut execution time
        x = x & (x - 1) # shift 1 bit right then and to clear it
    return y

if __name__ == '__main__':
    print(hammingDistance(29,30))
