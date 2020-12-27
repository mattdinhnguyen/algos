
def test():
    assert 1.0 == probAlive(5, 0, 0, 0)
    assert 0.5 == probAlive(5, 1, 0, 0)
    assert 0.75 == probAlive(5, 1, 1, 4)
    assert 1.0 == probAlive(5, 2, 2, 2)
    assert 0.9375 == probAlive(5, 3, 2, 2)

def probAlive(N,n,x,y):
    probAliveMap = dict()
    if (x < 0 or x >= N or y < 0 or y >= N): return 0.0
    if (n == 0): return 1.0
    if (x,y,n) in probAliveMap: return probAliveMap((x,y,n))
    p = 0.0
    if (x > 0): p += 0.25 * probAlive(N, n-1, x-1, y)
    if (x+1 < N): p += 0.25 * probAlive(N, n-1, x+1, y)
    if (y > 0): p += 0.25 * probAlive(N, n-1, x, y-1)
    if (y+1 < N): p += 0.25 * probAlive(N, n-1, x, y+1)
    print("({},{},{}): {}".format(x,y,n,p))
    return p

test()
