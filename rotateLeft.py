from collections import deque
import string

def rotLeft(a, d):
    _a = deque(a)
    for _ in range(d):
        _a.append(_a[0])
        _a.popleft()

    return _a

def rotleft(a, d):
    return a[d:] + a[:d]

print(rotLeft(list(string.ascii_lowercase), 2))
print(rotleft(list(string.ascii_lowercase), 2))
