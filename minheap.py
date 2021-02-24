from typing import List
from sys import stdin, maxsize
from heapq import heappush, heappop, merge

class MyBST():

    def __init__(self, vals: List[int] = []) -> None:
        self.heap: List[int] = []
        self.len: int = 0
        for v in vals:
            self.insert(v)

    def insert(self, v: int) -> int:
        curIdx = 0
        while curIdx < self.len:
            if self.heap[curIdx] is None:
                break
            if v < self.heap[curIdx]:
                curIdx = curIdx*2 + 1                    
            else:
                curIdx = curIdx*2 + 2

        if curIdx == 0:
            if curIdx < self.len:
                self.heap[curIdx] = v
            else:
                self.heap.append(v)
                self.len += 1
        else:
            while curIdx >= self.len:
                self.heap += [None]*2
                self.len += 2
            self.heap[curIdx] = v

        return v

    def verifyBST(self) -> bool:
        curIdx = 0
        while curIdx < self.len:
            leftIdx = curIdx*2 + 1
            rightIdx = curIdx*2 + 2
            if rightIdx >= self.len:
                break
            if self.heap[leftIdx] and self.heap[curIdx] and self.heap[leftIdx] > self.heap[curIdx]:
               return False
            if self.heap[rightIdx] and self.heap[curIdx] and self.heap[rightIdx] < self.heap[curIdx]:
                return False
            curIdx += 1
        return True

    def findMax(self, cIdx: int) -> int:
        rIdx = cIdx*2 + 2
        if rIdx < self.len and self.heap[rIdx]:
            return self.findMax(rIdx)
        else :
            return cIdx

    def findMin(self, cIdx: int) -> int:
        lIdx = cIdx*2 + 1
        if lIdx < self.len and self.heap[lIdx]:
            return self.findMin(lIdx)
        else:
            return cIdx

    def delete(self, v: int, curIdx: int = 0) -> int:
        while curIdx < self.len:
            if self.heap[curIdx] == v:
                break
            elif self.heap[curIdx] and self.heap[curIdx] > v:
                curIdx = curIdx*2 + 1
            else:
                curIdx = curIdx*2 + 2

        leftIdx = curIdx*2 + 1
        rightIdx = curIdx*2 + 2
        if self.len - rightIdx < 2 and not any(self.heap[leftIdx:leftIdx+2]):
            self.heap[curIdx] = None
        elif leftIdx < self.len and self.heap[leftIdx]:
            idx = self.findMax(leftIdx)
            self.heap[curIdx] = self.heap[idx]
            if any(self.heap[leftIdx:leftIdx+2]):
                self.delete(self.heap[idx], idx)
        elif rightIdx < self.len:
            idx = self.findMin(rightIdx)
            self.heap[curIdx] = self.heap[idx]
            if any(self.heap[leftIdx:leftIdx+2]):
                self.delete(self.heap[idx], idx)
        return v

class Node:
    def __init__(self, val: int):
        self.value = val
        self.left = None
        self.right = None
        self.parent = None
        self.level = None

class MyBST2():

    def __init__(self, vals: List[int] = []) -> None:
        self.root: Node = None
        for v in vals:
            self.insert(self.root, Node(v))

    def insert(self, cur: Node, new: Node) -> Node:
        while (cur):
            if new.value <= cur.value:
                if cur.left:
                    self.insert(cur.left, new)
                else:
                    cur.left = new
                    new.parent = cur
                    break
            else:
                if cur.right:
                    self.insert(cur.right, new)
                else:
                    cur.right = new
                    new.parent = cur
                    break
        else:
            cur = new

        return new

    def verifyBST(self, cur: Node) -> bool:
        ret = True
        if cur.right:
            if cur.right.value < cur.value:
                return False
            else:
                ret = verifyBST(cur.right)
                if not ret:
                    return ret
        if cur.left:
            if cur.left.value > cur.value:
                return False
            else:
                ret = verifyBST(cur.left)

        return ret

    def findMax(self, cur: Node) -> Node:
        return findMax(cur.right) if cur.right else cur

    def findMin(self, cur: Node) -> Node:
        return findMin(cur.left) if cur.left else cur

    def delete(self, v: int, cur: Node) -> None:
        while cur:
            if cur.value > v:
                delete(v, cur.left)
            elif cur.value < v:
                delete(v, cur.right)
            elif cur.left:
                replace = findMax(cur.left)
                cur.value = replace.value
                if replace.value <= replace.parent.value:
                    replace.parent.left = None
                else:
                    replace.parent.right = None
            elif cur.right:
                replace = findMin(cur.right)
                cur.value = replace.value
            elif cur.parent.value < cur.value:
                cur.parent.right = None
            else:
                cur.parent.left = None

heap = []
item_lookup = set()

def push(v):
    heappush(heap, v)
    item_lookup.add(v)
    
def discard(v):
    item_lookup.discard(v)
    
def print_min():
    while heap[0] not in item_lookup:
        heappop(heap)
      
    print(heap[0])
    
cmds = {
    1: push,
    2: discard,
    3: print_min
}

fptr = open("minheap.txt", "r")
n = int(fptr.readline())
for _ in range(n):
    data = list(map(int,fptr.readline().split(" ")))
    cmds[data[0]](*data[1:])

h = MyBST()
def print_bstMin():
    # print(h.heap, h.heap[h.findMin(0)], h.heap[h.findMax(0)])
    print(h.heap[h.findMin(0)])
    assert(h.heap[h.findMin(0)] == int(fptro.readline()))

cmds = {
    1: h.insert,
    2: h.delete,
    3: print_bstMin
}

fptr = open("minheap.ut", "r")
fptro = open("minheap.uto", "r")
n = int(fptr.readline())
for _ in range(n):#     data = list(map(int,fptr.readline().split(" ")))
    cmds[data[0]](*data[1:])

# print(f"{MIN_INT},{maxsize}")
# h = MyBST([15, 4, 10, 24, 26, 9, 12, 20, 5, 18, 22, 30])
# print(h.heap,h.len,h.heap[h.findMax(0)])
# h.delete(30)
# print(h.heap,h.len,h.heap[h.findMax(0)])
# h.delete(26)
# print(h.heap,h.len,h.heap[h.findMax(0)])
testData = [15, 4, 10, 24, 26, 9, 12, 20, 5, 18, 22, 30]
# testData = [15,20]
print(testData)
h = MyBST2()

for v in testData:
    h.insert(h.root, Node(v))
    if not h.verifyBST():
        print(f"Insert {v}", h.heap, h.heap[h.findMin(0)], h.heap[h.findMax(0)])
# for v in sorted(testData):
#     print(f"Delete {v}", h.heap, h.heap[h.findMin(0)], h.heap[h.findMax(0)])
#     h.delete(v)
#     if not h.verifyBST():
#         print(f"Delete {v}", h.heap, h.heap[h.findMin(0)], h.heap[h.findMax(0)])
for v in sorted(testData, key=None, reverse=True):
    print(f"Delete {v}", h.heap, h.heap[h.findMin(0)], h.heap[h.findMax(0)])
    h.delete(v)
    if not h.verifyBST():
        print(f"Delete {v}", h.heap, h.heap[h.findMin(0)], h.heap[h.findMax(0)])
# h.delete(15)
# print(h.heap, h.heap[h.findMin(0)], h.heap[h.findMax(0)])
# h.delete(10)
# print(h.heap, h.heap[h.findMin(0)], h.heap[h.findMax(0)])

# for v in testData[::-1]:
#     h.delete(v)
#     print(h.heap, h.heap[h.findMin(0)], h.heap[h.findMax(0)])
n,a = input(),sorted(map(int, input().split()))
print(min(abs(x-y) for x,y in zip(a,a[1:])))
if len(arr) != len(list(set(arr))):
    result = 0