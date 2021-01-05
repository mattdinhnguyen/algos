import sys
import random
from collections import defaultdict, deque
from dijsktraheap import Graph
from typing import List, Dict, DefaultDict, Tuple, Deque

def xEdges(x, r):
    for y, v in enumerate(r):
        if y < yMax:
            if r[y+1] == v:
                G.add_edge((x,y), (x,y+1), 1)
            elif (x,y) not in G.edges:
                G.edges.setdefault((x,y),[])

def yEdges(x, r, nextR):
    for y, v in enumerate(r):
        if nextR[y] == v:
            G.add_edge((x,y), (x+1,y), 1)

def createEdges(mData):
    for x, r in enumerate(mData):
        xEdges(x, r)
        if x < xMax:
            yEdges(x, r, mData[x+1])

def countEast(x, y):
    count = 1
    next_pos = []
    value = TestData[x][y]
    while y < yMax and TestData[x][y+1] == value:
        count += 1
        y += 1
        if x < xMax and y <= yMax and TestData[x+1][y] == value:
            c, _x, _y = countSouth(x, y)
            count += c
            if _x < xMax or _y < yMax:
                next_pos.append((_x, _y))
    if x < xMax or y < yMax:
        next_pos.append(x, y)
    return (count, next_pos)

def countSouth(x, y):
    pass

# print(countEast(0, 0))

Counted = set()
def countElems(x, y):
    count = 0
    if (x,y) not in Counted:
        value = TestData[x][y]
        if y < yMax:
            count += countEast(x, y)
        if x > 0:
            count += countWest(x, y)
    count += countNorth(x, y)
    count += countSouth(x, y)
    for r in range(x, rMax):
        for c in range(y, cMax):
            if TestData[r][c] == color:
                count += 1
            elif r < rMax and TestData[r+1][0] == color:
                break
            else:
                return(count, r, c, TestData[r][c])

TestData: List[List[int]] = dict()
def testData(rows, cols, value):
    for r in range(rows):
        TestData[r] = [ random.randint(0, value) for c in range(cols) ]
    return TestData

# testData(5, 5, 1)

vMax: int = 1
TestData: List[List[int]] = [[0, 0, 0, 1, 0],[1, 0, 1, 0, 0],[0, 1, 0, 1, 1],[0, 0, 1, 1, 1],[0, 1, 1, 1, 0]]
xMax: int = len(TestData) - 1
yMax: int = len(TestData[0]) - 1
for r in TestData:
    print(r)

vB: DefaultDict[int, List[Tuple[int,int]]] = defaultdict(list)
# for v in range(vMax+1):
#     vB[v] = list()

def createVBin(mData) -> None:
    for x, r in enumerate(mData):
        for y, v in enumerate(r):
            vB[v].append((x,y))

createVBin(TestData)
for v, pos in vB.items():
    print(v, pos)

nB: DefaultDict[Tuple[int,int],List[Tuple[int,int]]] = defaultdict(list)

def areNeighbors(a,b):
    return a[0] == b[0] and abs(a[1] - b[1]) == 1 or a[1] == b[1] and abs(a[0] - b[0]) == 1

def createXNBin(vB: DefaultDict[int, List[Tuple[int,int]]]) -> None:
    for vb in vB.values():
        currentB = nB[vb[0]] = [vb[0]]
        for i, pos in enumerate(vb[:-1]):
            npos = vb[i+1]
            if any([areNeighbors(_pos, npos) for _pos in currentB]):
                currentB.append(npos)
            else:
                currentB = nB[npos] = [npos]

createXNBin(vB)
for pos, bin in nB.items():
    print(pos, bin)

# G = Graph()
# createEdges(TestData)
# for k,v in G.edges.items():
#     print(k,v)
lenX = len(TestData)
lenY = len(TestData[0])
def findNeighbors(pos: Tuple[int,int]) -> List[Tuple[int,int]]:
    v: int = TestData[pos[0]][pos[1]]
    r = set()
    for x in [pos[0]-1,pos[0]+1]:
        if -1 < x < lenX and TestData[x][pos[1]] == v:
            r.add((x,pos[1]))
    for y in [pos[1]-1,pos[1]+1]:
        if -1 < y < lenX and TestData[pos[0]][y] == v:
            r.add((pos[0],y))
    return r

graph: DefaultDict[Tuple[int,int],List[Tuple[int,int]]] = defaultdict(list)

for x in range(lenX):
    for y in range(lenY):
        graph[(x,y)] = findNeighbors((x,y))

visited = set()

for x in range(lenX):
    for y in range(lenY):
        if (x,y) not in visited and graph[(x,y)]:
            visited.add((x,y))
            stack: Deque[Tuple[int,int]] = deque(graph[(x,y)])
            while stack:
                pos = stack.pop()
                visited.add(pos)
                for i in graph[pos]:
                    if i not in visited:
                        stack.append(i)
                        graph[(x,y)].add(i)
            
 
for k,v in graph.items():
    print(k, v)
