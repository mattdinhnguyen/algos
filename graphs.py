from collections import deque
from typing import List

EDGE_DISTANCE = 6
class Graph:
    # def __init__(self, n):
    #     self.edges = dict((i,[]) for i in range(n))
    def __init__(self, grid):
        self.grid = grid
        self.edges = dict()
        # self.edges = dict([((i,j),[]) for i in range(len(grid)) for j in range(len(grid[0]))])

    def connect(self, x, y):
        self.edges[x].append(y)
        self.edges[y].append(x)

    def findIslands(self):
        islands = 0
        visited = set()
        dqu = deque()
        for xy in self.edges:
            if self.grid[xy[0]][xy[1]] and xy not in visited:
                visited.add(xy)
                dqu.append(xy)
                islands += 1
                while dqu:
                    top = dqu.pop()
                    for neibr in self.edges[top]:
                        if neibr not in visited:
                            dqu.append(neibr)
                            visited.add(neibr)
        return islands

    def findMaxRegion(self):
        maxReg = 0
        visited = set()
        dqu = deque()
        for xy in self.edges:
            if self.grid[xy[0]][xy[1]] and xy not in visited:
                visited.add(xy)
                dqu.append(xy)
                regCount = 1
                while dqu:
                    top = dqu.pop()
                    for neibr in self.edges[top]:
                        if neibr not in visited:
                            dqu.append(neibr)
                            visited.add(neibr)
                            regCount += 1
                maxReg = max(maxReg, regCount)
        return maxReg

    def find_all_distances(self, s):
        distance2s = dict((i, -1) for i in range(n))
        distance2s[s] = 0
        dque = deque([s])

        while dque:
            head = dque.popleft()
            for neibor in self.edges[head]:
                if distance2s[neibor] == -1:
                    distance2s[neibor] = distance2s[head] + EDGE_DISTANCE
                    dque.append(neibor)
            
        return [d for n,d in distance2s.items() if n != s]

def createGraph(grid):
    xMax = len(grid)
    yMax = len(grid[0])
    def findNeibors(x,y):
        xrang = [max(0,x-1),min(xMax,x+1)]
        yrang = [max(0,y-1),min(yMax,y+1)]
        n = []
        for x1 in range(*xrang):
            for y1 in range(*yrang):
                if x1 != x and y1 != y: continue
                if (x,y) != (x1,y1) and grid[x1][y1] == grid[x][y]:
                    n.append((x1,y1))
        return n

    g = Graph(grid)
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1: 
                g.edges.setdefault((i,j),[])
                for neibor in findNeibors(i,j):
                   g.connect((i,j),neibor)
    return g                    
        
class Solution:
    def __init__(self):
        pass

    def numIslands(self, grid):
        if not grid:
            return 0
        
        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    self.dfs(grid, i, j)
                    count += 1
        return count

    def dfs(self, grid, i, j):
        if i<0 or j<0 or i>=len(grid) or j>=len(grid[0]) or grid[i][j] != 1:
            return
        grid[i][j] = 0
        self.dfs(grid, i+1, j)
        self.dfs(grid, i-1, j)
        self.dfs(grid, i, j+1)
        self.dfs(grid, i, j-1)

if __name__ == '__main__':
    # fptr = open("bfs.ut", "r")
    fptr = open("matrices.ut", "r")
    # fptro = open("bfs.uto", "r")
    t = int(fptr.readline())
    for i in range(t):
        # n,m = [int(value) for value in fptr.readline().split()]
        # graph = Graph(n)
        # for i in range(m):
        #     x,y = [int(x) for x in fptr.readline().split()]
        #     graph.connect(x,y) 
        # s = int(fptr.readline())
        # ans = graph.find_all_distances(s)
        # expected = [int(value) for value in fptro.readline().split()]
        # print([(a,e) for a,e in zip(ans,expected) if a != e])
        # assert(ans == expected)
        n = int(fptr.readline().rstrip())
        m = int(fptr.readline().rstrip())
        grid = []

        for _ in range(n):
            grid.append(list(map(int, fptr.readline().rstrip().split())))
        g = createGraph(grid)
        print(g.findMaxRegion(),g.findIslands(),Solution().numIslands(grid))
