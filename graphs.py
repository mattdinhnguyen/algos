from collections import deque, defaultdict
from heapq import heappop, heappush, heapify
from itertools import product
from typing import List
import os, math, json

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
    def numIslands(self, A):
        if not A: return 0
        def dfs(grid, x, y):
            dq = deque([[x,y]])
            while dq:
                i, j = dq.popleft()
                if i<0 or j<0 or i>=len(grid) or j>=len(grid[0]) or grid[i][j] != 1: continue
                grid[i][j] = 0
                for a,b in [(1,0),(-1,0),(0,1),(0,-1)]: dq.append([i+a, j+b])
        count = 0; grid = [list(map(int,list(s))) for s in A]
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    dfs(grid, i, j)
                    count += 1
        return count
    # https://www.interviewbit.com/problems/word-search-board/
    def existAllowReuse(self, A, B):
        moves = ((0, 1), (1, 0), (0, -1), (-1, 0))
        def dfs(A, i, j, word, pos):
            if A[i][j] != word[pos]: return False
            if pos == len(word) - 1:
                return True
            for move in moves:
                tx, ty = move
                if 0 <= i + tx < len(A) and 0 <= j + ty < len(A[i + tx]):
                    if dfs(A, i + tx, j + ty, word, pos + 1):
                        return True
            return False
        for i in range(len(A)):
            for j in range(len(A[0])):
                if A[i][j] == B[0]:
                    if dfs(A, i, j, B, 0): return 1
        return 0
    # https://leetcode.com/problems/word-search/discuss/747144/Python-dfs-backtracking-solution-explained Word Search Board
    def exist(self, A: List[str], word: str) -> bool:
        board = [list(s) for s in A]
        def dfs(ind, i, j): # word index, cell coordinates
            if self.Found: return        #early stop if word is found
            if ind == k:
                self.Found = True        #for early stopping
                return
            if i < 0 or i >= m or j < 0 or j >= n: return # went outside board
            tmp = board[i][j] # save board position value for backtrackking
            if tmp != word[ind]: return # return on failed match
            board[i][j] = "#"           # visited cell
            for x, y in [[0,-1], [0,1], [1,0], [-1,0]]:
                dfs(ind + 1, i+x, j+y)  # find match for next char
            board[i][j] = tmp           # backtracking
        self.Found = False; m, n, k = len(board), len(board[0]), len(word)
        for i, j in product(range(m), range(n)): # bruteforce search with backtracking
            if self.Found: return True           # early stop if word is found
            dfs(0, i, j)
        return self.Found
    # https://leetcode.com/problems/word-search-ii/discuss/59804/27-lines-uses-complex-numbers
    def findWords(self, board, words):
        root = {}
        for word in words: # add word to words dicts
            node = root
            for c in word:
                node = node.setdefault(c, {})
            node[''] = word # convert board from 2D to 1D using complex number key w/ imaginary unit 1j 
        board = {i + 1j*j: c
                 for i, row in enumerate(board)
                 for j, c in enumerate(row)}
        found = []
        def search(node, z):
            if '' in node: found.append(node.pop('')) # found word end, pop(None) to avoid finding this same word again
            c = board.get(z)
            if c in node: # c matched
                board[z] = None
                for k in range(4): # matching next char
                    search(node[c], z + 1j**k) 
                board[z] = c       # backtrack
        for z in board: search(root, z) # words, board position
        return found
    # leihao1313 crazy fast 32ms
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        pos = defaultdict(list)
        for i, r in enumerate(board):
            for j, c in enumerate(r):
                pos[c] += (i, j), # map char to coordinate
        m, n = len(board), len(board[0])
        seen = defaultdict(lambda:False) # default all seen coordinates to False
        def find(i, j, w):
            if not w: return True # matched all chars in words
            if i < 0 or j < 0 or i >= m or j >= n or seen[i,j] or board[i][j] != w[0]: return False
            seen[i,j] = True
            res = any(map(find,(i+1,i-1,i,i), (j,j,j+1,j-1), (w[1:],w[1:],w[1:],w[1:]))) # map find to its coord's and remaining chars
            seen[i,j] = False # backtrack
            return res
        return [w for w in words if any(find(i, j, w) for i, j in pos[w[0]])] # find each word at all positions pos[w[0]]
    def validPath(self, X: int, Y: int, N: int, R: int, A: List[int], B: List[int]) -> str:
        xDir = [0, 0, 1, 1, 1, -1, -1, -1] # 8 surrounding neighbors
        yDir = [1, -1, 0, 1, -1, 0, 1, -1]
        rect = [[0]*(Y+1) for _ in range(X+1)]
        for i in range(X+1):
            for j in range(Y+1):
                for k in range(N): # set all positions inside circles to 1
                    if math.sqrt(pow(A[k] - i, 2) + pow(B[k] - j, 2)) <= R:
                        rect[i][j] = 1
                        break
        if rect[0][0] == 1 or rect[X][Y] == 1: return "NO"
        dq = deque([[0, 0]]); rect[0][0] = 1
        while dq: # BFS starting from [0,0]
            x, y = dq.popleft()
            if x == X and y == Y: return "YES"
            for i in range(len(xDir)): # 8 [x,y]'s neighbors
                newx = x + xDir[i]; newy = y + yDir[i]
                if newx >= 0 and newx <= X and newy >= 0 and newy <= Y and rect[newx][newy] == 0:
                    rect[newx][newy] = 1; dq.append([newx, newy])
        return "NO"
    def smallestMultiple(self, N: int) -> str:
        if N < 2: return str(N)
        zeroend = ""
        while N%10 == 0: N//=10; zeroend += '0' 
        while N%5 == 0: N//=5; zeroend += '0' 
        while N%2 == 0: N//=2; zeroend += '0' 
        ord0 = ord('0')
        def mod(t, N):
            rem = 0
            for i in range(len(t)):
                rem = rem*10 + ord(t[i]) - ord0
                rem %= N
            return rem
        dq = deque(['1']); visited = set()
        while dq:
            t = dq.popleft(); state = mod(t,N)
            if state == 0: return t+zeroend
            if state not in visited:
                visited.add(state)
                dq.extend([t+'0', t+'1'])
    def smallestMultiple(self, N: int) -> str:
        zeroend = ""
        while N%10 == 0: N//=10; zeroend += '0' 
        while N%5 == 0: N//=5; zeroend += '0' 
        while N%2 == 0: N//=2; zeroend += '0'
        if N < 2: return str(N)+zeroend
        dq = deque([1]); visited = set()
        while dq:
            t = dq.popleft(); state = t%N
            if state == 0: return str(t)+zeroend
            if state not in visited:
                visited.add(state)
                dq.extend([t*10, t*10+1])
    def smallestMultiple(self, N: int) -> str:
        zeroend = ""
        while N%10 == 0: N//=10; zeroend += '0' 
        while N%5 == 0: N//=5; zeroend += '0' 
        while N%2 == 0: N//=2; zeroend += '0'
        if N < 2: return str(N)+zeroend
        p = [-1]*N # parent state
        s = [-1]*N # step from parent to current
        steps = [0,1] # BFS
        q = deque([1])
        while q:
            curr = q.popleft()
            if curr ==0:
                break;
            for step in steps:
                next = (curr*10+step)%N
                if p[next]==-1:
                    p[next]=curr
                    s[next]=step
                    q.append(next)
        number = "" # build reversed string
        it=0
        while it!=1:
            number += '1' if s[it] else '0'
            it = p[it]
        number += '1'
        return number[::-1]+zeroend
    #
    def capture0s(self, board: List[List[str]]) -> None:
        if not board: return
        m,n = len(board), len(board[0])
        dx = [0,0,1,-1]; dy = [-1,1,0,0]
        def freeCell(x,y):
            board[x][y] = '#'
            for i in range(4):
                nx,ny = x+dx[i],y+dy[i]
                if 0 <= nx <m and 0 <= ny <n and board[nx][ny] == 'O':
                    freeCell(nx,ny)
        for i in range(m):
            if i in (0,m-1):
                for j in range(n):
                    if board[i][j] == 'O':
                        freeCell(i,j)
            else:
                for j in range(n):
                    if board[i][j] == 'O':
                        if j in (0,n-1): freeCell(i,j)
        for i,r in enumerate(board):
            for j,c in enumerate(r):
                board[i][j] = 'XO'[c == '#']
        return board
    def capture0s(self, board: List[List[str]]) -> None: # iterative
        if not board: return
        n, m = len(board), len(board[0])
        boardFilter = lambda ij: 0 <= ij[1] < m and 0 <= ij[0] < n and board[ij[0]][ij[1]] == 'O'
        buffer = [x for i in range(max(n, m)) for x in ((i, 0), (i, m - 1), (0, i), (n - 1, i))] 
        queue = filter(boardFilter, buffer); x, y = next(queue, [-1,-1])
        while x >= 0:
            board[x][y] = 'M'
            buffer.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])
            x, y = next(queue, [-1,-1])
        # board[:] = [['XO'[x == 'M'] for x in row] for row in board]
        for i,row in enumerate(board):
            for j,c in enumerate(row):
                board[i][j] = 'XO'[c == 'M']
        return board
    def mst_prim(self, A, B):
        def create_adj_list(nodes_num, bridges):
            adj_list = [[] for _ in range(nodes_num + 1)]
            for source, dest, cost in bridges:
                adj_list[source] += [(dest, cost)]
                adj_list[dest] += [(source, cost)]
            return adj_list
        def total_cost_using_prim(adj_list, start_node):
            visited = set(); pq = [(0, start_node)]; total_cost = 0
            while len(pq) > 0:
                cost, cur_node = heappop(pq)
                if cur_node in visited: continue
                total_cost += cost # add shortest edge, nearest node from all levels
                for neighbor, neighbor_cost in adj_list[cur_node]:
                    heappush(pq, (neighbor_cost, neighbor)) # push start_node descendants into min heap 1 level at a time BFS, add shortest edge first 
                visited.add(cur_node) # visited after added to cost and pushed all neighbors
            return total_cost
        adj_list = create_adj_list(A, B)
        start_node = 1  ## doesn't matter actually, because all nodes should be connected so 1 must also be connected
        return total_cost_using_prim(adj_list, start_node)
    def mst_kruskal(self, A, B):
        B = sorted(B, key=lambda x: x[2]) # edges in increasing cost
        disjoint_set = list(range(A+1)) # node index pointing to itself initially, larger node indices pointing to smaller node indices
        def get_p(des, idx): # disjoint_set, node index
            while des[idx] != idx: # follow disjoint set till reaching leaf pointing to itself
                idx, des[idx] = des[idx], des[des[idx]]
            return idx
        cost = 0
        for u,v,c in B: # add shortest edge first
            p0, p1 = get_p(disjoint_set,u), get_p(disjoint_set,v)
            if p0 != p1: # different leaves, u,v are not yet connected, in 2 different disjoint_sets, add edge
                cost += c
                if p0 < p1: disjoint_set[p1] = p0 # join 2 disjoint sets via 2 leaves, larger leaf index pointing to smaller leaf index
                else: disjoint_set[p0] = p1
        return cost

sol = Solution()
if __name__ == '__main__':
    fptr = open(os.path.dirname(__file__) + "/mst.ut")
    A = int(fptr.readline().rstrip())
    B = json.loads(fptr.readline().rstrip())
    assert sol.mst_prim(A, B) == sol.mst_kruskal(A, B) == 7399
    assert sol.mst_kruskal(5, [[1,3,5],[4,5,7],[1,2,8],[2,3,9],[3,5,10],[2,4,11]]) == 30
    edgescost = [[1,2,1],[2,3,4],[1,4,3],[4,3,2],[1,3,10]]
    assert sol.mst_kruskal(4, edgescost) == 6
    assert sol.mst_kruskal(4, [[1,2,1],[2,3,2],[3,4,4],[1,4,3]]) == 6
    board = [["X","O","X","X"],["O","X","O","X"],["X","O","X","O"],["O","X","O","X"],["X","O","X","O"],["O","X","O","X"]]
    assert sol.capture0s(board) == [["X","O","X","X"],["O","X","X","X"],["X","X","X","O"],["O","X","X","X"],["X","X","X","O"],["O","X","O","X"]]
    board = [["O","X","X","O","X"],["X","O","O","X","O"],["X","O","X","O","X"],["O","X","O","O","O"],["X","X","O","X","O"]]
    assert sol.capture0s(board) == [["O","X","X","O","X"],["X","X","X","X","O"],["X","X","X","O","X"],["O","X","O","O","O"],["X","X","O","X","O"]]
    board = [["X","O","X"],["O","X","O"],["X","O","X"]]
    assert sol.capture0s(board) == [["X","O","X"],["O","X","O"],["X","O","X"]]
    board = [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]
    assert sol.capture0s(board) == [["X","X","X","X"],["X","X","X","X"],["X","X","X","X"],["X","O","X","X"]]
    board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]]
    words = ["oath","pea","eat","rain"]
    assert sol.findWords(board, words) == ["oath","eat"]
    board = [ "FEDCBECD", "FABBGACG", "CDEDGAEC", "BFFEGGBA", "FCEEAFDA", "AGFADEAC", "ADGDCBAA", "EAABDDFF" ]
    assert sol.existAllowReuse(board, "BCDCB") == True
    board = ["ABCE","SFCS","ADEE"]
    for word,ans in [("ABCCED",1),("SEE",1),("ABCB",0),("ABFSAD",1),("ABCD",0)]:
        assert sol.exist(board, word) == bool(ans)
    mat = ['11110','11010','11000','00000']
    assert sol.numIslands(mat) == 1
    assert sol.smallestMultiple(10) == '10'
    assert sol.smallestMultiple(17) == '11101'
    assert sol.smallestMultiple(11011) == '11011'
    assert sol.smallestMultiple(8) == '1000'
    assert sol.smallestMultiple(8) == '1000'
    assert sol.smallestMultiple(25) == '100'
    assert sol.smallestMultiple(20) == '100'
    assert sol.smallestMultiple(2) == '10'
    assert sol.smallestMultiple(55) == '110'
    assert sol.validPath(41, 67, 5, 0, [ 17, 16, 12, 0, 40 ], [ 52, 61, 61, 25, 31 ]) == "YES"
    assert sol.validPath(2, 3, 1, 1, [2], [3]) == "NO"
    fptr = open(os.path.dirname(__file__) + "/bfs.ut", "r")
    # fptr = open(os.path.dirname(__file__) + "/bfs.uto", "r")
    t = int(fptr.readline())
    for i in range(t):
        n,m = [int(value) for value in fptr.readline().split()]
        graph = Graph(n)
        # for i in range(m):
        #     x,y = [int(x) for x in fptr.readline().split()]
        #     graph.connect(x,y) 
        # s = int(fptr.readline())
        # ans = graph.find_all_distances(s)
        # expected = [int(value) for value in fptr.readline().split()]
        # print([(a,e) for a,e in zip(ans,expected) if a != e])
        # assert(ans == expected)
    fptr = open(os.path.dirname(__file__) + "/matrices.ut", "r")
    t = int(fptr.readline())
    for i in range(t):
        n = int(fptr.readline().rstrip())
        m = int(fptr.readline().rstrip())
        grid = []
        for _ in range(n):
            grid.append(list(map(int, fptr.readline().rstrip().split())))
        g = createGraph(grid)
        print(g.findMaxRegion(),g.findIslands(),sol.numIslands(grid))
