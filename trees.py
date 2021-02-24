from os import ftruncate
import sys
from typing import List
from collections import defaultdict, deque
class LLNode:
    def __init__(self, tree, next):
        self.tree = tree
        self.next = next

class DLLNode:
    def __init__(self, tree, prev, next):
        self.tree = tree
        self.prev = prev
        self.next = next

class DLList:
    def __init__(self):
        self.head = self.tail = None
    def append(self, tree):
        newNode = DLLNode(tree,self.tail,None)
        if not self.head:
            self.head = newNode
        else:
            self.tail.next = newNode
        self.tail = newNode
    def prepend(self, tree):
        self.head = DLLNode(tree,None,self.head)
    def __str__(self):
        node = self.head
        values = []
        while node:
            values.append(node.tree.val)
            node = node.next
        return str(values)
class Stack:
    def __init__(self):
        self.head = None

    def push(self, tree):
        self.head = LLNode(tree, self.head)

    def pop(self):
        oldHead = self.head
        if (oldHead is None):
            return None
        self.head = oldHead.next
        return oldHead.tree

class NaryNode:
    def __init__(self, value):
        self.val = value
        self.children = []
        self.level = None
class Node:
    def __init__(self, value, left = None, right= None):
        self.val = value
        self.left = left
        self.right = right
        self.level = None

    def __str__(self):
        return str(self.val) 

    def contains(self, value, exclude):
        if (value < self.val):
            return self.left and self.left.contains(value, exclude)
        elif (value > self.val):
            return self.right and self.right.contains(value, exclude)
        else:
            return False if self == exclude else True

def preOrder(root):
    left, right = [], []
    if root:
        left = preOrder(root.left) if root.left else []
        right = preOrder(root.right) if root.right else []
    return [root.val] + left + right if root else []

def inOrder(root):
    left, right = [], []
    if root:
        left = inOrder(root.left) if root.left else []
        right = inOrder(root.right) if root.right else []
        left.append(root.val)
    return left + right if root else []
def inOrder(root):
    stack = []; ans = []
    while root or stack:
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop()
        ans.append(root.val)
        root = root.right
    return ans
# https://www.cnblogs.com/AnnieKim/archive/2013/06/15/morristraversal.html
def inOrder(root): # Morris Traversal
    res = []; cur = root; prev = None
    while cur:
        if cur.left:
            prev = cur.left
            while prev.right and prev.right != cur:
                prev = prev.right # find the right-most leaf of cur.left
            if prev.right == cur: # threading already exists
                res.append(cur.val)
                cur = cur.right
                prev.right = None
            else: # right-most leaf points to cur
                prev.right = cur
                cur = cur.left
        else:
            res.append(cur.val)
            cur = cur.right
    return res
def checkBST(root, min = -sys.maxsize, max = sys.maxsize):
    ans = True
    if root:
        if max <= root.val or root.val <= min:
            ans = False
        elif root.left:
            ans = checkBST(root.left, min, root.val) if root.left.val < root.val else False
        if ans and root.right:
            ans = checkBST(root.right, root.val, max) if root.right.val > root.val else False
    return ans

class NaryTree:
    def __init__(self, a = []):
        self.root = None
        self.m = dict()
        for i in range(len(a)):
            if a[i] == -1:
                self.root = self.m[i] = NaryNode(i)
            else:
                self.insert(a[i], i)

    def insert(self, val: int, neighbor: int) -> None:
        a = self.m.setdefault(val, NaryNode(val))
        b = self.m.setdefault(neighbor, NaryNode(neighbor))
        a.children.append(b)

class BinarySearchTree:
    def __init__(self, a = []):
        self.root = None
        for val in a: self.insert(val)
    def insert(self, val):
        def _insert(node, val):
            if val <= node.val:
                if node.left: _insert(node.left, val)
                else: node.left = Node(val)
            else:
                if node.right: _insert(node.right, val)
                else: node.right = Node(val)
        if self.root: _insert(self.root, val)
        else: self.root = Node(val)
        return self.root
    def __str__(self):
        bfs = [self.root]
        i = 0
        while i < len(bfs):
            bfs += [c for c in [bfs[i].left,bfs[i].right] if c]
            i += 1
        return str(list(map(lambda n: n.val, bfs)))
def height(root):
    if not root: return -1
    def h(node,level):
        print(f"node {node} level {level}")
        node.level = level
        cs = [c for c in [node.left,node.right] if c]
        return max([h(c, level+1) for c in cs]) if cs else level
    return h(root,0)
def convertBT2DLL(node):
    def _inOrder(node):
        left = _inOrder(node.left) if node.left else []
        right = _inOrder(node.right) if node.right else []
        left.append(node)
        return left + right if node else []
    dll = DLList()
    for t in _inOrder(node):
        dll.append(t)
    return dll
class Solution:
    def diameterOfNaryTree(self, root: NaryNode) -> int:
        self.ans = 0 # Return the global maximum
        def dfs(root):
            max1 = 0 # The longest path from the leaf node under the current node
            max2 = 0 # The second longest path to the leaf node under the current node
            for child in root.children:
                # Recursively view the longest path from the current child node to the leaf node
                # Add one to the distance from the current node to the child node
                length = dfs(child) + 1
                # Update the longest path that is the second longest path
                if length >= max1: max2, max1 = max1, length
                elif length >= max2: max2 = length
            # The sum of the two longest paths is the longest path through the current node
            # Update the global maximum length with this length
            self.ans = max(self.ans, max1+max2)
            return max1 # Return the maximum path under the current node
        if root.children:
            dfs(root)
        return self.ans
    def diameterOfBinaryTree(self, root: Node) -> int:
        self.ans = 0
        def depth(p):
            if not p: return 0
            left, right = depth(p.left), depth(p.right)
            self.ans = max(self.ans, left+right)
            return 1 + max(left, right)
        depth(root)
        return self.ans
    def diameterOfNaryTree(self, A: List[int]) -> int: # recursive
        max_ans = 0; n = len(A)
        if n<2: return 0
        def dfs(src, tree):
            nonlocal max_ans
            max1 = max2 = 0 # 2 longest paths from src to its leaves
            kids = len(tree[src]) # no of childs of src
            if kids == 0: return 1 # this leaf will contribute 1 edge when linked with its parent
            for i in range(kids):
                ans = dfs(tree[src][i], tree)
                if ans > max1: max2, max1 = max1, ans
                elif ans > max2: max2 = ans
            max_ans = max(max_ans, max1+max2)
            return 1+max(max1,max2) # +1 edge from src to furthest leaf
        root = -1 # nodes will be 0 to n-1
        tree = defaultdict(list)
        for i in range(n):
            if A[i] == -1: root = i # root has indegree 0
            tree[A[i]].append(i) # src -> destinations
        dfs(root,tree)
        return max_ans
    def diameterOfNaryTree(self, A: List[int]) -> int:
        n = len(A)
        if n<2: return 0
        tree = defaultdict(list); visited = [0]*n
        for i in range(n): # build unidirected graph
            if A[i] != -1:
                tree[A[i]].append(i) # src -> destinations
                tree[i].append(A[i])
        dq = deque([(0,0)]); visited[0] = 1 # root has distance 0
        def bfs(q, v, t): # node/distance queue, visited, node/kids map: return 1 leaf of the longest distance 
            maxi = -sys.maxsize; farthest = 0
            while q: # BFS traversal from root to all leaves to find max distance 
                node, distance = q.popleft()
                if distance > maxi: # find farthest/distance (max) till having explored all leaves
                    maxi = distance; farthest = node
                for c in t[node]: # explore children
                    if not visited[c]:
                        visited[c] = 1; q.append((c,distance+1)) # distance+1 for each kid level 
            return maxi,farthest # farthest is 1 leaf of the longest distance maxi
        furthest = bfs(dq, visited, tree)[-1]
        visited = [0]*n
        dq.append((furthest,0)); visited[furthest] = 1
        return bfs(dq, visited, tree)[0]
sol = Solution()
if __name__ == "__main__":
    assert sol.diameterOfNaryTree([-1, 0, 0, 0, 3]) == 3
    # tree = NaryTree([-1, 0, 0, 0, 3])
    # assert sol.diameterOfNaryTree(tree.root) == 3
    # tree = NaryTree([1,None,2,3,4,5,None,None,6,7,None,8,None,9,10,None,None,11,None,12,None,13,None ,None,14])
    # tree = BinarySearchTree()
    # arr = [4, 2, 3, 1, 7, 6]
    # arr = [3, 2, 5, 1, 4, 6, 7]
    # arr = [100,50,200,25,75,350]
    # arr = [6,2,7,1,4,9,3,5,8]
    # for i in range(len(arr)):
    #     tree.insert(arr[i])
    # assert sol.diameterOfBinaryTree(tree.root) == 6
    # inOrd = inOrder(tree.root)
    # print(inOrd,preOrder(tree.root))
    # print(height(tree.root),checkBST(tree.root), tree)
    # print(convertBT2DLL(tree.root))
