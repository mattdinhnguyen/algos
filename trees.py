import sys

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
            values.append(node.tree.value)
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

class Node:
    def __init__(self, value, left = None, right= None):
        self.value = value
        self.left = left
        self.right = right
        self.level = None

    def __str__(self):
        return str(self.value) 

    def contains(self, value, exclude):
        if (value < self.value):
            return self.left and self.left.contains(value, exclude)
        elif (value > self.value):
            return self.right and self.right.contains(value, exclude)
        else:
            return False if self == exclude else True

def preOrder(root):
    if root:
        left = preOrder(root.left) if root.left else []
        right = preOrder(root.right) if root.right else []
    return [root.value] + left + right if root else []

def inOrder(root):
    if root:
        left = inOrder(root.left) if root.left else []
        right = inOrder(root.right) if root.right else []
        left.append(root.value)
    return left + right if root else []

def checkBST(root, min = -sys.maxsize, max = sys.maxsize):
    ans = True
    if root:
        if max <= root.value or root.value <= min:
            ans = False
        elif root.left:
            ans = checkBST(root.left, min, root.value) if root.left.value < root.value else False
        if ans and root.right:
            ans = checkBST(root.right, root.value, max) if root.right.value > root.value else False
    return ans

class BinarySearchTree:
    def __init__(self, a = []):
        self.root = None
        for val in a: self.insert(val)
    def insert(self, val):
        def _insert(node, val):
            if val <= node.value:
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
        return str(list(map(lambda n: n.value, bfs)))
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

if __name__ == "__main__":
    tree = BinarySearchTree()
    # arr = [4, 2, 3, 1, 7, 6]
    # arr = [3, 2, 5, 1, 4, 6, 7]
    arr = [100,50,200,25,75,350]
    for i in range(len(arr)):
        tree.insert(arr[i])
    print(inOrder(tree.root),preOrder(tree.root))
    print(height(tree.root),checkBST(tree.root), tree)
    print(convertBT2DLL(tree.root))
