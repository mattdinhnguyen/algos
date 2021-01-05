import copy
from collections import deque, defaultdict

class BTNode:

    def __init__(self, value, left = None, right= None):
        self.value = value
        self.treeVal = 0
        self.left = left
        self.right = right
'''Given a binary tree, you can pick any number of nodes from it with the criteria that no two nodes can be directly related i.e be in a parent-child relationship.
What is the maximum sum of nodes that can be obtained by a set of nodes that satisfy the above criteria?

                1
          2        30
       4    5   6   7
   50     8

(30, 5, 50)
One possible subset is (1, 4, 5, 6, 7) and another is (2, 30,50). We should output the first set to get the max sum.

[1] => 1
[2] => 2
[1,2] => infeasible
[1,4,5] => 10
[1,4,5,6,7] => 23 => answer
'''
def treeLevelAvg(node):
    que = deque([(0,node)])
    levelVals = defaultdict(list)
    while que:
        level, n = que.popleft()
        que.extend([(level+1,c) for c in [n.left,n.right] if c])
        levelVals[level].append(n.value)
    return [ sum(vals)/len(vals) for level,vals in levelVals.items() ]

def maxSumNonRelatedNodes(root):
    nodes = deque()
    stack = deque([root])
    while stack: # BFS N
        cur = stack.pop()
        nodes.append(cur)
        stack.extend([c for c in (cur.left,cur.right) if c])
    stack = copy.copy(nodes)
    while stack: # treeVal includes grand children or great grand children values N
        n = stack.pop() # pop from leaves
        n.treeVal = n.value
        for c in (n.left, n.right):
            if c:
                for gc in (c.left,c.right):
                    if gc:
                        n.treeVal += max(gc.treeVal,sum([ggc.treeVal for ggc in (gc.left,gc.right) if ggc]))
    print([(n.value,n.treeVal) for n in nodes])
    stack = copy.copy(nodes)
    while stack:
        n = stack.popleft() # pop from root
        if n.left and n.right: # treeVal includes peer or peer's children treeVal
            temp = max(n.left.treeVal,sum([c.treeVal for c in (n.left.left,n.left.right) if c]))
            n.left.treeVal += max(n.right.treeVal,sum([c.treeVal for c in (n.right.left,n.right.right) if c]))
            n.right.treeVal += temp
    print([(n.value,n.treeVal) for n in nodes])
    return max([(n.treeVal) for n in nodes])
'''Given a binary tree of integers, determine the sum of all numbers formed by the paths from the root to each leaf node
  5 
1   2  
   3  4
 51
523
524
---
1098 <--- result we want to return

Input: reference to root node of tree
Output: single number, the sum of the numbers that are formed by paths from root to each leaf

'''
def treeSum(node):
    stax = [(node,node.value)]
    numbers = []
    while stax:
        n, val = stax.pop()
        if any((n.left,n.right)):
            stax.extend([(c,val*10+c.value) for c in (n.left,n.right) if c])
        else:
            numbers.append(val)
    return sum(numbers)
 

root = BTNode(1,
        BTNode(2,
            BTNode(4,
                BTNode(50)),
            BTNode(5,
                BTNode(8))),
        BTNode(30,
            BTNode(6),
            BTNode(7))
        )
root1 = BTNode(5,
        BTNode(1),
        BTNode(2,
            BTNode(3),
            BTNode(4))
        )
root2 = BTNode(3,
        BTNode(9),
        BTNode(20,
            BTNode(15),
            BTNode(7))
        )

if __name__ == '__main__':
    # print(maxSumNonRelatedNodes(root))
    print(treeSum(root1))
    # print(treeLevelAvg(root2))
