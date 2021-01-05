from trees import Node as BSTNode
from trees import Stack

'''
    We perform a depth-first search on the BST using a stack which size won't
    grow over O(height of the tree).
    For each node we find, we check in logarithmic time whether the complement is
    in the tree (taking care of excluding this node).
'''

def findNodesWithSum(root, sum):
    stack = Stack()
    stack.push(root)
    curTree = stack.pop()
    while (curTree):
        sValue = sum - curTree.value
        if (root.contains(sValue, curTree)):
            return (curTree.value, sum - curTree.value)
        if (sValue > curTree.value and curTree.right):
            stack.push(curTree.right)
        if (sValue < curTree.value and curTree.left):
            stack.push(curTree.left)
        curTree = stack.pop()
    return (None,None)

def findNodesWithSum(root, sum):
    stack = [root]
    while (stack):
        curTree = stack.pop()
        sValue = sum - curTree.value
        if (root.contains(sValue, curTree)):
            return (curTree.value, sum - curTree.value)
        if (sValue > curTree.value and curTree.right):
            stack.append(curTree.right)
        if (sValue < curTree.value and curTree.left):
            stack.append(curTree.left)
    return (None,None)

'''
                    13
            5               15
        3       8       14      17
'''
root = BSTNode(13,
        BSTNode(5,
            BSTNode(3),
            BSTNode(8)),
        BSTNode(15,
            BSTNode(14),
            BSTNode(17))
        )
for i in [10,9,13,20]:
    print(i, findNodesWithSum(root, i))
