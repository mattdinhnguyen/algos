#!/usr/local/bin/python3.7
# import random
# random.seed(0)
# lst = [0]
# lst += [random.randint(1, 10) for _ in range(10)]
# print(lst)
def parent(n):
    return int((n-1)/2)
def lchild(n):
    return 2*n+1
def rchild(n):
    return 2*n+2
def lchildV(l, n):
    return l[lchild(n)]
def rchildV(l, n):
    return l[rchild(n)]
def parentV(l, n):
    return l[parent(n)]

def isBST(tree, node, minV=0, maxV=0):
    def inRange():
        if minV and maxV:
            return minV <= tree[node] <= maxV
        elif minV:
            return tree[node] >= minV
        elif maxV:
            return tree[node] <= maxV
        else:
            return True
    if lchild(node) < len(tree) and rchild(node) < len(tree):
        print(lchildV(lst, node), lst[node], rchildV(lst, node))
        return inRange() and \
            lchildV(tree, node) <= tree[node] and tree[node] <= rchildV(tree, node) and \
            isBST(tree, lchild(node), maxV=tree[node]) and isBST(tree, rchild(node), minV=tree[node])
    else:
        return True

lst = [5, 5, 7, 4, 9, 8, 7, 3, 8, 6, 9]
print(isBST(lst, 1))
print(parent(1))
print(parentV(lst,1))