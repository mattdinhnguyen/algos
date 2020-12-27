# a) Each node when it enters has its "value" (Number of people greater in height) as its initial value. I am calling it the "current node" (till it reaches its final position).
# b) A "current node" goes "left" to the existing node, when its value is <= the existing node's value. At that time it increments the existing node's value by 1.
# When the "current node" goes "right", no change to any values.

# Repeat for all nodes. Nodes to be pushed in decreasing order of height. Finally do in-order traversal.
# The changes from the previous solution are no "right" rule & no +1 for the current node.
# It is imperative to insert the nodes in decreasing order
# With this, the following will be the iterations...
# a) 6 = 0
# b) 5 = 0, 6 = 1
# c) 4 = 0, 5 = 1, 6 = 2.
# d) 4 = 0, 5 = 1, 3 = 2, 6 = 3.
# e) 4 = 0, 5 = 1, 2 = 2, 3 = 3, 6 = 4.
# f) 4 = 0, 5 = 1, 2 = 2, 3 = 3, 1 = 4, 6 = 5.

# The "intuition" is - In each iteration, an element is inserted at the same position as its "value".


class Node:

    def __init__(self, data = (None,None)):

        self.left = None
        self.right = None
        self.id = data[0]
        self.value = data[1]

    def insert(self, data):

        if self.value != None:
            if data[1] <= self.value:
                if self.left is None:
                    self.left = Node(data)
                else:
                    self.left.insert(data)
                self.value += 1
            else:
                if self.right is None:
                    self.right = Node(data)
                else:
                    self.right.insert(data)
        else:
            self.id = data[0]
            self.value = data[1]

    def PrintTree(self):
        if self.left:
            self.left.PrintTree()
        print((self.id,self.value)),
        if self.right:
            self.right.PrintTree()

A = zip([6, 5, 4, 3, 2, 1],[0, 0, 0, 2, 2, 4])

root = Node()
for n in A:
    root.insert(n)

root.PrintTree()
