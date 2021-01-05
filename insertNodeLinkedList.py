import math
import os
import random
import re
import sys


class SinglyLinkedListNode:
    def __init__(self, v):
        self.data = v
        self.next = None

class SinglyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def insert_node(self, node_data):
        node = SinglyLinkedListNode(node_data)

        if not self.head:
            self.head = node
        else:
            self.tail.next = node


        self.tail = node

def print_singly_linked_list(node):
    nodes = []
    while node:
        nodes.append(node.data)
        node = node.next
    if nodes:
        print(nodes)

def insertNodeAtPosition(head, data, position):
    if position == 0:
        newHead = SinglyLinkedListNode(data)
        newHead.next = head.next
        head = newHead
    else:
        curNode = head
        position -= 1
        while position > 0 and curNode.next != None:
            curNode = curNode.next
            position -= 1
        newNode = SinglyLinkedListNode(data)
        newNode.next = curNode.next
        curNode.next = newNode

    return head
'''
step 0: A->B->C->D->E
step 1: A->B->C<-D<-E
step 2: A->E, B->C<-D
step 3: A->E->B->D, C
step 4: A->E->B->D->C
'''
# Splits in place a list in two halves, the first half is >= in size than the second.
# @return A tuple containing the heads of the two halves
def _splitList(head):
    fast = head
    slow = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    middle = slow.next
    slow.next = None

    return head, middle

# Reverses in place a list.
# @return Returns the head of the new reversed list
def _reverseList(head):

  last = None
  currentNode = head

  while currentNode:
    nextNode = currentNode.next
    currentNode.next = last
    last = currentNode
    currentNode = nextNode

  return last

# Merges in place two lists
# @return The newly merged list.
def _mergeLists(a, b):

    tail = a
    head = a

    a = a.next
    while b:
        tail.next = b
        tail = tail.next
        b = b.next
        if a:
            a, b = b, a
            
    return head


class Solution:

    # @param head, a ListNode
    # @return nothing
    @staticmethod
    def reorderList(head):

        if not head or not head.next:
            return

        a, b = _splitList(head)
        b = _reverseList(b)
        head = _mergeLists(a, b)

if __name__ == '__main__':
    fptr = open("insertNodeLinkedList.ut")
    data = map(int,fptr.readline().strip().split(','))
    llist = SinglyLinkedList()
    for val in data:
    # for val in ['A','B','C','D','E']:
        llist.insert_node(val)
    llist.insert_node('F') #case 2
    print_singly_linked_list(llist.head)
    # llist_head = insertNodeAtPosition(llist.head, data, position)
    r = Solution.reorderList(llist.head)
    print_singly_linked_list(r)
