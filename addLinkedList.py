from insertNodeLinkedList import _reverseList as reverse_list

class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

def gen_list(num):
    last = Node(num[0])
    start = last
    for digit in num[1:]:
        node = Node(digit)
        last.next = node
        last = node
    return start

def print_list(num):
    if num is None:
        print('')
        return
    string = '{}'.format(num.value)
    num = num.next
    while num is not None:
        string += ' -> {}'.format(num.value)
        num = num.next
    print(string)

def reverse_list0(fwd):
    last = None
    while fwd is not None:
        bwd = Node(fwd.value)
        if last is not None:
            bwd.next = last
        last = bwd
        fwd = fwd.next
    return last

def sum_reverse_list(node1, node2):
    curr1 = node1
    curr2 = node2
    carry = 0
    while curr1 and curr2:
        sum2 = curr1.value + curr2.value + carry
        curr1.value = sum2%10
        carry = sum2//10
        if not all([curr1.next,curr2.next]):
            break
        curr1 = curr1.next
        curr2 = curr2.next
    if any([curr1.next,curr2.next]):
        if curr2.next:
            curr1.next = curr2.next
            curr2.next = None
            curr1 = curr1.next
        while curr1:
            sum2 = curr1.value + carry
            if sum2 < 10:
                curr1.value = sum2
                break
            curr1.value = sum2%10
            carry = sum2//10
            if curr1.next == None:
                break
            curr1 = curr1.next
    if carry:
        curr1.next = Node(carry)
    return reverse_list(node1)
def sum_reverse_list0(node1, node2):
    c = 0
    node = None
    prev = None
    while node1 is not None or\
            node2 is not None:
        a = node1.value if node1 is not None else 0
        b = node2.value if node2 is not None else 0
        x = a + b + c
        if x > 9:
            node = Node(x%10)
            c = 1
        else:
            node = Node(x)
            c = 0
        node.next = prev
        prev = node
        if node1 is not None:
            node1 = node1.next
        if node2 is not None:
            node2 = node2.next
    if c == 1:
        node = Node(1)
        node.next = prev
    return node

def sum_list(node1, node2):
    node1 = reverse_list(node1)
    node2 = reverse_list(node2)
    return sum_reverse_list(node1, node2)

if __name__ == '__main__':
    num1 = [9, 9, 9]
    node1 = gen_list(num1)
    print_list(node1)
    print('+')
    num2 = [1, 2, 3]
    node2 = gen_list(num2)
    print_list(node2)
    print('-----')
    print_list(sum_list(node1, node2))
