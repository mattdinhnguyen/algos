import math
import os
import random
import re
import sys
# from functools import reduce
from typing import List
from heapq import heapify, heappop, heappush, heappushpop
from collections import deque, defaultdict
import queue
from visibleNodes import TreeNode
from trees import inOrder
class SinglyLinkedListNode:
    def __init__(self, v, next = None):
        self.val = v
        self.next = None
class SinglyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def insert_node(self, node_data): # at tail
        node = SinglyLinkedListNode(node_data)
        if not self.head: self.head = node
        else: self.tail.next = node
        self.tail = node

def print_singly_linked_list(node):
    nodes = []
    while node:
        nodes.append(node.val)
        node = node.next
    if nodes:
        print(nodes)

def insertNodeAtPosition(head, data, position):
    if position == 0:
        newHead = SinglyLinkedListNode(data)
        newHead.next = head.next
        head = newHead
    else:
        curNode = head # is position 1
        position -= 1
        while position > 0 and curNode.next != None:
            curNode = curNode.next
            position -= 1
        newNode = SinglyLinkedListNode(data)
        newNode.next = curNode.next
        curNode.next = newNode # after position

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
    tail = head = a; a = a.next
    while b:
        tail.next = b
        tail = tail.next
        b = b.next
        if a: a, b = b, a        
    return head
class Solution:

    # @param head, a ListNode
    # @return nothing
    @staticmethod
    def reorderList(head):
        if not head or not head.next: return
        a, b = _splitList(head)
        b = _reverseList(b)
        head = _mergeLists(a, b)
    @staticmethod
    def mergeKLists(lists: List[SinglyLinkedListNode]) -> SinglyLinkedListNode:
        hq = []; curheads = lists[:]
        for i,n in enumerate(curheads): hq.append((n.val,i,n))
        heapify(hq); n = heappop(hq); head = tail = n[-1]; heappush(hq, (tail.next.val, n[1], tail.next))
        while hq:
            n = heappop(hq)
            tail.next = n[-1]; tail = tail.next
            if tail.next:
                heappush(hq, (tail.next.val, n[1], tail.next))
        return head
    def sortedListToBST(self, head: SinglyLinkedListNode) -> TreeNode:
        if not head: return
        if not head.next: return TreeNode(head.val)
        slow, fast, pre = head, head, None
        while fast and fast.next:
            pre, slow, fast = slow, slow.next, fast.next.next
        pre.next = None   # cut off the left half
        root = TreeNode(slow.val)
        root.left = self.sortedListToBST(head)
        root.right = self.sortedListToBST(slow.next)
        return root
    def knight(self, X, Y, x1, y1, x2, y2):
        if [x1,y1] == [x2,y2]: return 0
        x = [1, 2, 2, 1, -1, -2, -1, -2]
        y = [2, 1, -1, -2, 2, 1, -2, -1]
        moves = 1; dq = deque([[x1,y1]]); nq = deque()
        visited = [[False]*Y for _ in range(X)]; visited[x1-1][y1-1] = True
        while dq:
            while dq:
                cx,cy = dq.popleft();
                for i in range(len(x)):
                    nx = cx+x[i]; ny = cy+y[i]
                    if nx > 0 and nx <= X and ny > 0 and ny <= Y and not visited[nx-1][ny-1]:
                        if [nx,ny] == [x2,y2]: return moves
                        nq.append([nx,ny]); visited[nx-1][ny-1] = True
            moves += 1
            dq, nq = nq, dq
        return -1
    # https://zhenyu0519.github.io/2020/03/24/lc127/
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        wordset = set(wordList)
        queue = deque()
        queue.append((beginWord, 1))
        word_length = len(beginWord)
        while queue:
            word, step = queue.popleft()
            if word == endWord: return step
            for i in range(word_length):
                for c in "abcdefghijklmnopqrstuvwxyz":
                    newWord = word[:i]+c+word[i+1:]
                    if newWord in wordset:
                        wordset.remove(newWord)
                        queue.append((newWord, step+1))
        return 0
    # ELind77 https://leetcode.com/problems/word-ladder/discuss/40723/Simple-to-understand-Python-solution-using-list-preprocessing-and-BFS-beats-95
    def ladderLength(self, begin: str, end: str, wordList: List[str]) -> int:
        # Creates a map of all combinations of words with missing letters mapped to all words in the list that match that pattern.
        p2w = defaultdict(list) # hot -> {'_ot': ['hot'], 'h_t': ['hot'], 'ho_': ['hot']}
        for word in wordList:
            for i in range(len(word)):
                p = word[:i] + "_" + word[i+1:]
                p2w[p].append(word)
        queue = deque([(begin, 1)]); visited = set([begin])
        while queue:
            word, depth = queue.popleft()
            for i in range(len(word)): # Get the node's children by recreating all possible patterns for that string
                p = word[:i] + "_" + word[i+1:]
                neighbor_words = p2w[p]
                for nw in neighbor_words: # Iterate through children
                    if nw not in visited:
                        if nw == end: return depth+1 # Goal check (before adding to the queue)
                        visited.add(nw)
                        queue.append((nw, depth+1))
        return 0
    def ladderLength(self, begin, end, words): # best time, concise
        if not begin or not end or end not in words: return 0
        graph = defaultdict(list)
        for i in range(len(begin)):
            for word in words:
                graph[word[:i] + '?' + word[i + 1:]].append(word)
        queue = deque([(begin, 1)]); visited = {begin}
        while queue:
            current, level = queue.popleft()
            for i in range(len(current)):
                intermediate = current[:i] + '?' + current[i + 1:]
                for word in graph[intermediate]:
                    if word == end: return level + 1
                    if not word in visited:
                        visited.add(word)
                        queue.append((word, level + 1))
                del graph[intermediate]
        return 0
    # https://leetcode.com/problems/word-ladder-ii/discuss/40482/Python-simple-BFS-layer-by-layer
    def findLadders(self, beginWord, endWord, wordList):
        wordList = set(wordList); res = []; layer = {beginWord:[[beginWord]]} # first layer word/paths: starting at begin, ending at word
        graph = defaultdict(list)
        for i in range(len(beginWord)):
            for word in wordList: # hot -> {'_ot': ['hot'], 'h_t': ['hot'], 'ho_': ['hot']}
                graph[word[:i] + '?' + word[i + 1:]].append(word)
        while layer:
            newlayer = defaultdict(list) # build new (next) layer
            for w in layer:
                if w == endWord: res.extend(layer[w]) # found endWord
                else:
                    for i in range(len(w)):
                        for neww in graph[w[:i]+'?'+w[i+1:]]: # new words in next layer
                            if neww in wordList: # updated wordList does not have found words already in paths
                                newlayer[neww]+=[j+[neww] for j in layer[w]] # key/value: new word/paths 'ted': [['red','ted]]
            wordList -= set(newlayer.keys()) # remove found (new) words already in paths
            layer = newlayer
        return res
    def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        tree, words, n = defaultdict(set), set(wordList), len(beginWord) # tree is a BFS trace for each word considered in words.
        if endWord not in wordList: return []
        res, found, q, nq = [], False, {beginWord}, set() # BFS uses q(queue), and nq(next queue) stores words in next level.
        graph = defaultdict(list)
        for i in range(len(beginWord)):
            for word in wordList:
                graph[word[:i] + '?' + word[i + 1:]].append(word)
        while q and not found: # stop when found, or no word left
            words -= set(q) # subtract words in current level so that they won't be used again.
            for x in q: # for each word in current level
                for i in range(n):
                    for y in graph[x[:i]+'?'+x[i+1:]]:
                        if y in words: # only care those in words set
                            if y == endWord: # if found (reach the shortest solution level), we won't do next level.
                                found = True
                            else: # if not found
                                nq.add(y) # prepare next queue
                            tree[x].add(y) # add trace
            q, nq = nq, set() # reset while loop for next level
        def bt(x, tmp): # x is the last element in tmp
            if x == endWord: # if a shallow copy of tmp is not created, its mutable and the state changes so result would be incorrect
                res.append(tmp[:])  # same =>  res.append(list(tmp))
            else:
                for y in tree[x]: # explore a new path to target via each of x neighbors 
                    bt(y, tmp + [y]) # same => tmp.append(y); bt(y, tmp); tmp.pop()
        bt(beginWord, [beginWord])
        return res
        # def bt(x): # backtracking (DFS) for solution
        #     return [[x]] if x == endWord else [[x] + rest for y in tree[x] for rest in bt(y)]
        # return bt(beginWord)
sol = Solution()
if __name__ == '__main__':
    a,z,d = "cet","ism",["kid","tag","pup","ail","tun","woo","erg","luz","brr","gay","sip","kay","per","val","mes","ohs","now","boa","cet","pal","bar","die","war","hay","eco","pub","lob","rue","fry","lit","rex","jan","cot","bid","ali","pay","col","gum","ger","row","won","dan","rum","fad","tut","sag","yip","sui","ark","has","zip","fez","own","ump","dis","ads","max","jaw","out","btu","ana","gap","cry","led","abe","box","ore","pig","fie","toy","fat","cal","lie","noh","sew","ono","tam","flu","mgm","ply","awe","pry","tit","tie","yet","too","tax","jim","san","pan","map","ski","ova","wed","non","wac","nut","why","bye","lye","oct","old","fin","feb","chi","sap","owl","log","tod","dot","bow","fob","for","joe","ivy","fan","age","fax","hip","jib","mel","hus","sob","ifs","tab","ara","dab","jag","jar","arm","lot","tom","sax","tex","yum","pei","wen","wry","ire","irk","far","mew","wit","doe","gas","rte","ian","pot","ask","wag","hag","amy","nag","ron","soy","gin","don","tug","fay","vic","boo","nam","ave","buy","sop","but","orb","fen","paw","his","sub","bob","yea","oft","inn","rod","yam","pew","web","hod","hun","gyp","wei","wis","rob","gad","pie","mon","dog","bib","rub","ere","dig","era","cat","fox","bee","mod","day","apr","vie","nev","jam","pam","new","aye","ani","and","ibm","yap","can","pyx","tar","kin","fog","hum","pip","cup","dye","lyx","jog","nun","par","wan","fey","bus","oak","bad","ats","set","qom","vat","eat","pus","rev","axe","ion","six","ila","lao","mom","mas","pro","few","opt","poe","art","ash","oar","cap","lop","may","shy","rid","bat","sum","rim","fee","bmw","sky","maj","hue","thy","ava","rap","den","fla","auk","cox","ibo","hey","saw","vim","sec","ltd","you","its","tat","dew","eva","tog","ram","let","see","zit","maw","nix","ate","gig","rep","owe","ind","hog","eve","sam","zoo","any","dow","cod","bed","vet","ham","sis","hex","via","fir","nod","mao","aug","mum","hoe","bah","hal","keg","hew","zed","tow","gog","ass","dem","who","bet","gos","son","ear","spy","kit","boy","due","sen","oaf","mix","hep","fur","ada","bin","nil","mia","ewe","hit","fix","sad","rib","eye","hop","haw","wax","mid","tad","ken","wad","rye","pap","bog","gut","ito","woe","our","ado","sin","mad","ray","hon","roy","dip","hen","iva","lug","asp","hui","yak","bay","poi","yep","bun","try","lad","elm","nat","wyo","gym","dug","toe","dee","wig","sly","rip","geo","cog","pas","zen","odd","nan","lay","pod","fit","hem","joy","bum","rio","yon","dec","leg","put","sue","dim","pet","yaw","nub","bit","bur","sid","sun","oil","red","doc","moe","caw","eel","dix","cub","end","gem","off","yew","hug","pop","tub","sgt","lid","pun","ton","sol","din","yup","jab","pea","bug","gag","mil","jig","hub","low","did","tin","get","gte","sox","lei","mig","fig","lon","use","ban","flo","nov","jut","bag","mir","sty","lap","two","ins","con","ant","net","tux","ode","stu","mug","cad","nap","gun","fop","tot","sow","sal","sic","ted","wot","del","imp","cob","way","ann","tan","mci","job","wet","ism","err","him","all","pad","hah","hie","aim","ike","jed","ego","mac","baa","min","com","ill","was","cab","ago","ina","big","ilk","gal","tap","duh","ola","ran","lab","top","gob","hot","ora","tia","kip","han","met","hut","she","sac","fed","goo","tee","ell","not","act","gil","rut","ala","ape","rig","cid","god","duo","lin","aid","gel","awl","lag","elf","liz","ref","aha","fib","oho","tho","her","nor","ace","adz","fun","ned","coo","win","tao","coy","van","man","pit","guy","foe","hid","mai","sup","jay","hob","mow","jot","are","pol","arc","lax","aft","alb","len","air","pug","pox","vow","got","meg","zoe","amp","ale","bud","gee","pin","dun","pat","ten","mob"]
    assert sorted(sol.findLadders(a,z,d)) == sorted([["cet","get","gee","gte","ate","ats","its","ito","ibo","ibm","ism"],["cet","cot","con","ion","inn","ins","its","ito","ibo","ibm","ism"],["cet","cat","can","ian","inn","ins","its","ito","ibo","ibm","ism"]])
    assert sorted(sol.findLadders("red","tax",["ted","tex","red","tax","tad","den","rex","pee"])) == sorted([['red', 'rex', 'tex', 'tax'], ['red', 'ted', 'tex', 'tax'], ['red', 'ted', 'tad', 'tax']])
    assert sorted(sol.findLadders("hit", "cog", ["hot","dot","dog","lot","log","cog"])) == sorted([["hit","hot","dot","dog","cog"],["hit","hot","lot","log","cog"]])
    assert sol.findLadders("hit", "cog", ["hot","dot","dog","lot","log"]) == []
    # assert sol.ladderLength("hit", "cog", ["hot","dot","dog","lot","log","cog"]) == 5
    # assert sol.ladderLength("hit", "cog", ["hot","dot","dog","lot","log"]) == 0
    # assert sol.knight(8,8,1,1,8,8) == 6
    # assert sol.knight(8,8,1,1,3,2) == 1
    # assert sol.knight(8,8,1,1,6,6) == 4
    # sortedL = [-10,-3,0,5,9]
    # llist = SinglyLinkedList()
    # for n in sortedL: llist.insert_node(n)
    # print_singly_linked_list(llist.head)
    # root = sol.sortedListToBST(llist.head)
    # print(inOrder(root))
    # tdata = [[1,4,5],[1,3,4],[2,6]]; llists = []
    # for l in tdata:
    #     llist = SinglyLinkedList()
    #     for n in l: llist.insert_node(n)
    #     llists.append(llist.head)
    # mllnodes = Solution.mergeKLists(llists)
    # print_singly_linked_list(mllnodes)
    # fptr = open(os.path.dirname(__file__) + "/insertNodeLinkedList.ut")
    # data = map(int,fptr.readline().strip().split(','))
    # llist = SinglyLinkedList()
    # for val in data:
    # for val in ['A','B','C','D','E']:
    #     llist.insert_node(val)
    # llist.insert_node('F') #case 2
    # print_singly_linked_list(llist.head)
    # llist_head = insertNodeAtPosition(llist.head, '2', 2)
    # r = Solution.reorderList(llist.head)
    # print_singly_linked_list(r)
