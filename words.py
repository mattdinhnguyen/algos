from typing import List
from collections import OrderedDict

class WordDictionary:

    def __init__(self):
        self.words = OrderedDict()

    def addWord(self, word: str) -> None:
        node = self.words
        for chr in word:
            node = node.setdefault(chr,{})
        node['$'] = None

    def search0(self, word: str) -> bool:
        return self.searchNode(self.    words, word)
    def searchNode(self, node, word: str) -> bool:
        for i, c in enumerate(word):
            if c == '.':
                return any(self.searchNode(node[w], word[i+1:]) for w in node if w != '$')
            if c not in node: return False
            node = node[c]
        return '$' in node

    def searchWords(self, word: str) -> bool:
        nodes = [self.words]
        for char in word + '$':
            nodes = [kid for node in nodes for kid in
                     ([node[char]] if char in node else
                      filter(None, node.values()) if char == '.' else [])]
        return bool(nodes)

    def search2(self, word: str) -> bool:
        nodes = [self.words]
        for char in word:
            nodes = [kid
                     for node in nodes
                     for key, kid in node.items()
                     if char in (key, '.') and kid]
        return any('$' in node for node in nodes)
    def insert(self, word: str) -> None:
        node = self.words
        for chr in word:
            node = node.setdefault(chr,{})
        node['$'] = word

    def search(self, word: str) -> bool:
        return self.startsWith(word + '$')

    def startsWith(self, prefix: str) -> bool:
        nodes = [self.words]
        for char in prefix:
            nodes = [kid for node in nodes for kid in ([node[char]] if char in node else [])]
        return bool(nodes)
    def remove(self, word:str) -> None:
        nodes = []
        node = self.words
        for ch in word+'$':
            if ch in node:
                nodes.append(node)
                node = node[ch]
        if nodes[-1]['$'] == word:
            del nodes[-1]['$']
        while len(node[-1]) == 0:
            del nodes[-1]
            nodes.pop()
# https://leetcode.com/problems/word-search-ii/discuss/59804/27-lines-uses-complex-numbers

class WordSearch(WordDictionary):
    def findWords(self, board, words):
        root = {}
        for word in words:
            node = root
            for c in word:
                node = node.setdefault(c, {})
            node[None] = True
        board = {i + 1j*j: c
                 for i, row in enumerate(board)
                 for j, c in enumerate(row)}

        found = []
        def search(node, z, word):
            if node.pop(None, None):
                found.append(word)
            c = board.get(z)
            if c in node:
                board[z] = None
                for k in range(4):
                    search(node[c], z + 1j**k, word + c)
                board[z] = c
        for z in board:
            search(root, z, '')

        return found

    def dfs(self, board, wdic, r, c, path, res):
        if self.wordCnt == 0: return
        if wdic.get('$', False):
            res.append(path)
            self.wordCnt -= 1
            wdic['$'] = False

        if 0 <= r < self.row and 0 <= c < self.col:
            tmp = board[r][c]
            if tmp not in wdic: return
            board[r][c] = "#"
            for x,y in [[0,-1], [0,1], [1,0], [-1,0]]:
                self.dfs(board, wdic[tmp], r+x, c+y, path+tmp, res)
            board[r][c] = tmp
    # best time
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        res = []
        self.row = len(board)
        self.col = len(board[0])
        self.wordCnt = len(words)
        for w in words:
            self.insert(w)
        for r in range(self.row):
            for c in range(self.col):
                self.dfs(board, self.words, r, c, "", res)         
        return res

if __name__ == '__main__':
    ws = WordSearch()
    board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]]
    words = ["oath","pea","eat","rain"]
    print(ws.findWords(board,words))

    words = WordDictionary()
    ans = []
    words.insert("apple");
    print("apple: ", words.search("apple"))   # returns true
    print("app: ", words.search("app"))  # returns false
    print("startsWith app: ", words.startsWith("app")) # returns true
    words.insert("app")
    # words.remove("app")
    print("app: ", words.search("app"))  # returns true
    for w in ["at","and","an","add"]:
        words.addWord(w)
    for w in ["a",".at"]:
        ans.append(words.searchWords(w))
    words.insert("bat")
    words.remove("bat")
    for w in [".at","an.","a.d.","b.","a.d","."]:
        ans.append(words.searchWords(w))
    print(ans)
'''["WordDictionary","addWord","addWord","addWord","addWord","search","search","addWord","search","search","search","search","search","search"]
[[],["at"],["and"],["an"],["add"],["a"],[".at"],["bat"],[".at"],["an."],["a.d."],["b."],["a.d"],["."]]
[null,null,null,null,null,false,false,null,true,true,false,false,true,false]'''