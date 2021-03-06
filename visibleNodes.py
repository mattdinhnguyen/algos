#!/usr/local/bin/python
from heapq import heapify, heappop, heappush, nlargest, nsmallest, heappushpop
from typing import List, Optional, Deque, Set
from bisect import insort, bisect, bisect_left, bisect_right
from subArraySumX import NumSubArrSumX
from collections import deque, Counter, defaultdict
from trees import BinarySearchTree, inOrder, preOrder
import string
from itertools import combinations, permutations, combinations_with_replacement
from functools import reduce, lru_cache, cache
import sys
import math
import operator
from timeit import timeit
class TreeNode: 
  def __init__(self,key): 
    self.left = None
    self.right = None
    self.val = key
  def __lt__(self, other):
    return self.val < other.val

class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
class MountainArray:
    def __init__(self, A: List[int]) -> None:
        self.arr = A
    def get(self, index: int) -> int:
       return self.arr[index]
    def length(self) -> int:
       return len(self.arr)

def visible_nodes0(root):
  visibleCount = 1
  descendants = [visible_nodes(c) for c in [root.left,root.right] if c]
  if descendants:
    visibleCount += max(descendants)
  return visibleCount

def visible_nodes(root):
  visibleCount = 0
  descendants = [(1,root)]
  while descendants:
    level, n = descendants.pop()
    ds = [(level+1,c) for c in [n.left,n.right] if c]
    if ds:
      descendants.extend(ds)
    else:
      visibleCount = max(visibleCount, level)
  
  return visibleCount
class Solution:
    # reverse decimal
    def reverse(self, x: int) -> int:
        # if (2**31 -1) < x < (-2**31): return 0
        n = 0
        sign = -1 if x < 0 else 1
        x = x*sign
        while x:
            n = n*10 + x%10
            x //= 10
        return n*sign if (-2**31) <= n*sign <= (2**31 -1) else 0
    # https://leetcode.com/problems/string-compression/
    def compress(self, chars: List[str]) -> int:
        def countS(countL, cnt):
            while cnt:
                countL.insert(1,str(cnt%10))
                cnt = cnt//10
            return countL

        if len(chars) < 2: return len(chars)
        prev, count, nextI = chars[0], 1, 0
        for i in range(1,len(chars)):
            if chars[i] == prev:
                count += 1
            elif count == 1:
                chars[nextI] = prev
                nextI += 1
                prev, count = chars[i], 1
            else:
                chLst = countS([prev],count)
                chars[nextI:nextI+len(chLst)] = chLst
                nextI += len(chLst)
                prev, count = chars[i], 1
        chLst = [prev] if count == 1 else countS([prev],count)
        chars[nextI:nextI+len(chLst)] = chLst
        nextI += len(chLst)
        chars = chars[:nextI]
        return nextI
    # https://leetcode.com/problems/string-compression/discuss/92568/Python-Two-Pointers-O(n)-time-O(1)-space
    def compress(self, chars):
        left = i = 0
        while i < len(chars):
            char, length = chars[i], 1
            while (i + 1) < len(chars) and char == chars[i + 1]:
                length, i = length + 1, i + 1
            chars[left] = char
            if length > 1:
                len_str = str(length)
                chars[left + 1:left + 1 + len(len_str)] = len_str
                left += len(len_str)
            left, i = left + 1, i + 1
        return left
    # 
    def compress(self, chars):
        st = i = 0 # st: start a new block
        while i < len(chars):
            while i < len(chars) and chars[i] == chars[st]: i += 1  
            if i - st == 1: st = i # next block st at i
            else:
                chars[st + 1 : i] = str(i - st) # copy the str of char count, shrinking the char block to len(str(i-st))
                st = st + 1 + len(str(i - st)) # st start next block
                i = st
        return len(chars)
    # https://leetcode.com/problems/rotate-image/discuss/18872/A-common-method-to-rotate-the-image
    # clockwise rotate: first reverse up to down, then swap the symmetry
    def rotate(self, matrix):
        n, m = len(matrix), len(matrix[0])
        for i in range(len(matrix)//2):
            j = n-i-1
            for k in range(m):
                matrix[i][k], matrix[j][k] = matrix[j][k], matrix[i][k]
        for i in range(n):
            for j in range(i+1,m):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        return matrix
    # anticlockwise rotate: first reverse left to right, then swap the symmetry
    def rotateAC(self, matrix: List[List[int]]) -> None:
        m, n = len(matrix), len(matrix[0])
        for i,r in enumerate(matrix):
            for j in range(n//2):
                k = n-j-1
                r[j], r[k] = r[k], r[j]
        for i in range(len(matrix)):
            for j in range(i+1,len(matrix[0])):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        return matrix
    # https://leetcode.com/problems/spiral-matrix/discuss/20599/Super-Simple-and-Easy-to-Understand-Solution
    def spiralOrder(self, matrix):
        res = []
        if matrix == None or len(matrix) == 0: return res
        n, m = len(matrix), len(matrix[0]) # n rows, m cols
        nm = n * m
        up,  down = 0, n - 1
        left, right = 0, m - 1
        while len(res) < nm:
            for j in range(left,right+1): # go left to right on row up 0..
                if len(res) < nm: res.append(matrix[up][j])
            for i in range(up + 1, down): # go up to down on col right
                if len(res) < nm: res.append(matrix[i][right])
            for j in range(right,left-1,-1): # go right to left on row down
                if len(res) < nm: res.append(matrix[down][j])
            for i in range(down-1,up,-1): # go down to up on col left
                if len(res) < nm: res.append(matrix[i][left])
            left += 1; right -= 1; up +=1; down -=1 
        return res
    # https://leetcode.com/problems/spiral-matrix/discuss/20571/1-liner-in-Python-%2B-Ruby
    # Take the first row plus the spiral order of the rotated remaining matrix.
    #     |1 2 3|      |6 9|      |8 7|      |4|  =>  |5|  =>  ||
    #     |4 5 6|  =>  |5 8|  =>  |5 4|  =>  |5|
    #     |7 8 9|      |4 7|
    def spiralOrder(self, matrix):
        return matrix and [*matrix.pop(0)] + self.spiralOrder([*zip(*matrix)][::-1])
    # https://leetcode.com/problems/spiral-matrix-ii/discuss/963128/Python-rotate-when-need-explained
    # Rotate 90 degrees clockwise on 1) reaching border, or 2) a filled cell
    def generateMatrix(self, n: int) -> List[List[int]]:
        x = y = dx = 0; dy = 1
        matrix = [[0] * n for _ in range(n)]
        for i in range(n*n):
            matrix[x][y] = i+1
            if not 0 <= x+dx < n or not 0 <= y+dy < n or matrix[x+dx][y+dy] != 0:
                dx, dy = dy, -dx
            x += dx; y += dy
        return matrix
    # https://www.youtube.com/watch?v=vAIp15jqQbU&t=442s
    def generateMatrix(self, n: int) -> List[List[int]]: # reverse of spiralORder
        if n == 1: return [[1]]
        matrix = [[None]*n for _ in range(n)]
        n2p1 = n * n +1
        up,  down = 0, n - 1
        left, right = 0, n - 1
        val = 1
        while val < n2p1:
            for j in range(left,right+1): # go left to right on row up 0..
                if val < n2p1: matrix[up][j] = val; val += 1
            for i in range(up + 1, down): # go up to down on col right
                if val < n2p1: matrix[i][right] = val; val += 1
            for j in range(right,left-1,-1): # go right to left on row down
                if val < n2p1: matrix[down][j] = val; val +=1
            for i in range(down-1,up,-1): # go down to up on col left
                if val < n2p1: matrix[i][left] = val; val += 1
            left += 1; right -= 1; up +=1; down -=1 
        return matrix
    # https://leetcode.com/problems/spiral-matrix-ii/discuss/22282/4-9-lines-Python-solutions
    # Start with the empty matrix, add the numbers in reverse order until we added the number 1.
    # Always rotate the matrix clockwise and add a top row:
    # ||  =>  |9|  =>  |8|      |6 7|      |4 5|      |1 2 3|
    #                  |9|  =>  |9 8|  =>  |9 6|  =>  |8 9 4|
    #                                      |8 7|      |7 6 5|
    def generateMatrix(self, n):
        A, lo = [], n*n+1
        while lo > 1:
            lo, hi = lo - len(A), lo
            A = [tuple(range(lo, hi))] + list(zip(*A[::-1])) # example shows A value at each iteration
        return A
    # https://www.interviewbit.com/problems/max-distance/ Completed Solution
    # https://medium.com/solvingalgo/solving-algorithmic-problems-max-distance-in-an-array-7e8c9f71c8b
    # max j-i where A[i] <= A[j]
    def maximumGap(self, A):
        array = list(range(len(A)))
        array.sort(key = lambda i: A[i]) # sorted A indices 
        a = b = maxDistance = 0
        minSofar = array[0]
        for i in array:
            if i <= minSofar:
                minSofar = i
            elif i - minSofar > maxDistance: # re-compute maxDistance when i (index) > minSofar index as walking the A[i] >= A[minSofar]
                maxDistance = i - minSofar
                a, b = i, minSofar
        return maxDistance
    # https://leetcode.com/problems/set-matrix-zeroes/submissions/
    def setZeroes(self, matrix: List[List[int]]) -> None:
        zy = []
        zx = []
        def merge(l1,l2): # in-place merge l2 into l1
            i = j = 0
            while i < len(l1) and j < len(l2):
                if l1[i] == l2[j]: i += 1; j += 1
                elif l1[i] > l2[j]: l1[i], l2[j] = l2[j], l1[i]; i += 1
                else: i += 1
            if j < len(l2): l1.extend(l2[j:])
        for i, r in enumerate(matrix):
            _zx = [j for j, x in enumerate(r) if x == 0]
            if _zx: zy.append(i)
            if len(zx) == 0: zx = _zx
            elif _zx:
                if zx[-1] < _zx[0]: zx.extend(_zx)
                elif zx[-1] > _zx[0]: merge(zx, _zx)
                else:
                    j = 0
                    while j < len(_zx):
                        if zx[-1] != _zx[j]:
                            zx.append(_zx[j])
                        j += 1
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if i in zy or j in zx:
                    matrix[i][j] = 0
    # Store information at the first element of each columns and rows. If a column contains a 0, it's first element will be 0. Same for rows.
    # However, both first column and first row use matrix[0][0] which is problematic so she creates another variable for first column, col0.
    # Finally, start setting zeros from the bottom right corner.
    # https://leetcode.com/problems/set-matrix-zeroes/discuss/26014/Any-shorter-O(1)-space-solution
    def setZeroes(self, matrix: List[List[int]]) -> None:
        col0, rows, cols = 1, len(matrix), len(matrix[0])
        for i in range(rows):
            if matrix[i][0] == 0: col0 = 0 # global col0
            for j in range(1, cols):
                if matrix[i][j] == 0: # row i, col j = 0
                    matrix[i][0] = matrix[0][j] = 0 # mark first of row/col to 0
        for i in range(rows-1, -1, -1):
            for j in range(1, cols):
                if matrix[i][0] == 0 or matrix[0][j] == 0: # row i, col j == 0, all elements in same row/col = 0
                    matrix[i][j] = 0
            if col0 == 0:
                matrix[i][0] = 0
        # print(matrix)
    # https://leetcode.com/problems/first-missing-positive/discuss/17071/My-short-c%2B%2B-solution-O(1)-space-and-O(n)-time
    def firstMissingPositive(self, A):
        A = list(filter(lambda x : x > 0, A))
        A.sort()
        for i in range(len(A)):
            if(A[i]!=i+1):
                return i+1
        return len(A)+1
    def firstMissingPositive(self, a: List[int]) -> int:
        if not a: return 1
        alen = len(a)
        for i in range(alen):
            pos = a[i]-1
            while 0 < a[i] <= alen and a[i] != a[pos]:
                a[i], a[pos] = a[pos], a[i] # keep putting a[i] in its position
                pos = a[i]-1 # pos for new a[i] value
        for i in range(alen):
            if a[i] != i+1:
                return i+1 # look for first value not in its position, return the value based on position
        return alen+1 # reach end of a, no missing positive
    # https://leetcode.com/problems/find-the-duplicate-number/discuss/72844/Two-Solutions-(with-explanation)%3A-O(nlog(n))-and-O(n)-time-O(1)-space-without-changing-the-input-array
    def findDuplicate(self, nums: List[int]) -> int: # slower
        low, high = 1, len(nums)-1
        while low < high:
            mid = low+(high-low)//2
            count = 0
            for i in nums:
                if i <= mid:
                    count+=1
            if count <= mid: low = mid+1
            else: high = mid
        return low
    # https://leetcode.com/problems/find-the-duplicate-number/discuss/72846/My-easy-understood-solution-with-O(n)-time-and-O(1)-space-without-modifying-the-array.-With-clear-explanation.
    # slow/fast pointers starting at position 0: slow == fast at circle entry point
    # https://keithschwarz.com/interesting/code/?dir=find-duplicate
    # slow/fast(2*slow)
    def findDuplicate(self, nums: List[int]) -> None:
        if len(nums) > 1:
            slow = nums[0]
            fast = nums[nums[0]] # 2*slow
            while (slow != fast): # slow == fast slow meets fast in circle
                slow = nums[slow]
                fast = nums[nums[fast]]
            fast = 0 # fast start at 0, slow starts at meet point, slow == fast at circle entry point
            while (fast != slow): # duplicate number must be the entry point of the circle when visiting the array from nums[0]
                fast = nums[fast]
                slow = nums[slow]
            return slow
        return -1
    # https://www.interviewbit.com/problems/maximum-unsorted-subarray/
    def subUnsort(self, A):
        b=sorted(A) # sorted form of A is B
        if(A==b):
            return [-1]
        else:
            # LIST OF ALL POINTS WHICH ARE NOT IN CORRECT PLACE
            L = [ i for i in range(len(A)) if A[i]!=b[i] ]
            return [min(L),max(L)]
    # leetcode
    def subUnsort(self, nums):
        if nums == None or len(nums) < 2: return 0
        maxVal, end = -sys.maxsize, -2
        for i in range(len(nums)):
            if nums[i] < maxVal: end = i # end points to the last nums value < maxVal (on left of end)
            else: maxVal = nums[i]
        minVal, begin = sys.maxsize, -1
        for i in range(len(nums)-1, -1, -1):
            if nums[i] > minVal: begin = i # begin points to the last (lowest index in reverse) nums value > minVal (on right of begin)
            else: minVal = nums[i]
        return [-1] if [begin,end] == [-1,-2] else [begin,end]
    # https://leetcode.com/discuss/interview-question/749936/maximum-absolute-difference-amazon
    #  1. (A[i]+i) - (A[j]+j) 2. (A[i]-i) - (A[j]-j)
    def maxAbsDiff(self, nums):
        mx1 = mx2 = -sys.maxsize; mn1 = mn2 = sys.maxsize
        for i in range(len(nums)):
            mx1 = max(mx1, nums[i]+i)
            mn1 = min(mn1, nums[i]+i)
            mx2 = max(mx2, nums[i]-i)
            mn2 = min(mn2, nums[i]-i)
        return max(mx1 - mn1, mx2 - mn2)
    def maxArr(self, a):
        ap = [n + i for i,n in enumerate(a)]
        am = [n - i for i,n in enumerate(a)]
        return max(max(ap) - min(ap), max(am) - min(am))
    # Loop over all items, for even positioned items, if the item is larger than next item, swap next and current.
    # For odd positioned items, if current item is smaller than next item, swap next and current.
    # Use logic to prove after swapping next and current, the inequality for current and previous still holds.
    def waveArray(self, A):
        A.sort()
        for i in range(len(A) -1):
            if i%2: # even due to 0-based index
                if A[i] > A[i+1]:
                    A[i], A[i+1] = A[i+1], A[i]
            elif A[i] < A[i+1]:
                A[i], A[i+1] = A[i+1], A[i]
        return A
    # https://leetcode.com/problems/rotate-array/discuss/54426/Summary-of-solutions-in-Python
    def rotate(self, A: List[int], k: int) -> None:
        def reverse(l,r):
            mid = (l+r)//2
            for i in range(l,mid+1):
                A[i], A[r] = A[r], A[i]
                r -= 1
        k %= len(A); m = len(A)-k; last = len(A)-1 # reverse first n-k elem, reverse rest, reverse A
        if k>0 and last>0:
            for l,r in [(0,m-1), (m,last), (0, last)]:
                reverse(l,r)
        return A
    def rotate(self, A: List[int], k: int) -> None:
        n = len(A); k, j = k % n, 0
        while n > 0 and k % n != 0:
            for i in range(0, k):
                A[j + i], A[len(A) - k + i] = A[len(A) - k + i], A[j + i] # swap
            n, j = n - k, j + k
            k = k % n
        return A
    # https://leetcode.com/problems/path-sum-ii/ find all root-to-leaf paths where each path's sum equals the given sum.
    def pathSumL(self, root: TreeNode, target: int) -> List[List[int]]:
        res = []
        if not root:
            return []
        stack = [(root, [root.val])] # each path starts at root, add values of descendants till leaves
        while stack: # iterative, N
            n, ls = stack.pop()
            if not n.left and not n.right and sum(ls) == target:
                insort(res, ls) # save found path
            if n.left:
                stack.append((n.left, ls + [n.left.val]))
            if n.right:
                stack.append((n.right, ls + [n.right.val]))
        return res
    def pathSumL(self, root: TreeNode, sum: int) -> List[List[int]]:
        if not root: return []
        if not root.left and not root.right and sum == root.val: return [[root.val]]
        tmp = self.pathSumL(root.left, sum-root.val) + self.pathSumL(root.right, sum-root.val)
        return [[root.val]+i for i in tmp]
    # https://leetcode.com/problems/sum-root-to-leaf-numbers/
    def sumNumbers(self, root: TreeNode) -> int:
        if not root: return 0
        sum = 0
        def dfs(node,val):
            nonlocal sum
            cs = [c for c in [node.left,node.right] if c]
            if cs:
                for c in cs:
                    dfs(c, (val*10)+c.val)
            else:
                sum += val
        dfs(root,root.val)
        return sum
    # https://leetcode.com/problems/binary-tree-maximum-path-sum/
    def maxPathSum(self, root: TreeNode) -> int:
        res = float('-inf')
        def subSum(node):
          nonlocal res
          if not node: return 0 # at leaf.left/right returns 0
          left = max(0, subSum(node.left))
          right = max(0, subSum(node.right))
          res = max(res, left + node.val + right) # maxPathSum = max(left + node.val + right)
          return node.val + max(left, right)
        subSum(root)
        return res
    # https://leetcode.com/problems/path-sum-iii/discuss/141424/Python-step-by-step-walk-through.-Easy-to-understand.-Two-solutions-comparison.-%3A-)
    def pathSum(self, root, target):
        self.result = 0
        cache = {0:1} # currPathSum, freq
        if root:
            self.dfs(root, target, 0, cache)
        return self.result
    '''The total count of valid paths in the subtree rooted at the current node, can be divided into three parts:
       - the number of valid paths ended by the current node
       - the total number of valid paths in the subtree rooted at the current node's left child
       - the total number of valid paths in the subtree rooted at the current node's right child'''
    def dfs(self, root, target, currPathSum, cache): #NlogN
        # calculate currPathSum and required oldPathSum
        currPathSum += root.val
        oldPathSum = currPathSum - target
        # update result and cache
        self.result += cache.get(oldPathSum, 0) # part 1: number of valid paths ended by the current node
        cache[currPathSum] = cache.get(currPathSum, 0) + 1
        
        # dfs breakdown
        for n in (root.left, root.right):
            if n:
                self.dfs(n, target, currPathSum, cache) # part 2 and 3
        # when move to a different branch, the currPathSum is no longer available, hence remove one. 
        cache[currPathSum] -= 1
    # https://leetcode.com/problems/subtree-of-another-tree/submissions/
    def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:
        def treeMatch(n1: TreeNode, n2: TreeNode) -> bool:
            if not (n1 and n2): return n1 is n2
            return n1.val == n2.val and treeMatch(n1.left, n2.left) and treeMatch(n1.right, n2.right)
        if treeMatch(s, t): return True
        if not s: return False
        return self.isSubtree(s.left, t) or self.isSubtree(s.right, t)
    # https://leetcode.com/problems/serialize-and-deserialize-binary-tree/
    def serialize(self, root):          
        res = []
        if root:
            stax = deque([root])
            while stax:
                n = stax.popleft()
                if n:
                    res.append(str(n.val)) # root,left,right BFS traversal
                    stax.extend([n.left,n.right])
                else:
                  res.append('#')
            while res[-1] == '#': res.pop() # trim null children at tail
        return ','.join(res)
    def deserialize(self, data: str) -> TreeNode:
        root = None
        if data:
            vals = map(lambda x: None if x == '#' else int(x), data.split(','))
            root = TreeNode(next(vals))
            nodes = deque([root])
            while nodes:
                node = nodes.popleft()
                left = next(vals, None)
                if left != None:
                    node.left = TreeNode(left)
                    nodes.append(node.left)
                right = next(vals, None)
                if right != None:
                    node.right = TreeNode(right)
                    nodes.append(node.right)
        return root
    def serialize(self, root):    
        if not root: return ""
        q = deque([root])
        res = []
        while q:
            node = q.popleft()
            if node:
                q.append(node.left)
                q.append(node.right)
            res.append(str(node.val) if node else '#')
        while res[-1] == '#': res.pop() # trim null children at tail
        return ','.join(res)
    def deserialize (self, data):
        if not data: return None
        nodes = data.split(',')
        root = TreeNode(int(nodes[0]))
        q = deque([root])
        index = 1
        while q:
            node = q.popleft()
            if nodes[index] != '#':
                node.left = TreeNode(int(nodes[index]))
                q.append(node.left)
            index += 1
            if nodes[index] != '#':
                node.right = TreeNode(int(nodes[index]))
                q.append(node.right)
            index += 1
        return root
    def serialize(self, root: TreeNode) -> str:
        if not root: return ""
        serializedVals: List[str] = []
        queue: Deque[TreeNode] = deque([root])
        while queue:
            node = queue.popleft()
            if node:
                serializedVals.append(str(node.val))
                queue.append(node.left)
                queue.append(node.right)
            else:
                serializedVals.append('#')
        while serializedVals[-1] == '#': serializedVals.pop()  # Strip trailing '#' nodes.
        return ','.join(serializedVals)
    def deserialize(self, data: str) -> Optional[TreeNode]:
        if not data: return None
        valsIter = (int(val) if val != '#' else None for val in data.split(','))
        root = TreeNode(next(valsIter))
        queue = deque([root])
        while queue:
            node = queue.popleft()
            val = next(valsIter, None)
            if val != None:
                node.left = TreeNode(val)
                queue.append(node.left)
            val = next(valsIter, None)
            if val != None:
                node.right = TreeNode(val)
                queue.append(node.right)
        return root
    # https://leetcode.com/problems/kth-largest-element-in-an-array/submissions/
    def findKthLargest(self, nums: List[int], k: int) -> int:
        return nlargest(k,nums)[-1]
    # https://leetcode.com/problems/find-k-closest-elements/
    # https://leetcode.com/problems/find-k-closest-elements/discuss/202785/Very-simple-Java-O(n)-solution-using-two-pointers
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]: # same speed, simple logic
        lo, hi = 0, len(arr)-1
        while hi - lo >= k:
            if x - arr[lo] > arr[hi] - x:
                lo += 1
            else:
                hi -= 1
        return arr[lo:hi+1]
    # https://leetcode.com/problems/find-k-closest-elements/discuss/106426/JavaC%2B%2BPython-Binary-Search-O(log(N-K)-%2B-K)
    # Assume we are taking A[i] ~ A[i + k -1]. We can binary research i
    # We compare the distance between x - A[mid] and A[mid + k] - x
    # if arr[mid] is farther from target than arr[mid+k] which is k places ahead of mid then we need to pull lo to mid with 1 offset;
    # otherwise we can pull hi at mid.
    # at the end we'll end up with a value contained by lo's index which can be the starting index of our solution
    def findClosestElements(self, A: List[int], k: int, x: int) -> List[int]:
        left, right = 0, len(A) - k
        while left < right: # O(log(N - K))
            mid = (left + right) // 2
            if x - A[mid] > A[mid + k] - x:
                left = mid + 1
            else:
                right = mid
        return A[left:left + k] # O(k)
    # https://leetcode.com/problems/kth-smallest-element-in-a-bst/
    def kthSmallestRe(self, root: TreeNode, k: int) -> int:
        res = []
        def lrr(node): # in-order
            nonlocal k, res
            if node:
                if node.left: lrr(node.left)
                k -= 1
                if k == 0: res.append(node.val); return
                if node.right: lrr(node.right)
        lrr(root)
        return res[0]
    def kthSmallestIt(self, root: TreeNode, k: int) -> int:
        stack = []
        while root or stack:
            while root:
                stack.append(root) # push nodes into stack till left-most leaf snd None
                root = root.left
            root = stack.pop(); k -= 1 # pop leaf, decr k, then its parent, then right child
            if k == 0: return root.val
            root = root.right
    # https://leetcode.com/problems/maximum-binary-tree/
    def constructMaximumBinaryTree(self, nums: List[int]) -> TreeNode:
        def findMax(nums: List[int], start: int, end: int) -> int:
            maxIdx = -1
            maxVal = float('-inf')
            for i in range(start,end+1):
                if nums[i] > maxVal:
                    maxVal = nums[i]
                    maxIdx = i
            return maxIdx
        def constructMBT(nums: List[int], start: int, end: int) -> TreeNode:
            if start > end:
                return None
            if start == end:
                return TreeNode(nums[start])
            maxIdx = findMax(nums, start, end)
            maxNode = TreeNode(nums[maxIdx])
            maxNode.left = constructMBT(nums, start, maxIdx-1)
            maxNode.right = constructMBT(nums, maxIdx+1, end)
            return maxNode
        return constructMBT(nums, 0, len(nums)-1) if nums else None
    # https://leetcode.com/problems/maximum-binary-tree/discuss/106147/c-8-lines-on-log-n-map-plus-stack-with-binary-search
    def constructMaximumBinaryTree(self, nums: List[int]) -> TreeNode:
        stax = [] # to ensure a decreasing order
        for val in nums:
            n = TreeNode(val)
            while stax and stax[-1].val < val: # pop all nodes in stack < current. last (largest < current) popped node becomes its left child
                n.left = stax.pop() # previous popped nodes still are right descendants of largest popped node
            if stax:
                stax[-1].right = n # current becomes right child of the nearest larger node in stack
            stax.append(n)
        return stax[0]
    def constructMaximumBinaryTree(self, nums: List[int]) -> TreeNode:
        s = [TreeNode(float('inf'))] # monotonous decreasing stack, same as above, but init 'inf' to save having to check stax empty
        for n in nums:
            node = TreeNode(n)
            while s[-1].val < n:
                node.left = s.pop()
            s[-1].right = node
            s.append(node) # push node into decreasing stack as right child of stack top
        return s[0].right
    def constructMinimumBinaryTree(self, nums: List[int]) -> TreeNode:
        stax = []
        for val in nums:
            n = TreeNode(val)
            while stax and stax[-1].val > val:
                n.left = stax.pop()
            if stax:
                stax[-1].right = n
            stax.append(n)
        return stax[0]
    def MinBinaryTreeSort(self, nums: List[int]) -> List[int]:
        root = self.constructMinimumBinaryTree(nums)
        h = [root]
        heapify(h)
        res = []
        while h:
            n = heappop(h)
            res.append(n.val)
            if n.left: heappush(h, n.left)
            if n.right: heappush(h, n.right)
        return res
    # https://leetcode.com/problems/maximum-width-of-binary-tree/
    def widthOfBinaryTree(self, root: TreeNode) -> int:
        stax = [(0,0,root)] # level, position in array
        ldic = {}
        while stax:
            level,p,n = stax.pop() # level, position in array, node
            ldic.setdefault(level,[]).append((p,n))
            if n.right: stax.append((level+1,p*2+2,n.right))
            if n.left: stax.append((level+1,p*2+1,n.left))
        res = 0
        for k,v in ldic.items():
            res = max(res, v[-1][0] - v[0][0] + 1)
        return res
    def widthOfBinaryTree(self, root):
        width = 0
        level = [(1, root)] # root number 1
        while level:
            width = max(width, level[-1][0] - level[0][0] + 1) # right most - left most numbers + 1
            # kid: number,node, set level to next level children
            level = [kid
                 for number, node in level
                 for kid in enumerate((node.left, node.right), 2 * number) # left,right 2,3
                 if kid[1]]
        return width
    # https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        def findPath(root: TreeNode, dest: TreeNode) -> List[TreeNode]:
            if root == dest:
                return [root]
            if root.val > dest.val:
                if root.left:
                    if root.left == dest:
                        return [root,root.left]
                    return [root] + findPath(root.left, dest)
            elif root.right:
                if root.right == dest:
                    return [root,root.right]
                return [root] + findPath(root.right, dest)
            return []
        pPath = findPath(root, p)
        qPath = findPath(root, q)
        for i,n in enumerate(pPath):
            if n != qPath[i]:
                return pPath[i-1]
        return None
    def lowestCommonAncestor(self, root, p, q):
        return root if (root.val - p.val) * (root.val - q.val) < 1 else \
           self.lowestCommonAncestor((root.left, root.right)[p.val > root.val], p, q)
    def lowestCommonAncestorBST(self, root, p, q): # best time, bestreadability
        while root:
            if p.val < root.val > q.val:
                root = root.left
            elif p.val > root.val < q.val:
                root = root.right
            else:
                return root
    def lowestCommonAncestorBST(self, root, p, q):
        a, b =  (p.val,q.val) if p.val < q.val else (q.val,p.val)
        while not a <= root.val <= b:
            root = (root.left, root.right)[a > root.val]
        return root
    def lowestCommonAncestor(self, root, p, q):
        while (root.val - p.val) * (root.val - q.val) > 0:
            root = (root.left, root.right)[p.val > root.val]
        return root
    # https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/
    def lowestCommonAncestorBT(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root in (None, p, q): return root # fall off leaf, or return first found target
        left = self.lowestCommonAncestor(root.left, p, q) # best time
        right = self.lowestCommonAncestor(root.right, p, q)
        # left, right = (self.lowestCommonAncestor(kid, p, q) # slower than above 2 lines
        #                for kid in (root.left, root.right))
        return root if left and right else left or right # the common ancestor node will get both p/q in left/right, intermediate ancestors get p or q, not both
    def lowestCommonAncestor(self, root, p, q): # slower than above method
        if root in (None, p, q): return root
        subs = [self.lowestCommonAncestor(kid, p, q)
                for kid in (root.left, root.right)]
        return root if all(subs) else subs[0] or subs[1]
    # https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/discuss/65236/JavaPython-iterative-solution
    # find where is p and q and a way to track their ancestors. A parent pointer for each node found is good for the job.
    # After we found both p and q, we create a set of p's ancestors.
    # Then we travel through q's ancestors, the first one appears in p's is our answer.
    def lowestCommonAncestor(self, root, p, q):
        stack = [root]
        parent = {root: None}
        while p not in parent or q not in parent: # build node-parent map for all descendents to p and q
            node = stack.pop() # start from root, find parent of all nodes to p and q
            if node.left:
                parent[node.left] = node
                stack.append(node.left)
            if node.right:
                parent[node.right] = node
                stack.append(node.right)
        ancestors = set()
        while p: # build p path up to root
            ancestors.add(p)
            p = parent[p]
        while q not in ancestors:
            q = parent[q] # find first q ancestor in p ancestors set
        return q # return the common ancestor
    # https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/discuss/65245/Iterative-Solutions-in-PythonC%2B%2B
    def lowestCommonAncestor(self, root, p, q):
        answer = []
        stack = [[root, answer]] # node, parent pair
        while stack:
            top = stack.pop()
            (node, parent), subs = top[:2], top[2:]
            if node in (None, p, q): parent += node, # found node or reached leaf
            elif not subs: stack += top, [node.right, top], [node.left, top]
            else: parent += node if all(subs) else max(subs),
        return answer[0]
    # find the paths to p and q and then find the last node where the paths match.
    def lowestCommonAncestorBT(self, root, p, q): # better time than above
        def path(root, goal):
            path, stack = [], [root]
            while stack:
                node = stack.pop() # build path starting at root, left, repeat till leaf, if goal not found, then backtrack path to right branch
                if node:
                    if node not in path[-1:]:
                        path += node, # build path starting at root, left, repeating till leaf
                        if node == goal: return path
                        stack += node, node.right, node.left # grow stack till left then right leaf. Push node to backtrack node out of path
                    else:
                        path.pop() # path has no goal, backrtack: pop path till stack & path diverge, then branch right, repeat till leaf
        pathp,pathq = path(root, p), path(root, q)
        return next(a for a, b in list(zip(pathp,pathq))[::-1] if a == b) if pathp and pathq else None
    # https://www.interviewbit.com/problems/least-common-ancestor/
    def lca(self, A, B, C):
        def getPath(root, val):
            if not root: return [] # on falling off leaf
            if root.val == val: return [root]
            left = getPath(root.left, val)
            right = getPath(root.right, val)
            if right: return [root] + right
            if left: return [root] + left
            return [] # empty list if not found val
        path1, path2 = getPath(A, B), getPath(A, C)
        lca = -1
        for a, b in zip(path1, path2):
            if a is not b: break # on path1, path2 diverge
            lca = a.val
        return lca
    def lca(self, root: TreeNode, p: int, q: int) -> int:
        def path(root, goal):
            path, stack = [], [root]
            while stack:
                node = stack.pop() # build path starting at root, left, repeat till leaf, if goal not found, then backtrack path to right branch
                if node:
                    if node not in path[-1:]:
                        path += node, # build path starting at root, left, repeating till leaf
                        if node.val == goal: return path
                        stack += node, node.right, node.left # grow stack till left then right leaf. Push node to backtrack node out of path
                    else:
                        path.pop() # path has no goal, backrtack: pop path till stack & path diverge, then branch right, repeat till leaf
        pathp,pathq = path(root, p), path(root, q)
        if not pathp or not pathq: return -1
        if p == q: return q
        return next(a for a, b in list(zip(pathp,pathq))[::-1] if a == b).val
    def reorganizeString(self, S: str) -> str:
        lenABC = len(string.ascii_lowercase)
        counters = [0]*lenABC
        orda = ord('a')
        hp = []
        for c in S:
            counters[ord(c)-orda] += 1
        for i,cnt in enumerate(counters):
            heappush(hp, (cnt,chr(i+orda)))
        A = []
        while hp:
            cnt, a = heappop(hp)
            A += [a]*cnt
        h = len(A) // 2
        A[1::2], A[::2] = A[:h], A[h:]
        return ''.join(A) * (A[-1:] != A[-2:-1])
    def reorganizeString(self, S):
        a = sorted(sorted(S), key=S.count)
        h = len(a) // 2
        a[1::2], a[::2] = a[:h], a[h:]
        return ''.join(a) * (a[-1:] != a[-2:-1])
    def reorganizeString(self, S: str) -> str: # best time
        res, c = [], Counter(S)
        pq = [(-value,key) for key,value in c.items()] # n
        heapify(pq) # nlogn
        p_a, p_b = 0, '' # previous a,b
        while pq:
            a, b = heappop(pq) # pop, take 1, save to prev, pop next, push prev. Repeat to cycle through most 2 frequent chars alternately 
            res += [b]
            if p_a < 0:
                heappush(pq, (p_a, p_b))
            a += 1
            p_a, p_b = a, b
        res = ''.join(res)
        if len(res) != len(S): return ""
        return res        
    def __init__(self):
        self.xfreq = defaultdict(int)
        self.freqs = defaultdict(list)
        self.maxfreq = 0

    def push(self, x: int) -> None:
        self.xfreq[x] += 1
        if self.xfreq[x] > self.maxfreq:
            self.maxfreq = self.xfreq[x]
        self.freqs[self.xfreq[x]].append(x)

    def pop(self) -> int:
        top = self.freqs[self.maxfreq].pop()
        self.xfreq[top] -= 1
        if not self.freqs[self.maxfreq]:
            self.maxfreq -= 1
        return top

    def solveSudoku(self, board: 'List[List[str]]') -> 'None':
        self.board = board
        self.solve()
        
    def findUnassigned(self):
        for row in range(9):
            for col in range(9):
                if self.board[row][col] == '.':
                    yield row, col
        yield -1, -1
        
    def solve(self):
        row, col = next(self.findUnassigned())
        if (row, col) == (-1, -1):
            return True
                       
        for num in map(str, range(1, 10)): # naive tries all values
            if self.isSafe(row, col, num):
                self.board[row][col] = num
                if self.solve():
                    return True
                self.board[row][col] = '.' # backtrack
        
    def isSafe(self, row, col, ch):
        rowSafe = all(self.board[row][_] != ch for _ in range(9))
        colSafe = all(self.board[_][col] != ch for _ in range(9))            
        squareSafe = all(self.board[r][c] != ch for r in self.getRange(row) for c in self.getRange(col))
        return rowSafe and colSafe and squareSafe
    
    def getRange(self, x):
        x -= x % 3
        return range(x, x + 3)
    # https://leetcode.com/problems/sudoku-solver/discuss/365070/~35ms-beats-100.-Python-DFS-Solution-With-Explanation
    # Keep track of 
    # 1. all empty spots along with all candidate digits that can be placed at those spots
    # - Use DFS to try the empty spot with fewest candidates (using 1) (optimization)
    # - Try place each candidate at that spot.
    # - Scan through all other empty spots and modify their candidate ditis
    #   if they are affected by this placement. (keep track of this update)
    #   Terminate early if any other spot loses all candidates. (important optimization)
    # - Backtrack if this path failed, and un-modify all the changes made.
    def solveSudoku(self, board: 'List[List[str]]') -> 'None': # best time
        # [Map (Int, Int) [Set Int]]
        # Keys are remaining empty spots during the search represented as (row, col).
        # Values are the candidate digits that can be placed at (row, col)
        empty_spots = {}

        # row_valids[i] is a Set of digits which is found at i-th row in board
        row_digits = [set() for j in range(9)]
        # col_valids[i] is a Set of digits which is found at i-th column in board
        col_digits = [set() for j in range(9)]
        # sqr_valids[i] is a Set of digits which is found at i-th square in board
        sqr_digits = [set() for j in range(9)]

        # convert strings in the board to integers
        # fill in the above lists during this process
        int_board = []
        for row in range(9):
            int_row = []
            for col in range(9):
                s = board[row][col]
                sqr = (row // 3) * 3 + (col // 3)
                if s.isdigit():
                    num = int(s)
                    int_row.append(num)
                    row_digits[row].add(num)
                    col_digits[col].add(num)
                    sqr_digits[sqr].add(num)
                else:
                    int_row.append(0)
                    empty_spots[(row, col)] = set()
            int_board.append(int_row)

        # fill in empty_spots
        for row, col in empty_spots:
            candidates = empty_spots[(row, col)] # empty_spots in lieu of findUnassigned()
            sqr = (row // 3) * 3 + (col // 3)
            for n in range(1, 10):
                # if n is can be placed at (row, col) and sqr-th square
                # it's a valid candidate for (row, col)
                if not (n in row_digits[row] or n in col_digits[col] or n in sqr_digits[sqr]):
                    candidates.add(n)


        def dfs():
            # Try filling in each empty spot per rule of sudoku,
            # do a dfs and backtrack to look for a potential solution.
            # Modify the board in place and return True iff a solution was found.

            # if no more empty spots left, a solution was found
            if not empty_spots:
                return True

            # find the empty spot with fewest candidates
            target_row, target_col = min([spot for spot in empty_spots], key=lambda s : len(empty_spots[s]))
            # (target_row, target_col), candidates = min([(s,v) for s,v in empty_spots.items()], key=lambda t: len(t[1]))
            target_sqr = (target_row // 3) * 3 + (target_col // 3)
            candidates = empty_spots[(target_row, target_col)]

            # remove this from empty_spots
            del empty_spots[(target_row, target_col)]
            # try placing each candidate at (target_row, target_col)
            for n in candidates:
                # for rest of empty spots, if they are in the same row/col/sqr
                # as n, and contains n, they need to be updated. We keep
                # track of these updates in case of backtracking
                updated_spots = []
                # whether the placement of n failed (it invalidates other empty spots)
                failed = False
                for spot, valids in empty_spots.items():
                    row, col = spot
                    sqr = (row // 3) * 3 + (col // 3)
                    if n in valids and (target_row == row or target_col == col or target_sqr == sqr):
                        valids.remove(n)
                        updated_spots.append(spot)
                    if not valids:
                        failed = True
                        break

                # If the placement was successful,
                # keep doing the dfs with leftover empty spots
                if not failed and dfs():
                    # modify the board iff a solution was found
                    # this keeps number of modification minimal
                    int_board[target_row][target_col] = n
                    return True

                # Backtrack and un-modify all changes
                for spot in updated_spots:
                    empty_spots[spot].add(n)
                int_board[target_row][target_col] = 0
                
            # All candidates failed, this path of search should be abandonded.
            # Restore the target to empty_spots and backtrack
            empty_spots[(target_row, target_col)] = candidates
            return False

        if not dfs():
            raise RuntimeError("No solution possible for given board")
        
        # Write solution back into the original board
        for row in range(9):
            for col in range(9):
                board[row][col] = str(int_board[row][col])
    # https://leetcode.com/problems/sudoku-solver/discuss/140837/Python-very-simple-backtracking-solution-using-dictionaries-and-queue-~100-ms-beats-~90
    def solveSudoku(self, board): # simplest, most elegant, 2nd best time
        rows, cols, triples, visit = defaultdict(set), defaultdict(set), defaultdict(set), deque([])
        for r in range(9):
            for c in range(9):
                if board[r][c] != ".":
                    rows[r].add(board[r][c]) # nums in each row
                    cols[c].add(board[r][c]) # nums in each col
                    triples[(r // 3, c // 3)].add(board[r][c]) # nums in each sub-square
                else: visit.append((r, c)) # positions to be assigned nums
        def dfs():
            if not visit: return True # assigned nums to all positions successfully
            r, c = visit[0] # look-ahead from head of queue (first empty position in square)
            t = (r // 3, c // 3)
            for dig in {"1", "2", "3", "4", "5", "6", "7", "8", "9"}:
                if dig not in rows[r] and dig not in cols[c] and dig not in triples[t]: # try digits have not been assigned
                    board[r][c] = dig # update board, rows, cols, sub-squares
                    rows[r].add(dig); cols[c].add(dig); triples[t].add(dig)
                    visit.popleft() # pop look-ahead position before dfs
                    if dfs(): return True
                    else: # failed, back track
                        board[r][c] = "."; rows[r].discard(dig); cols[c].discard(dig); triples[t].discard(dig)
                        visit.appendleft((r, c))
            return False
        dfs()
    # darkTianTian in above link
    def solveSudoku(self, g: List[List[str]]) -> None:
        to_add = []; row = [[True]*9 for i in range(9)]; col = [[True]*9 for i in range(9)]
        sub = [[True]*9 for i in range(9)]  # 3*3 sub-box, from left to right, top to bottom.
        for i in range(9):
            for j in range(9):
                if g[i][j] != '.':
                    d = int(g[i][j]) - 1
                    row[i][d] = col[j][d] = sub[i//3*3+j//3][d] = False # has digits
                else: to_add.append((i, j))
        def backtrack():
            if not to_add:
                return True
            i, j = to_add.pop()
            for d in range(9):
                if row[i][d] and col[j][d] and sub[i//3*3+j//3][d]:
                    g[i][j] = str(d+1)
                    row[i][d] = col[j][d] = sub[i//3*3+j//3][d] = False
                    if backtrack(): return True
                    g[i][j] = '.' # d value failed, undo
                    row[i][d] = col[j][d] = sub[i//3*3+j//3][d] = True
            to_add.append((i, j))
            return False
        backtrack()
    # 
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        n = len(board); ord0 = ord('0')
        rows = [set() for _ in range(n)]; cols = [set() for _ in range(n)]; subm = [[set() for _ in range(3)] for _ in range(3)]
        for i,r in enumerate(board):
            for j,c in enumerate(r):
                if c != '.':
                    d = ord(c)-ord0
                    if d in rows[i] or d in cols[j] or d in subm[i//3][j//3]:
                        return False
                    rows[i].add(d); cols[j].add(d); subm[i//3][j//3].add(d)
        return True
    # https://leetcode.com/problems/n-queens/discuss/19971/Python-recursive-dfs-solution-with-comments.
    def solveNQueens(self, n: int) -> List[List[str]]:
        res = []; queens = [-1] * n
        # queens [1, 3, 0, 2] means [.Q..] [...Q] [Q...] [..Q.]
        def dfs(index): # index represents row no and value represents col no
            if index == len(queens): # n queens have been placed correctly
                res.append(queens[:])
                return  # backtracking
            for i in range(len(queens)):
                queens[index] = i # index represents row no and value i represents col no
                if valid(index):  # pruning
                    dfs(index + 1)
        def valid(n): # check whether nth queens can be placed
            for i in range(n):
                if abs(queens[i] - queens[n]) == n - i:  # same diagonal
                    return False
                if queens[i] == queens[n]:  # same column
                    return False
            return True
        def make_board(queens): # given queens = [1,3,0,2] this function returns [.Q..] [...Q] [Q...] [..Q.]
            n = len(queens)
            board = []
            board_temp = [['.'] * n for _ in range(n)]
            for row, col in enumerate(queens):
                board_temp[row][col] = 'Q'
            for row in board_temp:
                board.append("".join(row))
            return board
        def make_all_boards(res):
            actual_boards = []
            for queens in res:
                actual_boards.append(make_board(queens))
            return actual_boards
        dfs(0)
        return make_all_boards(res)
    # 
    def solveNQueens(self, n):
        def valid(nums, n): # check whether nth queen can be placed in that column
            for i in range(n):
                if abs(nums[i]-nums[n]) == n -i or nums[i] == nums[n]:
                    return False
            return True
        def dfs(nums, index, path, res):
            if index == len(nums):
                res.append(path)
                return  # backtracking
            for i in range(len(nums)):
                nums[index] = i
                if valid(nums, index):  # pruning
                    tmp = "."*len(nums)
                    dfs(nums, index+1, path+[tmp[:i]+"Q"+tmp[i+1:]], res)
        res = []
        dfs([-1]*n, 0, [], res)
        return res
    #
    def grayCode(self, n):
        return [i^(i>>1) for i in range(2**n)]
    # https://leetcode.com/problems/palindromic-substrings/
    def countSubstrings(self, s: str) -> int: # "a", "a", "a", "aa", "aa", "aaa"
        palCnt = 0
        for a in range(len(s)):
            for z in range(a+1,len(s)+1):
                if self._is_palindrome(s,a,z-1):
                    palCnt += 1
        return palCnt
    def countSubstrings(self, s: str) -> int: # "a", "a", "a", "aa", "aa", "aaa"
        slen = len(s)
        dp = [[0]*slen for _ in range(slen)]
        palCnt = 0
        for l in range(slen-1,-1,-1):
            for r in range(l, slen):
                dp[l][r] = s[l] == s[r] and ((r-l+1) < 3 or dp[l+1][r-1])
                palCnt += dp[l][r]
        return palCnt
    # https://leetcode.com/problems/palindromic-substrings/discuss/105687/Python-Straightforward-with-Explanation-(Bonus-O(N)-solution)
    def countSubstrings(self, S: str) -> int: # best time
        def manachers(S):
            A = '@#' + '#'.join(S) + '#$'
            Z = [0] * len(A)
            center = right = 0
            for i in range(1, len(A) - 1):
                if i < right:
                    Z[i] = min(right - i, Z[2 * center - i])
                while A[i + Z[i] + 1] == A[i - Z[i] - 1]:
                    Z[i] += 1
                if i + Z[i] > right:
                    center, right = i, i + Z[i]
            return Z
        return sum((v+1)//2 for v in manachers(S))
    # https://leetcode.com/problems/valid-palindrome/
    def isPalindrome(self, s: str) -> bool:
        # sa = s.translate(str.maketrans('', '', '`\'-\][}{)("#,;! :.@_?')).lower()
        # delchars = ''.join(c for c in map(chr, range(256)) if not c.isalnum())
        # sa = s.translate(str.maketrans('','', delchars)).lower()
        delchars = ''.join(c for c in map(chr, range(256)) if not c.isalnum())
        sa = s.translate(str.maketrans('','', delchars)).lower()
        l, r = 0, len(sa)-1
        while(l < r and sa[l] == sa[r]):
            l += 1
            r -= 1
        return True if l >= r else False
    def isPalindromeK(self, sa: str, k: int = 1) -> bool:
        l, r = 0, len(sa)-1
        while(l < r):
            if sa[l] == sa[r]:
                l += 1
                r -= 1
            elif k:
                s1 = sa[l+1:r+1]
                s2 = sa[l+1:r-1]
                return s1 == s1[::-1] or s2 == s2[::-1]
            else:
                return False
        return True if l >= r else False
    def convert2Palin(self, sa: str): # convert to palindrome by deleting 1 https://www.interviewbit.com/problems/convert-to-palindrome/
        l, r = 0, len(sa)-1; k = 1
        while(l < r):
            if sa[l] == sa[r]:
                l += 1
                r -= 1
            elif k:
                s1 = sa[l+1:r+1]
                s2 = sa[l:r]
                return 1 if s1 and s1 == s1[::-1] or s2 and s2 == s2[::-1] else 0
            else:
                return 0
        return 1 if l >= r else 0
    # https://leetcode.com/problems/monotonic-array/
    def isMonotonic(self, A: List[int]) -> bool:
        i = 1
        while i < len(A):
            if A[i] != A[i-1]: break
            i += 1
        if i == len(A): return True
        if A[0] <= A[i]:
            for i in range(i,len(A)):
                if A[i] < A[i-1]: return False
        else:
            for i in range(i,len(A)):
                if A[i] > A[i-1]: return False
        return True
    def isMonotonic(self, A: List[int]) -> bool:
        return len({x < y for x, y in zip(A, A[1:]) if x != y}) <= 1 
    # https://leetcode.com/problems/move-zeroes/
    def moveZeroes(self, nums: List[int]) -> None:
        for i,n in enumerate(nums):
            if n == 0:
                j = i
                break
        for i in range(j+1,len(nums)):
            if nums[i]:
                nums[j] = nums[i]
                j += 1
        while j < len(nums):
            nums[j] = 0
            j += 1
    # https://leetcode.com/problems/jump-game/
    def canJump(self, nums: List[int]) -> bool:
        end = len(nums)-1
        canJumpHeap = [end]
        for i in range(end-1,-1,-1):
            if nums[i] >= canJumpHeap[0] - i:
                heappush(canJumpHeap,i)
        return True if canJumpHeap[0] == 0 else False
    # https://leetcode.com/problems/jump-game/discuss/20907/1-6-lines-O(n)-time-O(1)-space
    def canJump(self, nums):
        m = 0 # Going forwards. m: maximum index we can reach so far.
        for i, n in enumerate(nums):
            if i > m: return False
            m = max(m, i+n)
        return True
    def canJump(self, nums):
        m = 1
        for i,n in enumerate(nums,1):
            m = max(m, i+n) * (i <= m)
        return m > 0
    def canJump(self, nums): # Going backwards, most people seem to do that, here's my version.
        l = len(nums); goal = l - 1
        for i in range(len(nums))[::-1]:
            if i + nums[i] >= goal:
                goal = i
        return not goal
    # https://leetcode.com/problems/unique-paths/discuss/22954/C%2B%2B-DP
    # https://leetcode.com/problems/unique-paths/discuss/184248/8-lines-Java-DP-solution-0ms-beats-100-explained-with-graph
    def uniquePaths(self, m: int, n: int) -> int:
        # m+n-2 C n-1 = (m+n-2)! / (n-1)! (m-1)! 
        ans = 1
        for i in range(n, m + n - 1):
            ans = ans*i//(i - n + 1) # best time
        return ans # reduce(lambda ans,i: ans*i//(i - n + 1), range(n, m + n - 1), 1)
    def uniquePaths(self, m: int, n: int) -> int:
        cur = [1]*n # init 1st row to 1
        for i in range(1,m): # start adding on 2nd row
            for j in range(1,n):
                cur[j] += cur[j-1]
        return cur[-1]
    # https://leetcode.com/problems/unique-paths-ii/discuss/527282/Python-DFS%2BDP-explained-solution
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        M, N = len(obstacleGrid), len(obstacleGrid[0])
        @lru_cache(maxsize=None)
        def dfs(i, j):
            if obstacleGrid[i][j]:      # hit an obstacle
                return 0
            if i == M-1 and j == N-1:   # reach the end
                return 1
            count = 0
            if i < M-1:
                count += dfs(i+1, j)    # go down
            if j < N-1:
                count += dfs(i, j+1)    # go right
            return count
        return dfs(0, 0)
    # https://leetcode.com/problems/unique-paths-ii/discuss/23250/Short-JAVA-solution
    # obstacleGrid =  0,0,0
    #                 0,1,0
    #                 0,0,0
    #    index of dp 0,1,2,3
    #   first time   0,1,1,1
    #   sec   time   0,1,0,1
    #   third time   0,1,1,2
    #
    #  #*****************
    # obstacleGrid =  0,0,0
    #                 0,0,0
    #                 0,0,0
    #    index of dp 0,1,2,3
    #   first time   0,1,1,1
    #   sec   time   0,1,2,3
    #   third time   0,1,3,6
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        dp = [0,1] + [0]*(n-1)
        for i in range(m):
            for j in range(1,n+1):
                if obstacleGrid[i][j-1] == 1: dp[j] = 0
                else: dp[j] += dp[j -1]
        return dp[n]
    # 
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        width = len(obstacleGrid[0])
        dp = [1] + [0]*(width-1)
        for row in obstacleGrid:
            for j in range(width):
                if row[j] == 1: dp[j] = 0
                elif j: dp[j] += dp[j - 1] # recurrence: new dp[j] = old dp[j] + dp[j-1]
        return dp[-1]
   # https://leetcode.com/problems/minimum-path-sum/submissions/
    def minPathSum(self, grid: List[List[int]]) -> int:
        width = len(grid[0])
        hei = len(grid)
        for i in range(1,width):
            grid[0][i] += grid[0][i-1]
        for j in range(1,hei):
            grid[j][0] += grid[j-1][0]
        for i in range(1,hei):
            for j in range(1,width):
                grid[i][j] += min(grid[i-1][j],grid[i][j-1])
        return grid[-1][-1]
    # https://leetcode.com/problems/cherry-pickup/discuss/329945/Very-easy-to-follow-%3A-step-by-step-recursive-backtracking-with-memoization-N4.
    def cherryPickup(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        mem = [ [ [-2**31 for i in range(m)] for j in range(n) ] for k in range(m)]
        def dfs(grid, mem, m, n, x1, y1, x2, y2):
            if x1>=m or x2>=m or y1 >= n or y2>=n or grid[x1][y1] == -1 or grid[x2][y2] == -1: return -2**31
            if mem[x1][y1][x2] != -2**31: return mem[x1][y1][x2]
            cherry = 0
            if grid[x1][y1] == 1: cherry+=1
            if x1!=x2 and grid[x2][y2] == 1: cherry+=1
            if x1==m-1 and y1==n-1: return cherry
            cherry += max(
                dfs(grid, mem, m, n, x1+1, y1, x2+1, y2),
                dfs(grid, mem, m, n, x1+1, y1, x2, y2+1),
                dfs(grid, mem, m, n, x1, y1+1, x2+1, y2),
                dfs(grid, mem, m, n, x1, y1+1, x2, y2+1),
                )
            mem[x1][y1][x2] = cherry
            return cherry
        return max(0, dfs(grid, mem, m, n, 0, 0, 0, 0))
    def cherryPickup(self, grid: List[List[int]]) -> int:
        n = len(grid)
        @lru_cache(None)
        def dfs(x1, y1, x2, y2):
            if any(a >= n for a in (x1, y1, x2, y2)) or grid[x1][y1] == -1 or grid[x2][y2] == -1:
                return float('-inf')
            if (n-1, n-1) in ((x1, y1),(x2, y2)): return grid[-1][-1]            
            if (x1, y1) == (x2, y2): cherries = grid[x1][y1]
            else: cherries = grid[x1][y1] + grid[x2][y2]
            return cherries + max(dfs(x1 + 1, y1, x2 + 1, y2), dfs(x1 + 1, y1, x2, y2 + 1), dfs(x1, y1 + 1, x2 + 1, y2), dfs(x1, y1 + 1, x2, y2 + 1))
        return max(0, dfs(0, 0, 0, 0))
    def climbStairs0(self, n: int) -> int:
        def ways(i):
            if i < 3: return i
            return ways(i-1) + ways(i-2)
        return ways(n)
    def climbStairs(self, n: int) -> int:
        a, b = 1, 1
        for i in range(n):
            a, b = b, a + b
        return a
    # https://leetcode.com/problems/decode-ways/
    # https://leetcode.com/problems/decode-ways/discuss/608268/Python-Thinking-process-diagram-(DP-%2B-DFS)
    def numDecodings(self, s: str) -> int:
        if not s or s[0] == '0': return 0
        @lru_cache(maxsize=None)
        def dfs(string):
            if string and string[0] == '0':
                return 0
            if string == "" or len(string) == 1:
                return 1
            if int(string[:2]) <= 26:
                first = dfs(string[1:])
                second = dfs(string[2:])
                return first+second
            else:
                return dfs(string[1:])
        return dfs(s)
    def numDecodings(self, s: str) -> int:
        if not s or s[0] == '0': return 0 # s[0] starts with 0
        dp = [0]*(len(s)+1)
        dp[0:2] = [1,1] # dp[0] = 1 supporting case i == 2, 2nd if conditiion is True
        # dp[1] = 1 if int(s[0]) else 0 # s[0] starts with 0
        for i in range(2,len(s)+1): # s[1] to s[-1]
            if '0' < s[i-1]: # 1 step
                dp[i] += dp[i-1]
            if 10 <= int(s[i-2:i]) <= 26: # 2 steps
                dp[i] += dp[i-2]
        return dp[len(s)]
    # https://leetcode.com/problems/maximum-product-subarray/submissions/
    def maxProduct(self, nums: List[int]) -> int:
        imin = imax = maxP = nums[0]
        for i,n in enumerate(nums[1:],1):
            if n < 0: imin, imax = imax, imin
            imin = min(n, imin*n)
            imax = max(n, imax*n)
            maxP = max(maxP, imax)
        return maxP
    # https://leetcode.com/problems/maximum-product-subarray/discuss/48276/Python-solution-with-detailed-explanation
    def maxProduct(self, nums: List[int]) -> int:
        max_prod, min_prod, ans = nums[0], nums[0], nums[0]
        for i in range(1, len(nums)):
            x = max(nums[i], max_prod*nums[i], min_prod*nums[i])
            y = min(nums[i], max_prod*nums[i], min_prod*nums[i])            
            max_prod, min_prod = x, y
            ans = max(max_prod, ans)
        return ans
    # https://leetcode.com/problems/maximum-product-subarray/discuss/183483/JavaC%2B%2BPython-it-can-be-more-simple
    def maxProduct(self, A: List[int]) -> int:
        B = A[::-1]
        for i in range(1, len(A)):
            A[i] *= A[i - 1] or 1 # prefix products, restart at A[i] if A[i-1] == 0
            B[i] *= B[i - 1] or 1 # suffix products
        return max(A + B) # max of both
    # https://leetcode.com/problems/house-robber/discuss/55696/Python-solution-3-lines.
    # f(0) = nums[0]
    # f(1) = max(num[0], num[1])
    # f(k) = max( f(k-2) + nums[k], f(k-1) )
    # Approach 1:- Construct dp table
    def rob(self, nums: List[int]) -> int: # iterative + memo
        if not nums: return 0
        if len(nums) == 1: return nums[0]
        dp = [0] * len(nums)
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        for i in range(2, len(nums)):
            dp[i] = max(nums[i] + dp[i-2], dp[i-1])
        return dp[-1] # return the last element
    def robLinear(self, nums: List[int]) -> int: # iterative + 2 variables
        prev = cur = 0
        for n in nums:
            prev, cur = cur, max(cur, prev+n)
        return cur
    # first and last are adjancent in circle
    # https://leetcode.com/problems/house-robber-ii/discuss/299071/Python-O(n)-time-O(1)-space
    def robCircle(self, nums: List[int]) -> int:
        return max(self.robLinear(nums[1:]), self.robLinear(nums[:-1]))
    # pass index, O(1) space
    def rob(self, nums: List[int]) -> int:
        if len(nums) < 2:
            return max(nums + [0])
        def regular_rob(l,r):
            last, now = 0, 0
            while l < r:
                last, now, l = now, max(last + nums[l], now), l+1
            return now
        return max(regular_rob(1, len(nums)), regular_rob(0, len(nums)-1))
    # https://leetcode.com/problems/longest-continuous-increasing-subsequence/submissions/ 
    def findLengthOfLCIS(self, nums: List[int]) -> int:
        if not nums: return 0
        maxlcis = lcis = 1
        for i in range(1,len(nums)):
            if nums[i] > nums[i-1]:
                lcis +=1
                if lcis > maxlcis:
                    maxlcis = lcis
            else:
                lcis = 1
        return max(maxlcis, lcis)
    # https://www.lintcode.com/problem/minimum-window-subsequence/description
    # dp[i][j] = k, S(0…i) has T(0…j) starting at index k of S.
    # For T, our starting index would be one amongst dp[0][t-1], dp[1][t-1], …, dp[s-1][t-1], where s and t are the lengths of s and t respectively.
    # recurrence: dp[i][j] = dp[i-1][j-1] if S[i] == T[j]  else dp[i-1][j],
    # when S[i] == T[j], S(0…i) contains T(0…j) and the last character is the same,
    # so we have to search for the remaining T, i.e, T(0…j-1) in S(0…i-1) else we would have to search for the entire T(0…j) in S(0…i-1).
    # Also, dp[i][j] = -1 if subsequence T is not found in S.
    def minWindow(self, S, T):
        s = len(S)
        t = len(T)
        dp = [[-1]*t for _ in range(s)]
        for i in range(s):
            if S[i] == T[0]: dp[i][0] = i # S(0..i) contains T(0)
            elif i: dp[i][0] = dp[i-1][0]
        for i in range(1,s):
            for j in range(1,t):
                if S[i] == T[j]: dp[i][j] = dp[i-1][j-1]
                else: dp[i][j] = dp[i-1][j]
        begin, length = -1, s+1
        for i in range(s):
            index = dp[i][t-1] # start at T(0..t-1)
            if index != -1:
                curLength = i - index + 1
                if curLength < length:
                    begin = index
                    length = curLength
        return "" if begin == -1 else S[begin:begin+length]
    # https://github.com/shehabic/java-algorithms/blob/master/src/solutions/MinWindowSubSequence.java
    def minWindow(self, S, T): # O(1), O(s*t)
        window = ""
        i, j, min = 0, 0, len(S) + 1
        while i < len(S):
            if S[i] == T[j]:
                j += 1
                if j == len(T): # found all T chars
                    end = i + 1
                    j -= 1
                    while j >= 0:
                        if S[i] == T[j]:
                            j -= 1
                        i -= 1 # move i left till j<0 to get min window
                    j += 1
                    i += 1
                    if end - i < min:
                        min = end - i
                        window = S[i:end]
            i += 1 # start next window
        return window
    # xhttps://en.wikipedia.org/wiki/Patience_sorting
    # https://leetcode.com/problems/longest-increasing-subsequence/discuss/74824/JavaPython-Binary-search-O(nlogn)-time-with-explanation
    def lengthOfLIS(self, nums: List[int]) -> int:
        tails = [0] * len(nums); size = 0
        for x in nums:
            i, j = 0, size 
            while i != j: # binary search between 0 to size for tails[i] > x
                m = (i + j) // 2
                if tails[m] < x:
                    i = m + 1
                else:
                    j = m
            tails[i] = x
            size = max(i + 1, size) # size must be at least i+1 (could be a new pile)
        # https://www.cs.princeton.edu/courses/archive/spring13/cos423/lectures/LongestIncreasingSubsequence.pdf
        print(tails)
        return size # longest strictly inscreasing sequence, see princeton
    def lengthOfLIS(self, nums: List[int]) -> int: # patience sort, tails[i] has the smallest number of bin i
        tails = []
        for x in nums:
            i = bisect_left(tails, x)
            if i == len(tails):
                if not tails or tails[i-1] != x: # only append 1st 7, skip others 7 7 7 7 7
                    tails.append(x)
            else:
                tails[i] = x
        print(tails)
        return len(tails)
    def findNumberOfLIS(self, nums: List[int]) -> int:
        tails = []
        dp = defaultdict(Counter)
        dp[-1][-10**6] = 1
        for x in nums:
            i = bisect_left(tails, x)
            if i == len(tails):
                if not tails or tails[i-1] != x: # only append 1st 7, skip others 7 7 7 7 7
                    tails.append(x)
            else:
                tails[i] = x
            # number of LIS of len i, ends with x dp[0][all values < 10**6]
            dp[i][x] += sum(dp[i-1][j] for j in dp[i-1] if j < x) # for all LIS of len i-1, end values j < x
        return sum(dp[max(0, len(tails)-1)].values())
    # https://leetcode.com/problems/russian-doll-envelopes/discuss/82763/Java-NLogN-Solution-with-Explanation
    def maxEnvelopes(self, envelopes):
        if not envelopes: return 0
        envelopes.sort(key=lambda x: (x[0], -x[1])) # Sort: Ascend on width, descend on height if width are same.
        max_idx = 0
        heights = [envelopes[0][1]] + [0] * (len(envelopes) - 1)
        for e in envelopes: # Since width is increasing, we only need to consider height.
            idx = bisect_left(heights, e[1], hi=max_idx + 1) # Find LIS based on height
            heights[idx] = e[1] # heights has the slow increasing values overriding high values 
            max_idx = max(max_idx, idx)
        return max_idx + 1
    # https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
    def maxProfit(self, prices: List[int]) -> int: # max profit, only 1 transaction
        maxProfit = 0; curMin = prices[0] if prices else 0
        for p in prices[1:]: # scan prices, look for min, update max profit based on new current min
            curMin = min(curMin,p)
            maxProfit = max(maxProfit, p - curMin)
        return maxProfit
    # https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/discuss/75928/Share-my-DP-solution-(By-State-Machine-Thinking)
    '''There are three states, according to the action that you can take. Hence, from there, you can now the profit at a state at time i as:
    s0[i] = max(s0[i - 1], s2[i - 1]); // Stay at s0, or rest from s2
    s1[i] = max(s1[i - 1], s0[i - 1] - prices[i]); // Stay at s1, or buy from s0
    s2[i] = s1[i - 1] + prices[i]; // Only one way from s1
    Then, you just find the maximum of s0[n] and s2[n], since they will be the maximum profit we need (No one can buy stock and left with more profit that sell right :) )
    Define base case:
    s0[0] = 0; // At the start, you don't have any stock if you just rest
    s1[0] = -prices[0]; // After buy, you should have -prices[0] profit. Be positive!
    s2[0] = 0; // Lower base case
    '''
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) < 2: return 0
        s0 = s2 = 0
        s1 = -prices[0]
        for p in prices[1:]:
            last_s2 = s2
            s2 = s1 + p
            s1 = max(s0-p, s1)
            s0 = max(s0, last_s2)
        return max(s0,s2)
    # https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/discuss/54125/Very-understandable-solution-by-reusing-Problem-III-idea
    def maxProfit(self, prices: List[int]) -> int: # k = 2
        buy1 = buy2 = float('-inf')
        sell1 = sell2 = 0
        for price in prices:
            buy1 = max(buy1, -price) # max profit after 1st buy
            sell1 = max(sell1, buy1 + price) # max prpofit after 1st sale
            buy2 = max(buy2, sell1 - price) # must sell to buy 2nd time
            sell2 = max(sell2, buy2 + price)
        return sell2 # max profit from up to 2 transactions on last day
    def maxProfit(self, k: int, prices: List[int]) -> int: # for any k
        if k >= len(prices)//2: # if k >= n/2, then you can make maximum number of transactions
            profit = 0
            for i in range(1,len(prices)):
                if prices[i] > prices[i - 1]:
                    profit += prices[i] - prices[i - 1]
            return profit
        buy = [-sys.maxsize]*(k + 1)
        sell = [0]*(k + 1) # profit from prev day, ith transaction
        for price in prices:
            for i in range(1,k+1):
                buy[i] = max(buy[i], sell[i - 1] - price)
                sell[i] = max(sell[i], buy[i] + price)
        return sell[k] # max profit from up to k transactions on last day
    # https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/discuss/54131/Well-explained-Python-DP-with-comments
    def maxProfit(self, k: int, prices: List[int]) -> int: # for any k
        if k >= len(prices)//2: # if k >= n/2, then you can make maximum number of transactions
            profit = 0
            for i in range(1,len(prices)):
                if prices[i] > prices[i - 1]:
                    profit += prices[i] - prices[i - 1]
            return profit
        profits = [0]*len(prices) # profit from up to k transactions for day i
        for j in range(k):
            preprofit = 0
            for i in range(1,len(prices)):
                profit = prices[i] - prices[i-1]
                preprofit = max(preprofit+profit, profits[i]) # total profit for day i-1
                profits[i] = max(profits[i-1], preprofit)
        return profits[-1]
    # http://www.tachenov.name/2016/02/14/128/
    # https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/discuss/75924/Most-consistent-ways-of-dealing-with-the-series-of-stock-problems
    # A profitable transaction takes at least two days (buy at one day and sell at the other, provided the buying price is less than the selling price).
    # If the length of the prices array is n, the maximum number of profitable transactions is n/2 (integer division).
    # After that no profitable transaction is possible, which implies the maximum profit will stay the same.
    # Therefore the critical value of k is n/2. If the given k is no less than this value, i.e., k >= n/2,
    # we can extend k to positive infinity and the problem is equivalent to Case II.
    # https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/discuss/54113/A-Concise-DP-Solution-in-Java
    def maxProfit(self, k: int, prices: List[int]) -> int: # for any k
        def qSolve(prices):
            profit = 0
            for i in range(1,len(prices)):
                # generate profits grom price gaps between consecutive days
                if prices[i] > prices[i-1]:
                    profit += prices[i] - prices[i-1]
            return profit
        if k >= len(prices)//2: return qSolve(prices)
        # t[0][j] = dp[i][0] = 0
        t = [[0]*(len(prices)) for _ in range(k+1)]
        for i in range(1,k+1):
            tmpMax = -prices[0]
            for j in range(1,len(prices)):
                # max profit from up to j transactions with prices[0..j]
                t[i][j] = max(t[i][j-1], prices[j]+tmpMax)
                # maximum profit from up to i-1 transactions, using at most first j-1 prices, and buying the stock at price[j]
                tmpMax = max(tmpMax, t[i-1][j-1] - prices[j])
        return t[k][len(prices)-1]
    # https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/discuss/54113/A-Concise-DP-Solution-in-Java yulingtianxia
    def maxProfit(self, k: int, prices: List[int]) -> int:
        N = len(prices)
        if k >= N//2: # if k >= n/2, then you can make maximum number of transactions
            profit = 0; sell = buy = -1
            for i in range(1,N):
                if prices[i] > prices[i - 1]:
                    sell = prices[i]
                    if buy == -1: buy = prices[i-1]
                else:
                    profit += sell-buy
                    sell = buy = -1
            return profit + sell - buy if sell > buy else profit
        buy = [sys.maxsize]*(k + 1)
        profit = [0]*(k + 1)
        for price in prices:
            for i in range(1,k+1): # 1 to k buy/sell transactions
                buy[i] = min(buy[i], price - profit[i-1])
                profit[i] = max(profit[i], price - buy[i])
        return profit[k] # max profit from up to k transactions on last day
    # https://leetcode.com/problems/coin-change/discuss/77361/Fast-Python-BFS-Solution
    # compute the fewest number of coins that you need to make up that amount
    def coinChange(self, coins: List[int], amount: int) -> int:
        if amount < 1: return 0
        n = len(coins)
        dp = [0]*(amount+1) # dp[rem]: minimum number of coins to sum up to rem
        def dfs(rem: int) -> int: # rem: remaining amount after the last step
            if not rem: return 0
            if rem < 0: return -1
            if dp[rem-1]: return dp[rem-1]
            min = sys.maxsize
            for coin in coins:
                res = dfs(rem-coin)
                if 0 <= res < min:
                    min = res+1
            dp[rem-1] = -1 if min == sys.maxsize else min
            return dp[rem-1]
        return dfs(amount)
    # https://leetcode.com/problems/coin-change/discuss/77372/Clean-dp-python-code
    def coinChangeDP(self, coins: List[int], amount: int) -> int:
        if not amount: return 0
        coins.sort()
        big = coins[-1]
        if not amount%big: return amount//big
        if amount%big in coins: return amount//big+1
        dp = [0] + [amount+1]*amount # coins count
        for c in coins: # start with smallest value coin
            for amt in range(c, amount+1): # Cant just range up to next coin value
                dp[amt] = min(dp[amt], dp[amt-c]+1) # dp[c] = dp[0]+1
        return -1 if dp[-1] > amount else dp[-1]
    # Use BFS which is to find the shortest path from 0 to amount. https://leetcode.com/problems/coin-change/discuss/77361/Fast-Python-BFS-Solution
    def coinChange(self, coins: List[int], amount: int) -> int:
        if not amount: return 0
        queue = deque([(0, 0)]) # totalCoins, currVal
        visited = [True] + [False] * amount # visited amount
        while queue:
            totalCoins, currVal = queue.popleft()
            totalCoins += 1  # Take a new coin.
            for coin in coins:
                nextVal = currVal + coin
                if nextVal == amount: return totalCoins # Find a combination.
                if nextVal < amount:  # Could add more coins.
                    if not visited[nextVal]:  # Current value not checked.
                        visited[nextVal] = True  # Prevent checking again.
                        queue.append((totalCoins, nextVal)) # queue (coins count, coins value) for each coin denomination
        return -1  # Cannot find any combination.
    def coinChange(self, coins: List[int], amount: int) -> int: # use list, more concise, same as above
        if not amount: return 0
        dp = [0] + [amount+1]*amount
        for c in coins:
            for amt in range(c, amount+1):
                dp[amt] = min(dp[amt], dp[amt-c]+1)
        return -1 if dp[-1] > amount else dp[-1]
    def coinChangeBFS(self, coins: List[int], amount: int) -> int:
        if amount == 0: return 0
        coins.sort()
        big = coins[-1]
        if not amount%big: return amount//big
        if amount%big in coins: return amount//big+1
        queue = [[0, 0]]
        visited = {0}
        for curAmt, cCount in queue: # cCount: the number of coins adding to current amount
            for coin in coins:
                if curAmt + coin in visited: continue
                if curAmt + coin == amount: return cCount + 1
                elif curAmt + coin < amount:
                    queue.append([curAmt + coin, cCount + 1])
                    visited.add(curAmt + coin)
        return -1
    # https://leetcode.com/problems/coin-change/discuss/77361/Fast-Python-BFS-Solution
    def coinChange(self, coins: List[int], amount: int) -> int: # best time, concise, 
        if amount == 0: return 0
        coins.sort()
        big = coins[-1]
        if not amount%big: return amount//big
        if amount%big in coins: return amount//big+1
        level = seen = {0}
        number = 0
        while level:
            if amount in level:
                return number # return the current coin count (level number) when found amount
            level = {a+c for a in level for c in coins if a+c <= amount} - seen # BFS: new a+c's at next level (pruning seen values at previous levels)
            seen |= level
            number += 1 # number the current level
        return -1
    # https://leetcode.com/problems/coin-change-2/discuss/141076/Unbounded-Knapsack
    # Find number of combinations that make up amount
    # 1. not using the ith coin, only using the first i-1 coins to make up amount j, then we have dp[i-1][j] ways.
    # 2. using the ith coin, since we can use unlimited same coin, we need to know how many ways to make up amount
    #  j - coins[i-1] by using first i coins(including ith), which is dp[i][j-coins[i-1]], excluding coin i value: coins[i-1]
    # Initialization: dp[i][0] = 1
    def change(self, amount: int, coins: List[int]) -> int:
        dp = [1] + [0]*amount # dp[i][j] : the number of combinations to make up amount j by using the first i types of coins
        for coinI in range(1,len(coins)+1):
            for curAmt in range(1,amount+1):
                dp[coinI][curAmt] = dp[coinI-1][curAmt] + dp[coinI][curAmt-coins[coinI-1]] if curAmt >= coins[coinI-1] else 0
        return dp[len(coins)][amount]
    def change(self, amount: int, coins: List[int]) -> int:
        dp = [1] + [0]*amount # dp[i][j] : the number of combinations to make up amount j by using the first i types of coins
        for coin in coins:
            for amt in range(coin,amount+1):
                dp[amt] += dp[amt-coin]
        return dp[amount]
    # unbounded (unlimited coins for each denomination) vs bounded (only 1 of each coin denomination
    # increasing i then the previous partial result dp[i - coin] is the result that has considered coin already
    # nonrepeat: decreasing i then the previous partial result dp[i - coin] is the result that has not considered coin yet
    # return number of ways to make sum s using repeated coins
    # number of ways to make sum s using non-repeated coins
    def coinnonrep(self, coins, amount):
        dp = [1] + [0]*amount # dp[i][j] : the number of combinations to make up amount j by using the first i types of coins
        for coin in coins:
            for curAmt in range(amount,coin-1,-1):
                dp[curAmt] += dp[curAmt - coin]              
        return dp[amount]
    # number of ways to make sum s using repeated coins
    def coinrep(self, coins, s):
        dp = [1] + [0]*s # dp[0]: 1 way to make sum 0 with no coin, no way non-zero sums with no coin
        for coin in coins:
            for amt in range(coin,s+1):
                dp[amt] += dp[amt - coin]
        return dp[-1]
    # number of ways to make sum s using non-repeated coins
    def coinnonrep(self, coins, s):
        dp = [1] + [0]*s # dp[0]: 1 way to make sum 0 with no coin, no way non-zero sums with no coin
        for coin in coins:
            for amt in range(s, coin-1, -1):
                dp[amt] += dp[amt - coin]
        return dp[-1]
    def change(self, amount: int, coins: List[int]) -> int:
        # Time Complexity: O(amount * len(coins)), Space Complexity: O(amount)
        # bottom up dp solution space optimised solution
        # when amount is 0 then the answer is 1 as we have found a way to reach the current sum with some coins
        dp = [1] + [0]*amount
        # recurrence is dp[amt][idx] = dp[amt-coins[idx]][idx] + dp[amt][idx-1] with and without coins[idx]
        # iterate in increasing order of idx and amount
        for idx in range(len(coins)): # coins[idx]
            # initialise the current table
            curr_dp = [1] + [0]*amount
            for amt in range(1, amount+1):
                if amt - coins[idx] >= 0:
                    curr_dp[amt] += curr_dp[amt-coins[idx]]
                if idx >= 1:
                    curr_dp[amt] += dp[amt]
            # update the previous table with current table
            dp = curr_dp
        return dp[-1]
    def countBits(self, num: int) -> List[int]:
        dp = [0,1] + [0]*(num-1) # dp[i] number of 1's in binary i
        offset = 1
        for i in range(2,num+1):
            if offset*2 == i:
                offset *= 2
            dp[i] = dp[i-offset] + 1
        return dp
# brute force, no cache
# (1, 1, 1, 1)
# (1, 1, 2)
# (1, 2, 1)
# (1, 3)
# (2, 1, 1)
# (2, 2)
# (3, 1)
    # https://leetcode.com/problems/partition-equal-subset-sum/discuss/90592/01-knapsack-detailed-explanation
    def canPartition(self, nums: List[int]) -> bool:
        target = sum(nums)
        if target%2: return False
        target = target//2
        n = len(nums)
        dp = [[True] + [False]*target for _ in range(n+1)]    
        for i in range(1,n+1):
            for j in range(nums[i-1],target+1):
                # sum j for nums from 0 to i-1
                dp[i][j] = dp[i-1][j] or dp[i-1][j-nums[i-1]]
        return dp[n][target]
    def canPartition(self, nums: List[int]) -> bool:
        target = sum(nums)
        if target%2: return False
        target = target//2
        n = len(nums)
        dp = [True] + [False]*target
        for n in nums:
            # If we don't go from high to low however go from low to high,
            # we would end up have DP[i][j] = DP[i-1][j] or DP[i][j-num] where DP[i][j-num] (or equally dp[j-num])
            # will be refreshed at previous steps in current iteration.
            for t in range(target,n-1,-1):
                dp[t] = dp[t] or dp[t-n]
        return dp[-1]
    def canPartition(self, nums: List[int]) -> bool: # recursive, top-down, best time, space O(n*t)
        target = sum(nums)
        if target%2: return False
        target = target//2
        n = len(nums)
        dp = dict()
        def dfs(index: int, t: int) -> bool:
            if index == -1 or t > target: return False
            if nums[index] > target: return False
            if t == target: return True
            if (index,t) in dp: return dp[(index,t)]
            else:
                dp[(index,t)] = dfs(index - 1, t + nums[index]) or dfs(index - 1, t)
            return dp[(index,t)]
        return dfs(n-1, 0)
    # https://leetcode.com/problems/partition-to-k-equal-sum-subsets/discuss/180014/Backtracking-Thinking-Process See chenzuojing
    def canPartitionKSubsets(self, nums, k):
        if k==1: return True
        self.n=len(nums)
        if k>self.n: return False
        total=sum(nums)
        if total%k: return False
        self.target=total/k
        visit=[0]*self.n
        nums.sort(reverse=True)
        def dfs(k,ind,sum,cnt):
            if k==1: return True
            if sum==self.target and cnt>0:
                return dfs(k-1,0,0,0)
            for i in range(ind,self.n):
                if not visit[i] and sum+nums[i]<=self.target:
                    visit[i]=1
                    if dfs(k,i+1,sum+nums[i],cnt+1): 
                        return True
                    visit[i]=0
            return False
        return dfs(k,0,0,0)
    # https://leetcode.com/problems/partition-to-k-equal-sum-subsets/discuss/146579/Easy-python-28-ms-beats-99.5
    # if sums[j] == 0: break
    # The key is, sums[j] == 0 means for all k > j, sum[k] == 0; because this algorithm always fill the previous buckets before trying the next.
    # So if by putting nums[i] in this empty bucket can't solve the game, putting nums[i] on other empty buckets can't solve the game too.
    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        bins = [0]*k
        # bin = [[] for _ in range(k)]
        n = len(nums)
        if not nums or n < k: return False
        target = sum(nums)
        if target%k: return False
        target = target//k
        nums.sort(reverse=True) # avoid TLE
        if nums[0] > target: return False
        def dfs(i):
            # nonlocal bin, bins
            if i == n:
                # print(bin, bins)
                return len(set(bins)) == 1 # processed last number, all bins should have target value
            for j in range(k):
                bins[j] += nums[i] # Recursive add each number to jth (1st,2nd,...) bin till reaching, then repeat for next bin
                if bins[j] <= target and dfs(i+1):
                    # bin[j].append(nums[i])
                    # if bins[j] == target:
                    #     print(i, nums[i], j, bins[j], bin[j])
                    return True # keep on filling current bin up to target then move to next bin
                bins[j] -= nums[i] # backtrack if bin value exceeds target
                if bins[j] == 0:
                    break # Exceeded target: Can't fill the 1st bin (find 1st subset of target), return False
            return False
        return dfs(0) # start with 1st largest number
    # my version, sorting nums and start from largest number to speed up
    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool: # best time, more intuitive, find 1 subset before looking for next subset
        n = len(nums)
        if not nums or n < k: return False
        target = sum(nums)
        if target%k: return False
        target = target//k
        used = set()
        nums.sort() # don't need, but speed up 30%
        def dfs(index: int, t: int, k: int) -> bool:
            if k==1: # Dont need to compute the last subset of target, which are the remaining 
                return True
            if target == t:
                return dfs(n-1,0,k-1) # found 1 subset, marked used, find next subset with k--
            if t > target:
                return False # trigger backtracking
            for i in range(index,-1,-1):
                if i not in used:
                    used.add(i)
                    if dfs(i-1, t + nums[i],k):
                        print(k,i,nums[i])
                        return True
                    used.remove(i) # backtracking
                    if t == 0: break # Cant find 1st subset of target, break fro loop, return False. This avoid useless run
            return False
        return dfs(n-1,0,k) # start from largest number
    # Topological sort: https://leetcode.com/problems/course-schedule/discuss/58586/Python-20-lines-DFS-solution-sharing-with-explanation
    def canFinish(self, numCourses, prerequisites): # iterative
        if numCourses <= 0:
            return False
        inDegree = inDegree = {i : 0 for i in range(numCourses)} # parent count
        graph = {i : [] for i in range(numCourses)} # courses -> dependents
        
        for child, parent in prerequisites:
            graph[parent].append(child)
            inDegree[child] += 1 # number of prereqs

        sources = deque()
        
        for key in inDegree:
            if inDegree[key] == 0: # start visiting courses with no prereq
                sources.append(key)
        visited = 0 # count courses taken   
        while sources:
            vertex = sources.popleft()
            visited += 1 # visit parent (prereq)
            for child in graph[vertex]:
                inDegree[child] -= 1 # decrement dependent children inDegree count
                if inDegree[child] == 0: # visit on inDegree == 0
                    sources.append(child)
        
        return visited == numCourses
    # using heap
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        if len(prerequisites) == 0: # no prereqs
            return True
        self.prereqs = dict() # courses to prereqs
        self.deps = dict() # courses to dependents
        heap, heap2 = [], [] # heap2 for deferred scheduling due to prereqs
        visited = set([numC for numC in range(numCourses)]) # courses taken
        for c,prereq in prerequisites:
            if c in self.deps and prereq in self.deps[c]:
                return False # 2-node prereq loop
            self.prereqs.setdefault(c,[]).append(prereq)
            self.deps.setdefault(prereq,[]).append(c)
            if c in visited: visited.remove(c) # remove courses having prereqs
        for numC in visited: # There must be at least 1 course without prereq
            heappush(heap, (0,-len(self.deps.get(numC,[])),numC))
        while(heap):
            x, y, numC = heappop(heap) # (2, -4, 233) prereq (3, -2, 212)
            if numC in self.prereqs and set(self.prereqs[numC]) - visited: # numC has unvisited/untaken prereqs
                heappush(heap2, (x, y, numC))
                continue
            visited.add(numC)
            for d in self.deps.get(numC, []):
                heappush(heap, (len(self.prereqs[d]),-len(self.deps.get(d,[])),d))
            if not heap:
                heap = heap2 # defer til prereqs visited
        return len(visited) == numCourses
    # if node v has not been visited, then mark it as 0.
    # if node v is being visited, then mark it as -1. If we find a vertex marked as -1 in DFS, then there is a ring.
    # if node v has been visited, then mark it as 1. If a vertex was marked as 1, then no ring contains v or its successors.
    # time efficiency is O(V^2+VE), because each dfs in adjacency list is O(V+E) and we do it V times
    # for i in range(numCourses): O(V) . * dfs() O(V+E)
    # Space efficiency is O(E).
    def canFinish(self, numCourses, prerequisites): # best time
        graph = [[] for _ in range(numCourses)]
        visit = [0 for _ in range(numCourses)]
        for x, y in prerequisites:
            graph[x].append(y) # prereq edges
        def dfs(i):
            if visit[i] == -1:
                return False
            if visit[i] == 1:
                return True
            visit[i] = -1
            for j in graph[i]: # recurse all prereqs courses
                if not dfs(j): 
                    return False
            visit[i] = 1
            return True
        for i in range(numCourses):
            if not dfs(i):
                return False
        return True
    def canFinish(self, N, edges):
        pre = defaultdict(list)
        for x, y in edges: pre[x].append(y)
        status = [0] * N
        def canTake(i):
            if status[i] in {1, -1}: return status[i] == 1
            status[i] = -1
            if any(not canTake(j) for j in pre[i]): return False
            status[i] = 1
            return True
        return all(canTake(i) for i in range(N))
    # https://leetcode.com/problems/course-schedule/discuss/162743/JavaC%2B%2BPython-BFS-Topological-Sorting-O(N-%2B-E)
    def canFinish(self, n, prerequisites):
        G = [[] for i in range(n)] # space(V+E)
        degree = [0] * n
        for j, i in prerequisites:
            G[i].append(j)
            degree[j] += 1
        bfs = [i for i in range(n) if degree[i] == 0]
        for i in bfs: # O(V+E)
            for j in G[i]:
                degree[j] -= 1
                if degree[j] == 0:
                    bfs.append(j)
        # return len(bfs) == n
    # https://leetcode.com/problems/course-schedule-ii/submissions/
        return bfs if len(bfs) == n else []
    # https://leetcode.com/problems/minimum-height-trees/discuss/76055/Share-some-thoughts
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        if n < 3: return [i for i in range(n)]
        G = [set() for _ in range(n)]
        for v1,v2 in edges:
            G[v1].add(v2)
            G[v2].add(v1)
        leaves = [v for v in range(n) if len(G[v]) == 1]
        while n > 2:
            n -= len(leaves)
            newLeaves = []
            for v in leaves: # trim leaves, build the new leaves till there are <=2 nodes left 
                v1 = G[v].pop()
                G[v1].remove(v)
                if len(G[v1]) == 1:
                    newLeaves.append(v1)
            leaves = newLeaves
        return leaves
    # https://leetcode.com/problems/minimum-height-trees/discuss/76052/Two-O(n)-solutions
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]: # RecursionError: maximum recursion depth exceeded
        neighbors = defaultdict(set)
        for v, w in edges:
            neighbors[v].add(w)
            neighbors[w].add(v)
        def maxpath(v, visited):
            visited.add(v)
            paths = [maxpath(w, visited) for w in neighbors[v] if w not in visited]
            path = max(paths or [[]], key=len)
            path.append(v)
            return path
        path = maxpath(0, set())
        path = maxpath(path[0], set())
        m = len(path)
        return path[(m-1)//2:m//2+1]
    #  dp 
    def findMinHeightTrees(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: List[int]
        """
        if n <= 2:
            return [i for i in range(0, n)]
            
        def dfs(vi, dp, graph, parent, sdp):
            maxLength = -9999999999999
            smaxLength = -9999999999999
            for end in graph[vi]:
                if end == parent:
                    continue
                if dp[end] == -1:
                    height = dfs(end, dp, graph, vi, sdp) + 1
                else:
                    height = dp[end] + 1
                if maxLength <= height:
                    smaxLength = maxLength
                    maxLength = height
                elif smaxLength <= height:
                    smaxLength = height
            if len(graph[vi]) >= 2:
                sdp[vi] = smaxLength
            dp[vi] = maxLength
            return max(dp[vi], 0)
        
        def calcdfs(vi, dp, sdp, ldp, acc, parent, graph):
            ldp[vi] = max(dp[vi], acc)
            for end in graph[vi]:
                if end == parent:
                    continue
                if dp[vi] == dp[end] + 1:
                    height = sdp[vi] + 1
                else:
                    height = dp[vi] + 1
                newacc = max(acc + 1, height)
                calcdfs(end, dp, sdp, ldp, newacc, vi, graph)

        dp = [-1 for _ in range(0, n)]
        sdp = [-9999999999999 for _ in range(0, n)]
        ldp = [9999999999999 for _ in range(0, n)]

        graph = {}
        
        for edge in edges:
            start, end = edge
            graph[start] = graph.get(start, []) + [end]
            graph[end] = graph.get(end, []) + [start]
            
        for i in range(0, n):
            if dp[i] == -1:
                dfs(i, dp, graph, -1, sdp)

        calcdfs(0, dp, sdp, ldp, 0, -1, graph)

        ret = []
        minDist = dp[0]
        minPos = 0
        for i in range(0, n):
            if minDist >= ldp[i]:
                minDist = ldp[i]
                minPos = i
        ret.append(minPos)
        for i in range(0, n):
            if minDist == ldp[i] and i != minPos:
                ret.append(i)
                break
        return ret
    # my find 2 longest paths
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        if n < 3: return [i for i in range(n)]
        G = [set() for _ in range(n)]
        for v1,v2 in edges:
            G[v1].add(v2)
            G[v2].add(v1)
        leaves = [v for v in range(n) if len(G[v]) == 1]
        paths = [[] for _ in range(n+1)]
        longest = 0
        def discoverPaths(q: Deque) -> None:
            nonlocal longest
            while q:
                p = q.popleft()
                n = p[-1]
                for i in G[n]:
                    if i not in p:
                        path = p + [i]
                        if len(G[i]) == 1:
                            paths[len(path)].append(path)
                            longest = max(longest,len(path))
                        else:
                            q.append(path)
        qu = deque([[leaves[0]]])
        discoverPaths(qu)
        qu.append([paths[longest][-1][-1]])
        discoverPaths(qu)
        roots = set()
        for path in paths[longest]:
            mid = len(path)//2
            if len(path)%2:
                roots.add(path[mid]) 
            else:
                roots.update(path[mid-1:mid+1])
        return list(roots)
    # https://leetcode.com/problems/pacific-atlantic-water-flow/discuss/90733/Java-BFS-and-DFS-from-Ocean
    def pacificAtlantic(self, grid: List[List[int]]) -> List[List[int]]: # recursive
        if not grid or len(grid) == 0 or len(grid[0]) == 0: return 0
        maxRow, maxCol = len(grid)-1, len(grid[0])-1
        pacific = [[False]*len(grid[0]) for _ in range(len(grid))] # True can flow to pacific, False cant
        atlantic = [[False]*len(grid[0]) for _ in range(len(grid))] # True can flow to atlantic, False cant
        dir = [[1,0],[-1,0],[0,1],[0,-1]]
        def dfs(visited: List[List[bool]], height: int, x: int, y: int):
            if x<0 or x>maxRow or y<0 or y>maxCol or visited[x][y] or grid[x][y] < height: return
            visited[x][y] = True
            for d in dir:
                dfs(visited, grid[x][y], x+d[0], y+d[1])
        for i in range(len(grid)):
            dfs(pacific, -1, i, 0)
            dfs(atlantic, -1, i, maxCol)
        for i in range(len(grid[0])):
            dfs(pacific, -1, 0, i)
            dfs(atlantic, -1, maxRow, i)
        pos = []
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if pacific[i][j] and atlantic[i][j]:
                    pos.append([i,j])
        return pos
    def pacificAtlantic(self, grid: List[List[int]]) -> List[List[int]]: # iterative using queue, better time, #Rows x #Cols
        if not grid or len(grid) == 0 or len(grid[0]) == 0: return 0
        maxRow, maxCol = len(grid)-1, len(grid[0])-1
        pacific = [[False]*len(grid[0]) for _ in range(len(grid))] # True can flow to pacific, False cant
        atlantic = [[False]*len(grid[0]) for _ in range(len(grid))] # True can flow to atlantic, False cant
        for i in range(len(grid)):
            pacific[i][0] = atlantic[i][maxCol] = True # edge cells can flow into ocean
        for j in range(len(grid[0])):
            atlantic[maxRow][j] = pacific[0][j] = True
        pacQ = deque([[0,y] for y in range(len(grid[0]))]) # start from True edge cells
        pacQ.extend([[x,0] for x in range(len(grid))])
        atlQ = deque([[x, maxCol] for x in range(len(grid))])
        atlQ.extend([[maxRow, y] for y in range(len(grid[0]))])
        direction = [[1,0],[-1,0],[0,1],[0,-1]] # compute neighboring cells
        def bfs(q: Deque[List[int]], visited: List[List[bool]]):
            nonlocal maxRow, maxCol
            while q:
                cur = q.popleft() # process cells from head of queue
                for d in direction: # compute 4 neighbors
                    x, y = cur[0]+d[0], cur[1]+d[1]
                    if x<0 or x>maxRow or y<0 or y>maxCol or visited[x][y] or grid[x][y] < grid[cur[0]][cur[1]]:
                        continue # if neighbor is lower, leave it as False
                    visited[x][y] = True # it can flow toward edge
                    q.append([x,y]) 
        bfs(pacQ,pacific) # update pacific cells
        bfs(atlQ,atlantic)
        pos = []
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if pacific[i][j] and atlantic[i][j]:
                    pos.append([i,j])
        return pos
    # https://leetcode.com/problems/longest-increasing-path-in-a-matrix/submissions/
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int: # 2-d kadane
        if not matrix: return 0
        directions = [(1,0),(-1,0),(0,1),(0,-1)]
        m, n = len(matrix), len(matrix[0])
        cache = [[-1 for _ in range(n)] for _ in range(m)]
        res = 0
        def dfs(i, j):
            if cache[i][j] != -1: return cache[i][j] # longest length starting at this cell
            res = 1 # include this cell
            for direction in directions:
                x, y = i + direction[0], j + direction[1]
                if x < 0 or x >= m or y < 0 or y >= n or matrix[x][y] <= matrix[i][j]:
                    continue # skip neighbors whose are less than current cell
                length = 1 + dfs(x, y) # include current, dfs higher value neighbors
                res = max(length, res)
            cache[i][j] = res
            return res
        for i in range(m): # MxN
            for j in range(n):
                cur_len = dfs(i, j)
                res = max(res, cur_len)
        return res
    # https://leetcode.com/problems/clone-graph/discuss/42314/Python-solutions-(BFS-DFS-iteratively-DFS-recursively).
    def cloneGraph(self, node): #DFS iteratively using stack
        if not node: return node
        m = {node: Node(node.val)} # create root
        stack = [node] # has nodes that requires neighbors
        while stack:
            n = stack.pop()
            for neigh in n.neighbors:
                if neigh not in m:
                    stack.append(neigh) # push neighbors into stack to fill neighbors' neighbors later
                    m[neigh] = Node(neigh.val) # create neighbors with missing neighbors
                m[n].neighbors.append(m[neigh]) # fill neighbors for current node n
        return m[node]
    def cloneGraph(self, node): # BFS interatively using queue
        if not node: return node
        m = {node: Node(node.val)} # cache of originals and copies
        dequ = deque([node]) # nodes in dequ have been cloned, but neighbors have yet to be updated
        while dequ:
            n = dequ.popleft()
            for neigh in n.neighbors:
                if neigh not in m:
                    dequ.append(neigh)
                    m[neigh] = Node(neigh.val) # clone neighbor for reference next
                m[n].neighbors.append(m[neigh])
        return m[node]
    def cloneGraph(self, node): # DFS recursively
        if not node: return node
        m = {node: Node(node.val)} # cache of originals and copies
        def dfs(node, m):
            for neigh in node.neighbors:
                if neigh not in m: # clone neighbors
                    m[neigh] = Node(neigh.val) # create neighbors
                    dfs(neigh, m) # dfs to create neighbor's neighbors
                m[node].neighbors.append(m[neigh]) # append created neighbors
        dfs(node, m) # root and cache
        return m[node]
    def cloneGraph(self, node: Node) -> Node: # recursive, fastest, concise
        def clone(node):
            cloned[node.val] = Node(node.val)
            cloned[node.val].neighbors = list(map(lambda n: cloned[n.val] if n.val in cloned else clone(n), node.neighbors))
            return cloned[node.val]
        cloned = {}
        return clone(node) if node else None
    # https://www.educative.io/m/search-rotated-array
    # at least one half of the array is always sorted. If the number ‘n’ lies within the sorted half of the array,
    # then our problem is a basic binary search.
    # Otherwise, discard the sorted half and keep examining the unsorted half.
    # Since we are partitioning the array in half at each step, this gives us O(log n) runtime complexity.
    # assuming all the keys are unique.
    def binary_search_rotated(arr, key): # logN
        def binary_search(arr, start, end, key):
            if (start > end): return -1
            mid = (start + end) >>1
            if arr[mid] == key: return mid # after this, arr[mid] != key
            if arr[start] <= arr[mid] and key < arr[mid] and key >= arr[start]: # key in sorted first half
                return binary_search(arr, start, mid - 1, key)
            elif arr[mid] <= arr[end] and key > arr[mid] and key <= arr[end]: # key in sorted 2nd half
                return binary_search(arr, mid + 1, end, key)
            elif arr[end] <= arr[mid]: return binary_search(arr, mid + 1, end, key) # search key in unsorted 2nd half
            elif arr[start] >= arr[mid]: return binary_search(arr, start, mid - 1, key) # search key in unsorted 1st half
            return -1
        return binary_search(arr, 0, len(arr)-1, key)
    # https://leetcode.com/problems/search-in-rotated-sorted-array/discuss/14435/Clever-idea-making-it-simple
    def search(self, nums: List[int], target: int) -> int: # 2 pointers i/j linear search, speed up finding target for rotated sorted list
        if nums: # O(N)
            mid = len(nums)//2
            for i in range(mid+1): # moving i from 0 to mid
                if nums[i] == target: return i
            for j in range(len(nums)-1,mid, -1): # moving j from end to mid
                if nums[j] == target: return j
        return -1
    def search(self, nums: List[int], target: int) -> int: # return True if target in nums
        if not nums: return False
        pivot = nums[0]
        if target == pivot: return True
        # we move lo and hi so pivot will never equal to lo or hi
        lo, hi = 0, len(nums)-1
        while hi >= 0 and nums[hi] == pivot: hi -= 1
        while lo <= len(nums) - 1 and nums[lo] == pivot: lo += 1
        if len(nums[lo:hi+1]) < 3: return target in nums
        while lo <= hi:
            mid = (hi - lo) // 2 + lo
            if nums[mid] == target:
                return True
            if nums[mid] < pivot:
                # mid is on the upper side
                if nums[mid] < target < pivot:
                    lo = mid + 1
                else:
                    hi = mid - 1
            if nums[mid] > pivot:
                if pivot < target < nums[mid]:
                    hi = mid - 1
                else:
                    lo = mid + 1
        return False
    # https://leetcode.com/problems/search-in-rotated-sorted-array-ii/discuss/28218/My-8ms-C%2B%2B-solution-(o(logn)-on-average-o(n)-worst-case)
    # everytime check if targe == nums[mid], if so, we find it.
    # otherwise, we check if the first half is in order (i.e. nums[left]<=nums[mid]) 
    # and if so, go to step 3), otherwise, the second half is in order,   go to step 4)
    # 3. check if target in the range of [left, mid-1] (i.e. nums[left]<=target < nums[mid]), if so, do search in the first half, i.e. right = mid-1; otherwise, search in the second half left = mid+1;
    # 4. check if target in the range of [mid+1, right] (i.e. nums[mid]<target <= nums[right]), if so, do search in the second half, i.e. left = mid+1; otherwise search in the first half right = mid-1;
    # The only difference is that due to the existence of duplicates, we can have nums[left] == nums[mid] and in that case,
    # the first half could be out of order (i.e. NOT in the ascending order, e.g. [3 1 2 3 3 3 3]) and we have to deal this case separately.
    # In that case, it is guaranteed that nums[right] also equals to nums[mid], so what we can do is to check if nums[mid]== nums[left] == nums[right] before the original logic, and if so, we can move left and right both towards the middle by 1. and repeat.
    def search(self, nums: List[int], target: int) -> int:
        left, right =  0, len(nums)-1
        while left<=right:
            mid = (left + right) >> 1
            if nums[mid] == target: return True
            # the only difference from the first one, trickly case, just update left and right
            if nums[left] == nums[mid]: left += 1
            # elif nums[mid] == nums[right]: right -= 1
            elif nums[left] <= nums[mid]: # 1st half is ordered
                if nums[left] <=target < nums[mid]: right = mid-1
                else: left = mid + 1 
            else: # 2nd half is ordered
                if nums[mid] < target <= nums[right]: left = mid+1
                else: right = mid-1
        return False
    # https://leetcode.com/problems/search-a-2d-matrix/discuss/26248/6-12-lines-O(log(m)-%2B-log(n))-myself%2Blibrary
    # Treat the matrix like a single big list of length m*n and use a simple binary search. Convert the list indexes to matrix indexes on the fly.
    def searchMatrix(self, matrix, target):
        n = len(matrix[0])
        lo, hi = 0, len(matrix) * n
        while lo < hi:
            mid = (lo + hi) / 2
            x = matrix[mid/n][mid%n]
            if x < target:
                lo = mid + 1
            elif x > target:
                hi = mid
            else:
                return True
        return False
    # use bisect to (approximately) find the candidate row and then bisect_left to find the candidate cell in that row.
    def searchMatrix(self, matrix, target):
        i = bisect(matrix, [target])
        if i < len(matrix) and matrix[i][0] == target:
            return True
        row = matrix[i-1]
        j = bisect_left(row, target)
        return j < len(row) and row[j] == target
    # 1 line with bisect
    def searchMatrix(self, matrix, target):
        return bool(matrix) and target in matrix[bisect(matrix, [target + 0.5]) - 1]
    def searchMatrix(self, matrix, target):
        m = bisect(matrix, [target+0.5])
        return len(matrix[0]) > 0 and matrix[m - 1][bisect(matrix[m - 1], target) - 1] == target if m else False
    def searchMatrix(self, matrix, target):
        m = bisect(matrix, [target + 0.5]) #log(R)
        if m:
            n = bisect(matrix[m - 1], target) #log(C)
            return len(matrix[0]) > 0 and matrix[m - 1][n - 1] == target
        return False
    # https://leetcode.com/problems/search-a-2d-matrix-ii/
    def searchMatrix2(self, matrix, target):
        for r in matrix:
            if r[0] <= target <= r[-1]:
                if r[bisect_left(r, target)] == target:
                    return True
        return False
    # https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if nums:
            i = j = bisect_left(nums, target)
            if i != len(nums) and nums[i] == target:
                while j+1 < len(nums) and nums[j+1] == target:
                    j += 1
                return [i,j]
        return [-1,-1]
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if nums:
            i = j = bisect_left(nums, target)
            if i != len(nums) and nums[i] == target:
                j = bisect_right(nums, target, lo=j)
                return [i,j-1]
        return [-1,-1]
    # https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/discuss/14707/9-11-lines-O(log-n)
    def searchRange(self, nums, target):
        def search(n):
            lo, hi = 0, len(nums)
            while lo < hi:
                mid = (lo + hi) // 2
                if nums[mid] >= n:
                    hi = mid
                else:
                    lo = mid + 1
            return lo
        lo = search(target)
        return [lo, search(target+1)-1] if target in nums[lo:lo+1] else [-1, -1]
    # https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/discuss/48491/1-2-lines-RubyPython
    # Use binary search to find the first number that's less than or equal to the last.
    # https://stackoverflow.com/questions/44937234/object-does-not-support-indexing-when-overriding-getitem
    # https://medium.com/@beef_and_rice/mastering-pythons-getitem-and-slicing-c94f85415e1c
    __getitem__ = lambda self, i: \
        self.nums[i] <= self.nums[-1]
    def findMin(self, nums):
        self.nums = nums
        return nums[bisect(self, False, 0, len(nums))]
    # https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/discuss/158940/Beat-100%3A-Very-Simple-(Python)-Very-Detailed-Explanation
    def findMin(self, nums: List[int]) -> int:
        left, right = 0, len(nums)-1
        while left < right: # left and right both converge to the minimum index; avoid left <= right, because that would loop forever
            mid = (left + right) // 2 # or mid = left + (right - left) // 2 to avoid overflow in Java/C
            # checks to converge the left and right bounds on the start of the pivot, never disqualify the index for a possible minimum value.
            # in normal binary search, we have a target to match exactly, and would have a specific branch for if nums[mid] == target.
            # we do not have a specific target here, so we just have simple if/else.
            if nums[mid] > nums[right]: # example:  [3,4,5,6,7,8,9,1,2], pivot must be to the right of the middle
                left = mid + 1
            else: # nums[mid] <= nums[right], pivot must be at mid or to the left of mid, example: [8,9,1,2,3,4,5,6,7]
                # right = mid
                # num[mid] == num[hi], we couldn't sure the position of minimum in mid's left or right, so just let upper bound reduce one.
                right = mid if nums[right] != nums[mid] else right - 1 # nums has duplicates https://leetcode.com/problems/find-minimum-in-rotated-sorted-array-ii/discuss/48908/Clean-python-solution
        # at this point, left and right converge to a single index (for minimum value) since our if/else forces the bounds of left/right to shrink each iteration:
        # when left bound increases, it does not disqualify a value that could be smaller than something else (we know nums[mid] > nums[right],
        # so nums[right] wins and we ignore mid and everything to the left of mid).
        # when right bound decreases, it also does not disqualify a value that could be smaller than something else (we know nums[mid] <= nums[right],
        # so nums[mid] wins and we keep it for now). so we shrink the left/right bounds to one value, without ever disqualifying a possible minimum
        return nums[left]
    # https://leetcode.com/problems/sqrtx/
    def sqrt(self, A):
        if A < 2: return A
        ans = start = 0; end = A
        while start <= end:
            mid = start + (end - start)//2
            if mid <= A//mid:
                start = mid + 1
                ans = mid
            else:
                end = mid - 1
        return ans
    def findPeakElement(self, nums): # linear search
        if nums[i] < nums[i-1]: return i-1
        return len(nums)-1
    def findPeakElement(self, nums): # iterative Binary Search
        l, r = 0, len(nums)-1
        while l < r:
            mid1 = (l+r)//2
            mid2 = mid1+1
            if nums[mid1] < nums[mid2]: l = mid2
            else: r = mid1
        return l
    def findPeakElement(self, nums): # recursive Binary Search
        def Helper(l, r):
            if l == r: return l
            mid1 = (l+r)//2
            mid2 = mid1+1
            if nums[mid1] > nums[mid2]: return Helper(l, mid1)
            else: return Helper(mid2, r)
        return Helper(0, len(nums)-1)
    def findPeakElement(self, nums):
        if len(nums) < 4: return nums.index(max(nums))
        l, r = 0, len(nums)-1
        while l < r:
            mid = (l+r)//2
            if nums[mid] < nums[mid-1]: r = mid-1
            elif nums[mid] < nums[mid+1]: l = mid+1
            else:
                return mid
        return l
    # https://leetcode.com/problems/count-of-range-sum/discuss/78006/Summary-of-the-Divide-and-Conquer-based-and-Binary-Indexed-Tree-based-solutions
    def countRangeSum(self, nums: List[int], lower: int, upper: int) -> int:
        preSum = [0]*(len(nums)+1)
        for i in range(len(nums)):
            preSum[i+1] = preSum[i] + nums[i]
        count = 0
        for i in range(len(preSum)):
            for j in range(i+1,len(preSum)): # Time n(n+1)/2
                if lower <= (preSum[j] - preSum[i]) <= upper:
                    count += 1
        return count
    # https://massivealgorithms.blogspot.com/2016/01/leetcode-count-of-range-sum.html
    # The merge sort based solution counts the answer while doing the merge.
    # During the merge stage, we have already sorted the left half [start, mid) and right half [mid, end).
    # We then iterate through the left half with index i. For each i, we need to find two indices k and j in the right half where
    # j is the first index satisfy S_j - S_i > upper;
    # k is the first index satisfy S_k - S_i >= lower.
    # Then the number of sums in [lower, upper] is j-k.
    # We also use another index t to copy the elements satisfy S_t < S_i to a cache in order to complete the merge sort.
    # Despite the nested loops, the time complexity of the merge & count stage is still linear.
    # Because the indices k, j, t will only increase but not decrease.
    # Therefore, each of them will traversal the right half once at most.
    # The total time complexity of this divide and conquer solution is then O(n \log n).
    def countRangeSum(self, nums: List[int], lower: int, upper: int) -> int:
        preSum = [0]
        for n in nums: preSum.append(preSum[-1] + n)
        def countWhileMergeSort(nums: List[int], start: int, end: int, lower: int, upper: int) -> int: # nlogn
            if end - start <= 1: return 0
            j = k = t = mid = (start + end)//2
            count = countWhileMergeSort(nums, start, mid, lower, upper) + countWhileMergeSort(nums, mid, end, lower, upper)
            cache = []
            for i in range(start,mid):
                while k < end and nums[k] - nums[i] < lower: k += 1
                while j < end and nums[j] - nums[i] <= upper: j += 1
                while t < end and nums[t] < nums[i]:
                    cache.append(nums[t])
                    t += 1
                cache.append(nums[i])
                count += j - k
            nums[start:t] = cache[:t]
            return count
        return countWhileMergeSort(preSum, 0, len(preSum), lower, upper)
    # https://leetcode.com/problems/count-of-range-sum/discuss/78006/Summary-of-the-Divide-and-Conquer-based-and-Binary-Indexed-Tree-based-solutions
    def countRangeSum(self, nums: List[int], lower: int, upper: int) -> int:
        if not nums or len(nums) == 0 or lower > upper: return 0
        def findIndex(preSum, val):
            l, r = 0, len(preSum) - 1
            while l <= r:
                m = l + (r-l)//2
                if preSum[m] <= val: l = m+1
                else: r = m-1
            return l
        def countRangeSumSub(l, r):
            if l == r: return 1 if nums[l] >= lower and nums[r] <= upper else 0
            preSum = []
            sum = 0
            m = l + (r-l)//2
            for i in range(m+1, r+1, 1): # prepare preSum for right half
                sum += nums[i]
                preSum.append(sum)
            preSum.sort() # sort for binary search later
            sum = count = 0
            for i in range(m,l-1,-1):
                sum += nums[i] # suffix sum on the left half
                # count += findIndex(preSum, upper - sum + 0.5) - findIndex(preSum, lower - sum - 0.5)
                count += bisect_left(preSum, upper - sum + 0.5) - bisect_left(preSum, lower - sum - 0.5) # faster than findIndex by 2/3
            return countRangeSumSub(l, m) + countRangeSumSub(m+1, r) + count
        return countRangeSumSub(0, len(nums)-1)
    # https://leetcode.com/problems/count-of-range-sum/discuss/77991/Short-and-simple-O(n-log-n)
    def countRangeSum(self, nums, lower, upper):
        first = [0]
        for num in nums: first.append(first[-1] + num)
        def sort(lo, hi):
            mid = (lo + hi) // 2
            if mid == lo:
                return 0
            count = sort(lo, mid) + sort(mid, hi)
            i = j = mid
            for left in first[lo:mid]:
                while i < hi and first[i] - left <  lower: i += 1
                while j < hi and first[j] - left <= upper: j += 1
                count += j - i
            first[lo:hi] = sorted(first[lo:hi])
            return count
        return sort(0, len(first))
    # Binary search
    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        n = len(letters)
        if n == 0: return None
        low, high = 0, n - 1
        while low <= high: # <= to look beyond target
            mid = (low + high) // 2
            if letters[mid] > target: high = mid - 1
            else: low = mid + 1
        return letters[low%n] # If it can not be found, must be the first element (wrap around) 
    # https://leetcode.com/problems/find-smallest-letter-greater-than-target/discuss/110005/Easy-Binary-Search-in-Java-O(log(n))-time
    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        n = len(letters)
        # hi starts at 'n' rather than the usual 'n - 1'. 
        # because the terminal condition is 'lo < hi' and if hi starts from 'n - 1', we can never consider value at index 'n - 1'
        lo, hi = 0, n
        # Terminal condition is 'lo < hi', to avoid infinite loop when target is smaller than the first element
        while lo < hi:
            mid = (lo + hi) // 2;
            if letters[mid] > target: hi = mid
            else: lo = mid + 1 # letters[mid] <= x
        return letters[lo % n] # Because lo can end up pointing to index 'n', in which case we return the first element
    # https://leetcode.com/problems/find-smallest-letter-greater-than-target/discuss/110030/Python-no-brainer!-U0001f921
    def nextGreatestLetter(self, letters, target):
        return next((c for c in letters if c > target), letters[0])
    def nextGreatestLetter(self, letters, target):
        return letters[bisect(letters, target) % len(letters)]
    def nextGreatestLetter(self, letters, target):
        return (letters * 2)[bisect.bisect(letters, target)]
    # https://leetcode.com/problems/binary-search/submissions/ 
    def search(self, nums: List[int], target: int) -> int: # best time
        n = len(nums)
        low, high = 0, n - 1
        while low < high:
            mid = (low + high) // 2
            if nums[mid] < target: low = mid + 1
            else: high = mid
        return low if nums[low] == target else -1
    # https://leetcode.com/problems/binary-search/discuss/148840/Python-typical-solutions-beat-100
    def search(self, nums: List[int], target: int) -> int:
        index = bisect_left(nums, target)
        return index if index < len(nums) and nums[index] == target else -1
    def search(self, nums, target):
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = (l + r) // 2
            if nums[mid] < target: l = mid + 1
            elif nums[mid] > target: r = mid - 1
            else: return mid
        return -1
    # https://leetcode.com/problems/peak-index-in-a-mountain-array/submissions/
    def peakIndexInMountainArray(self, nums: List[int]) -> int:
        if len(nums) == 3: return 1
        l, r = 0, len(nums)-1
        while l < r:
            mid = (l+r)//2
            if nums[mid] < nums[mid-1]: r = mid-1
            elif nums[mid] < nums[mid+1]: l = mid+1
            else:
                return mid
        return l
    # https://leetcode.com/problems/peak-index-in-a-mountain-array/discuss/139848/C%2B%2BJavaPython-Better-than-Binary-Search
    def peakIndexInMountainArray(self, A):
        return A.index(max(A))
    def peakIndexInMountainArray(self, A):
        for i in range(1,len(A)):
            if A[i] > A[i + 1]: return i
    def peakIndexInMountainArray(self, A): # Binary search logN
        l, r = 0, len(A) - 1
        while l < r:
            m = (l + r) // 2
            if A[m] < A[m + 1]: # rising slope
                l = m + 1
            else: # down slope
                r = m
        return l # peak index
    # faster than binary search: Approach 4, Golden-section search
    # It's guarenteed only one peak, we can apply golden-section search.
    # https://stackoverflow.com/questions/4247111/is-golden-section-search-better-than-binary-search
    def peakIndexInMountainArray(self, A):
        def gold1(l, r):
            return l + int(round((r - l) * 0.382))
        def gold2(l, r):
            return l + int(round((r - l) * 0.618))
        l, r = 0, len(A) - 1
        x1, x2 = gold1(l, r), gold2(l, r)
        while x1 < x2:
            if A[x1] < A[x2]:
                l = x1
                x1 = x2
                x2 = gold1(x1, r)
            else:
                r = x2
                x2 = x1
                x1 = gold2(l, x2)
        return A.index(max(A[l:r + 1]), l)
    # https://leetcode.com/problems/find-in-mountain-array/discuss/317607/JavaC%2B%2BPython-Triple-Binary-Search
    #  Find peak, find target on left slope, if not found, on right slope. Cache get data in case we need it again
    def findInMountainArray(self, target: int, mountain_arr: 'MountainArray') -> int:
        lenA = mountain_arr.length()
        l, r, A = 0, lenA - 1, [-1]*lenA
        while l < r:
            m = (l + r) // 2
            A[m], A[m+1] = mountain_arr.get(m), mountain_arr.get(m+1)
            if A[m] < A[m + 1]: # rising slope
                l = m + 1
            else: # down slope
                r = m
        peakI = l # peak index Binary search logN
        def findTarget(l, r, cmp = lambda a,b: a<b):
            while l < r:
                m = (l + r) // 2
                for i in (l,m,r):
                    if A[i] == -1: A[i] = mountain_arr.get(i)
                if target == A[m]: return m
                if cmp(target,A[m]): r = m - 1
                else: l = m + 1
            return l if target == A[l] else lenA
        targetI = peakI if target == A[peakI] else findTarget(0,peakI-1)
        if targetI == lenA:
            targetI = findTarget(peakI+1, lenA-1, cmp=lambda a,b: a>b)
        return targetI if targetI < lenA else -1
    def findInMountainArray(self, target: int, mountain_arr: 'MountainArray') -> int:
        def binary_search(left, right, cmp):            
            while left <= right:
                mid = (left + right) // 2                
                val = mountain_arr.get(mid)
                if val == target: return mid
                elif cmp(val, target): left = mid + 1                    
                else: right = mid - 1
            return -1
        left = 0
        right = lenn = mountain_arr.length() - 1
        while left < right: # find peak index
            mid = (left + right) // 2
            if mountain_arr.get(mid) < mountain_arr.get(mid + 1):
                left = mid + 1
            else: right = mid
        left_search = binary_search(0, left, operator.lt)
        return left_search if left_search != -1 else binary_search(left, lenn, operator.gt)
    # perfect peak greater max of all its left values, smaller than min of all its right values
    def perfectPeak(self, A):
        if len(A) < 3: return 0
        curMax = A[0]
        i = 1
        while i < len(A)-1:
            if A[i] > curMax:
                pivot = curMax = A[i] # A[i]: peak candidate, greater than curMax, hence all preceeding values
                j = i+1
                while j < len(A): # all values following curMax A[i]
                    if A[j] <= pivot:
                        i = j # continue outer loop at j, restart search for peak candidate
                        break
                    curMax = max(curMax, A[j]) # update curMax to be used for next peak candidate search, in case of break out of this loop
                    j += 1
                if j == len(A): return 1 # reach the end of A, found perfect peak i
            i += 1
        return 0
    # https://leetcode.com/problems/merge-intervals/submissions/
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort()
        output = [intervals[0]]
        for i in intervals[1:]:
            if i[0] <= output[-1][-1]:
                if i[-1] > output[-1][-1]:
                    output[-1][-1] = i[-1]
            else:
                output.append(i)
        return output
    # https://leetcode.com/problems/non-overlapping-intervals/submissions/
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        if len(intervals) < 2: return 0
        intervals.sort(key=lambda i: i[-1])
        cnt, cur = 0, intervals[0]
        for i in range(1,len(intervals)):
            if intervals[i][0] < cur[-1]:
                cnt +=1
            else:
                cur = intervals[i]
        return cnt
    # https://leetcode.com/discuss/interview-question/368573/google-onsite-restricted-knapsack-problem
    # greedy (2x lightest ONES or 1x lightest TWOS) is at worst 1 less than optimal. So we just need one swap at the end.
    def restricted_knapsack(self, capacity, weights, values):
        ones = sorted(((w, i) for i, w in enumerate(weights) if values[i] == 1), reverse=True) # Sort by weight
        twos = sorted(((w, i) for i, w in enumerate(weights) if values[i] == 2), reverse=True)
        indexes = []

        def take(a):
            nonlocal capacity
            w, i = a.pop()
            capacity -= w
            indexes.append(i)

        last_one = None  # While ONES.length >= 2 and TWOS.length >= 1: add to knapsack (ONES[0] + ONES[1]) or TWOS[0], whichever is lighter, 
        while len(ones) >= 2 and twos: # break if neither fits
            ones_weight = ones[-1][0] + ones[-2][0]
            twos_weight = twos[-1][0]
            if capacity < min(ones_weight, twos_weight):
                break
            elif twos_weight <= ones_weight:
                take(twos)
            else:
                take(ones)
                take(ones)
                if last_one is not None:
                    indexes.append(last_one)
                last_one = indexes.pop()
        for a in twos, ones: # Add lightest remaining TWOS that fit, then add lightest remaining ONES that fit
            while a and capacity >= a[-1][0]:
                take(a)

        if last_one is not None: # At this point, our solution is at worst one item away from optimal
            if twos and capacity + weights[last_one] - twos[-1][0] >= 0:
                take(twos)
            else:
                indexes.append(last_one)
        return indexes
    # https://levelup.gitconnected.com/understanding-dynamic-programming-in-theory-and-practice-7835610ca485
    @cache
    def computeMaxProfit(self, prices, size):
        if size <= 0: return 0
        maxProfit = 0
        for i in range(size):
            maxProf = prices[i] + self.computeMaxProfit(prices, size-i-1)
            maxProfit = max(maxProfit, maxProf)
        return maxProfit
    # https://www.youtube.com/watch?v=uQ_YsvOuXRY https://github.com/shreya367/InterviewBit/blob/master/Array/Repeat%20and%20missing%20number%20array
    def repeatMissingNumbers(self, A):
        aSum, a2Sum = sum(A), sum((n*n for n in A))
        nSum, n2Sum = sum(range(1,len(A)+1)), sum((n*n for n in range(1,len(A)+1)))
        x, y = aSum - nSum, a2Sum - n2Sum
        a = (x + y//x)//2
        return [a, a-x]
    # Pascal triangle https://github.com/shreya367/InterviewBit/blob/master/Array/Pascal%20triangle%20rows
    def pascalTriang(self, N):
        res = [[1]]
        for _ in range(N-1):
            _res = res[-1][:]
            _res.append(1)
            for i in range(1,len(_res)-1):
                _res[i] += res[-1][i-1]
            res.append(_res)
        return res    
    def pascalTriangRow(self, k):
        res = [1]
        for _ in range(k):
            _res = res[:]
            _res.append(1)
            for i in range(1,len(_res)-1):
                _res[i] += res[i-1]
            res = _res
        return res
    # https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree/
    def verticalTraversal(self, root: TreeNode) -> List[List[int]]: # Time O(NlogN), Space O(N)
        nodeMat = [[0,0,root.val]] # node matrix has root at x,y coordinates (0,0)
        curLevel = [[0,0,root.val,root]]; ans = []
        while curLevel:
            nextLevel = []
            for x,y,v,n in curLevel: 
                if n.left: heappush(nextLevel,[x-1,y+1,n.left.val,n.left])
                if n.right: heappush(nextLevel,[x+1,y+1,n.right.val,n.right])
            for t in nextLevel: heappush(nodeMat,t[:-1])
            curLevel = nextLevel
        g = nodeMat[0][0]; l = [] # vertical group
        while nodeMat:
            x,y,v = heappop(nodeMat)
            if x > g:
                g = x
                if l: ans.append(l)
                l = [v]
            else:
                l.append(v)
        ans.append(l)
        return ans
    # https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree/discuss/231256/python-queue-%2B-hash-map
    def verticalTraversal(self, root):
        g = defaultdict(list) # stage # to val map for all levels
        queue = [(root,0)] # stage # or vertical order
        while queue:
            new = []
            d = defaultdict(list)
            for node, s in queue:
                d[s].append(node.val) # stage # to val map at each level
                if node.left:  new += (node.left, s-1), 
                if node.right: new += (node.right,s+1),  
            for i in d: g[i].extend(sorted(d[i]))
            queue = new
        return [g[i] for i in sorted(g)]
    # https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree/discuss/231140/Solved-Add-clarification-for-the-output-order akhil_ak
    def verticalTraversal(self, root: TreeNode) -> List[List[int]]:     
        vertical_levels = defaultdict(list)
        def dfs(node,x=0,y=0):
            if node is None: return None
            dfs(node.left  ,x-1,y+1)
            dfs(node.right ,x+1,y+1)
            vertical_levels[x].append([y,node.val])
        dfs(root)
        return [  [node_value for y_value,node_value in sorted(values) ] for x_value,values  in sorted(vertical_levels.items())  ]
    # https://www.interviewbit.com/problems/vertical-order-traversal-of-binary-tree/
    # If 2 or more Tree Nodes shares the same vertical level then the one with earlier occurence in the pre-order traversal of tree comes first in the output.
    def verticalTraversalPreOrder(self, A: TreeNode) -> List[List[int]]: # Time O(NlogN), Space O(N)
        if not A: return []
        nodeMat = [[0,0,0,A.val]] # node matrix has root at x,y coordinates (0,0)
        curLevel = [[0,0,0,A.val,A]]; ans = []; preorder = 0
        while curLevel:
            nextLevel = []
            for x,y,o,v,n in curLevel:
                if n.left: preorder += 1;nextLevel.append([x-1,y+1,preorder,n.left.val,n.left])
                if n.right: preorder += 1;nextLevel.append([x+1,y+1,preorder,n.right.val,n.right])
            for t in nextLevel: heappush(nodeMat,t[:-1])
            curLevel = nextLevel
        g = nodeMat[0][0]; l = [] # vertical group
        while nodeMat:
            x,y,o,v = heappop(nodeMat)
            if x > g:
                g = x
                if l: ans.append(l)
                l = [v]
            else:
                l.append(v)
        ans.append(l)
        return ans

class Node(object):
    def __init__(self):
        self.outgoing_nodes = set()

class Solution1(object):
    def build_graph(self, edges):
        self.graph = { i: Node() for i in range(self.n) }
        for edge in edges:
            x, y = edge
            self.graph[x].outgoing_nodes.add(y)
            self.graph[y].outgoing_nodes.add(x)
    
    def bfs_max_dist_node(self, root):
        visited = set()
        dist = { root: 0 }
        queue = [root]
        max_dist = 0
        chosen = root
        while queue:
            now = queue.pop(0)
            visited.add(now)
            for node in self.graph[now].outgoing_nodes:
                if node in visited:
                    continue
                dist[node] = dist[now] + 1
                if dist[node] > max_dist:
                    max_dist = dist[node]
                    chosen = node
                queue.append(node)
        return (max_dist, chosen)
            
    def find_endpoint1(self):
        max_dist, chosen = self.bfs_max_dist_node(0)
        return chosen
    
    def find_path(self, src, target):
        path = [src]
        data = { 'success': False, 'final_path': None }
        
        def dfs(now, prev):
            if now == target:
                data['success'] = True
                data['final_path'] = path[:]
            if data['success']:
                return
            for node in self.graph[now].outgoing_nodes:
                if node == prev:
                    continue
                path.append(node)
                dfs(node, now)
                path.pop()
        
        dfs(src, -1)
        return data['final_path']
    
    def get_mid(self, path):
        l = len(path)
        print(path, l)
        if l % 2 == 0:
            return [path[l/2-1], path[l/2]]
        else:
            return [path[l/2]]
    
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        if n == 0: return []
        self.n = n
        self.build_graph(edges) # Build the graph
        ed1 = self.find_endpoint1() # BFS and find the endpoint #1 of longest path
        _, ed2 = self.bfs_max_dist_node(ed1) # calc height, root: #1, btw find ed2
        path = self.find_path(ed1, ed2) # DFS (optionally with simulated stack)  path between #1 and #2
        midpoints = self.get_mid(path) # find middle point(s)
        return midpoints
class Solution0:
    def pathSum1(self, root: TreeNode, sum: int) -> int:
        def findPaths(root: TreeNode) -> List[List[int]]:
            return [[10,-3,11],[10,5,2,1],[10,5,3,-2],[10,5,3,3]]
        res = 0
        
        for path in findPaths(root):
          num = NumSubArrSumX(path, sum)
          res += num
        return res
    def pathSum(self, root: TreeNode, target: int) -> int:
        self.numOfPaths = 0
        if root:
            self.dfs(root, target)
        return self.numOfPaths
    
    # define: traverse through the tree, at each treenode, call another DFS to test if a path sum include the answer
    def dfs(self, node, target):
        # dfs break down 
        self.test(node, target) # you can move the line to any order, here is pre-order
        for n in (node.left,node.right):
            if n:
                self.dfs(n, target)
        
    # define: for a given node, DFS to find any path that sum == target, if find self.numOfPaths += 1
    def test(self, node, target):
        if node.val == target:
            self.numOfPaths += 1
            
        # test break down
        for n in (node.left,node.right):
            if n:
                self.test(n, target-node.val)

# These are the tests we use to determine if the solution is correct.
# You can add your own at the bottom, but they are otherwise not editable!

def printInteger(n):
  print('[', n, ']', sep='', end='')

test_case_number = 1

def check(expected, output):
  global test_case_number
  result = False
  if expected == output:
    result = True
  rightTick = '\u2713'
  wrongTick = '\u2717'
  if result:
    print(rightTick, 'Test #', test_case_number, sep='')
  else:
    print(wrongTick, 'Test #', test_case_number, ': Expected ', sep='', end='')
    printInteger(expected)
    print(' Your output: ', end='')
    printInteger(output)
    print()
  test_case_number += 1

if __name__ == "__main__":
    sol = Solution()
#   assert sol.partition("aab") == [["a","a","b"],["aa","b"]]
#   assert sol.convert2Palin("pwxu") == 0
#   assert sol.convert2Palin("iph") == 0
#   print(sol.restricted_knapsack(7, [2, 2, 5, 6], [1, 1, 2, 1]))
#   @timeit
#   def tw():
#     sol.computeMaxProfit((1,5,8,9,10,17,17,20,24,30), 10)
#   print(tw())
#   print(sol.nextPermutation([1,2,3]))
#   arr = MountainArray([1,2,3,4,5,3,1])
#   print(sol.findInMountainArray(3, arr))
    print(sol.pascalTriangRow(3))
#   print(sol.firstMissingPositive([1,2,0]))
#   print(sol.firstMissingPositive([3,4,-1,1]))
#   print(sol.findDuplicate([1,3,4,2,2]))
#   expected=[1,2,3,6,9,8,7,4,5]
#   print(expected)
#   print(sol.spiralOrder([[1,2,3],[4,5,6],[7,8,9]]))
#   for r in sol.generateMatrix(3): print(r)
#   for r in sol.generateMatrix(4): print(r)
#   print(sol.repeatMissingNumbers([3, 1, 2, 5, 3]))
    print(sol.waveArray([3,1,7,2,8]))
    assert sol.rotate([1,2,3,4,5,6,7], 3) == [5,6,7,1,2,3,4]
#   print(sol.findDuplicate([1,3,4,2,2]))
#   print(sol.perfectPeak([1,3,2]))
    print(sol.reverse(1534236469))
    A = [1, 3, -1]
    assert sol.maxAbsDiff(A) == sol.maxArr(A)
    tdata = [69953237, 59183787, 16962742, 53647827, 80157178, 51106992, 58228227, 45131842, 70499719, 70765861, 43961028, 6698667, 99911553, 79107222, 67571988, 39721137, 78088316, 3759045, 19395856, 29387266, 68084358, 62564561, 24736359, 13212412, 66665326, 38724565, 61088241, 21263259, 89291805, 88650356, 58518225, 86449553, 78979492, 39596282, 43927666, 35451400, 80068197, 23391371, 25433080, 5888423, 67042527, 15586432, 57608751, 75903078, 95593533, 15702947, 39691466, 92690796, 18015358, 95172428, 72245309, 15424690, 41199673, 71322081, 27606512, 2347516, 1354382, 9924819, 63458285, 13170098, 40075662, 31237137, 45236128, 74375452, 92722404, 80087546, 23399482, 86945189, 3780890, 1963037, 76980637, 41676736, 74194802, 64788125, 88954508, 95737994, 21365859, 71092491, 67365387, 62345424, 77276892, 53193048, 30131824, 5365626, 66817225, 64511810, 46917019, 80497257, 20853093, 26175229, 85887940, 85764880, 78262084, 609284, 92269014, 46385693, 53718740, 17486900, 98427277, 92911988, 32225164, 72512163, 88678886, 65347756, 40460802, 33132933, 88603373, 26890724, 87077147, 99305881, 55925130, 83289365, 54166373, 50920143, 4427534, 29179799, 91572049, 95103705, 56304651, 69828935, 76914922, 33694020, 15575017, 77664401, 91393916, 96189668, 82107391, 95777779, 22244308, 65701218, 23227429, 98614556, 43407558, 67137144, 34515594, 89248417, 47520685, 14084016, 91725069, 94236666, 61860638, 82315091, 88113674, 31949150, 11718471, 51617813, 41754631, 15588021, 8130184, 52515921, 20663946, 70850137, 85578292, 93271926, 46273611, 27972346, 16865457, 94763722, 28780820, 52198047, 39535546, 9854737, 56888868, 82035778, 88667377, 71915993, 23061619, 71237088, 38215964, 99455111, 84338139, 9438659, 87387886, 35325804, 36964271, 40598041, 83828315, 30761279, 26893177, 19907874, 70129736, 62700567, 91806797, 11958671, 98578052, 8205009, 47197783, 53336153, 36819554, 47426225, 60695466, 55323353, 23435249, 16782401, 37928333, 19599390, 39797644, 9436396, 69658044, 45212110, 48265238, 14183162, 74865579, 58521415, 21894773, 65729368, 91458346, 39977875, 42236097, 12766362, 92049518, 13196912, 18064381, 89025324, 18154460, 77179251, 84814940, 3021813, 25547069, 65821055, 64653709, 94102131, 68518939, 46556175, 82058200, 80932197, 46454512, 62876026, 59675187, 4157064, 56677982, 76761579, 58446935, 51180436, 16416645, 65330136, 39435329, 81388109, 15522197, 78011106, 67617066, 34445170, 75555584, 87940550, 29128931, 61824105, 63418827, 21298249, 19352955, 91556471, 92675461, 888811, 61724633, 69153235, 83033789, 60689257, 54479401, 27546578, 66335535, 38378169, 6852142, 76001740, 64337900, 84778273, 7063943, 20469454, 8897942, 81095011, 89277538, 89473685, 13442135, 54332314, 99428305, 49227963, 1550084, 58626108, 3149845, 9592847, 63553909, 64442175, 52746464, 71731610, 68109474, 12014994, 28405025, 342244, 77415207, 6554496, 29199336, 24816541, 86021818, 93254912, 70091579, 24906579, 50431415, 92218181, 96921595, 70001355, 69278435, 48183386, 4909222, 9784895, 59239264, 82952057, 54807579, 85365017, 10747717, 36328910, 99503949, 25462804, 1948980, 54007615, 93059997, 96143890, 87526013, 36668467, 6585021, 59026699, 42437483, 49977217, 10735602, 29201394, 54519214, 94548380, 46479340, 50083557, 9899537, 43429844, 62002345, 57109791, 15539611, 37100828, 96624737, 58882555, 18287280, 73908899, 58094409, 31628808, 7279505, 87539497, 35968234, 73051936, 9182618, 6399914, 85644731, 13422910, 6027207, 40194013, 49063637, 47973273, 98583990, 4494264, 54221620, 99760871, 52094275, 44298611, 98771831, 39244836, 20234483, 63104483, 17667669, 77220658, 97536077, 67203516, 9567573, 96789571, 6954478, 9245869, 82742448, 72371341, 57459624, 25406080, 52099423, 49750024, 77497879, 16327199, 39571526, 14570597, 76246637, 93751329, 86339423, 97188050, 48457113, 52455837, 77244782, 81308324, 74575702, 31144906, 30685803, 21269958, 31958563, 20917120, 52658082, 33778983, 85532378, 6177062, 35138461, 27377273, 68310047, 27164914, 24236293, 90140894, 26903650, 31650250, 45975378, 10537609, 11918603, 92943222, 77479401, 42865540, 50885917, 57892395, 55161914, 82804155, 16502312, 25439804, 2105003, 52617639, 69471748, 48335908, 17352118, 36510341, 55222894, 6319593, 79662620, 37402118, 68949184, 48515926, 12224685, 8095432, 36227103, 37292769, 79979610, 78482598, 23728343, 26166144, 41593189, 44829841, 44241275, 43522523, 63777807, 3827324, 2683674, 3824814, 35307572, 1837804, 12651831, 24382914, 44014481, 16990793, 41894672, 33681016]
    tdata = [46158044, 9306314, 51157916, 93803496, 20512678, 55668109, 488932, 24018019, 91386538, 68676911, 92581441, 66802896, 10401330, 57053542, 42836847, 24523157, 50084224, 16223673, 18392448, 61771874, 75040277, 30393366, 1248593, 71015899, 20545868, 75781058, 2819173, 37183571, 94307760, 88949450, 9352766, 26990547, 4035684, 57106547, 62393125, 74101466, 87693129, 84620455, 98589753, 8374427, 59030017, 69501866, 47507712, 84139250, 97401195, 32307123, 41600232, 52669409, 61249959, 88263327, 3194185, 10842291, 37741683, 14638221, 61808847, 86673222, 12380549, 39609235, 98726824, 81436765, 48701855, 42166094, 88595721, 11566537, 63715832, 21604701, 83321269, 34496410, 48653819, 77422556, 51748960, 83040347, 12893783, 57429375, 13500426, 49447417, 50826659, 22709813, 33096541, 55283208, 31924546, 54079534, 38900717, 94495657, 6472104, 47947703, 50659890, 33719501, 57117161, 20478224, 77975153, 52822862, 13155282, 6481416, 67356400, 36491447, 4084060, 5884644, 91621319, 43488994, 71554661, 41611278, 28547265, 26692589, 82826028, 72214268, 98604736, 60193708, 95417547, 73177938, 50713342, 6283439, 79043764, 52027740, 17648022, 33730552, 42851318, 13232185, 95479426, 70580777, 24710823, 48306195, 31248704, 24224431, 99173104, 31216940, 66551773, 94516629, 67345352, 62715266, 8776225, 18603704, 7611906]
    print(sol.maximumGap(tdata))
    print(sol.maximumGap([3, 5, 4, 2]))
#   print(sol.nextPermutation([100,99,98,97,96,95,94,93,92,91,90,89,88,87,86,85,84,83,82,81,80,79,78,77,76,75,74,73,72,71,70,69,68,67,66,65,64,63,62,61,60,59,58,57,56,55,54,53,52,51,50,49,48,47,46,45,44,43,42,41,40,39,38,37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]))
#   print(sol.nextPermutation([3,2,1]))
#   print(sol.getPermutation(4,13))
#   res = sol.findPerm("IIIDDDDDDDDIDDDIDDIIDDDDDDIIIIDIIIDDDIDIIIDDDIDDDDDDIIIDDDIIDDIIDIDIIIDIDIDIIIDDIIIIIDIIIIIDDIDDIDDDDIDIIDDIDIIDDIIDDIDDIDDDIIIIDIDDIDDDIIDDDDDIIDDDDDDDIIIIIDDIDIDDDIIDDIDIDIIDDDDIIIDDIDDIIIIDDDDIIDIDDDDDDDIIIDIDDDIDIDIDIIIDIDDIDDDIIIDDDIDDDDIDIIIDIIIIDIDDIIDDIIDIIDIDDDIIDDDDDIIDIIDDDIIDDDIDDDDDIDIDDDIIIDDDDIDIIIDDDIIIDIDDIIIIIDIDIIDIDIDDIIDDDIIIIDIIDDDDDDIDIIIIIDIDIIIIIDDDIIDIDDIIIDIIDIDDIIIIDIDDIIIDDDIDDIIIDIDIIDIDDDIDDIDDDIIDIIIIIDDDDDIIIDIIIIDDIDIDIDIDDDIIDDIDIDDDDDDDIIDIIIDIDDIDIIIDDDDDIDIIDDDIIIDIIIIDIDDDIDDIIDIDIDIIDDIIIDIDIDDIDIDDDDIIIDIIDIIDIIIDDIDIIDDIIIIDIIIIDIIDIIIDDIIDIIIIDDIDIDDIDDDIDDIIDIIDIIIDIDIIDIIIDIDDDIDDIIDDDDIDDIIDIDDIIDIDIDIDDDDDIIIIDDDIIDDDDIIDDDDDDIIDDIIIIDDIIDIDIDDIDDDIDIIIDDDIDDDIIIDIDIIDIIIIDIDIDIDIIDIIID",743)
#   expected = "1 2 3 743 742 741 740 739 738 737 736 4 735 734 733 5 732 731 6 7 730 729 728 727 726 725 8 9 10 11 724 12 13 14 723 722 721 15 720 16 17 18 719 718 717 19 716 715 714 713 712 711 20 21 22 710 709 708 23 24 707 706 25 26 705 27 704 28 29 30 703 31 702 32 701 33 34 35 700 699 36 37 38 39 40 698 41 42 43 44 45 697 696 46 695 694 47 693 692 691 690 48 689 49 50 688 687 51 686 52 53 685 684 54 55 683 682 56 681 680 57 679 678 677 58 59 60 61 676 62 675 674 63 673 672 671 64 65 670 669 668 667 666 66 67 665 664 663 662 661 660 659 68 69 70 71 72 658 657 73 656 74 655 654 653 75 76 652 651 77 650 78 649 79 80 648 647 646 645 81 82 83 644 643 84 642 641 85 86 87 88 640 639 638 637 89 90 636 91 635 634 633 632 631 630 629 92 93 94 628 95 627 626 625 96 624 97 623 98 622 99 100 101 621 102 620 619 103 618 617 616 104 105 106 615 614 613 107 612 611 610 609 108 608 109 110 111 607 112 113 114 115 606 116 605 604 117 118 603 602 119 120 601 121 122 600 123 599 598 597 124 125 596 595 594 593 592 126 127 591 128 129 590 589 588 130 131 587 586 585 132 584 583 582 581 580 133 579 134 578 577 576 135 136 137 575 574 573 572 138 571 139 140 141 570 569 568 142 143 144 567 145 566 565 146 147 148 149 150 564 151 563 152 153 562 154 561 155 560 559 156 157 558 557 556 158 159 160 161 555 162 163 554 553 552 551 550 549 164 548 165 166 167 168 169 547 170 546 171 172 173 174 175 545 544 543 176 177 542 178 541 540 179 180 181 539 182 183 538 184 537 536 185 186 187 188 535 189 534 533 190 191 192 532 531 530 193 529 528 194 195 196 527 197 526 198 199 525 200 524 523 522 201 521 520 202 519 518 517 203 204 516 205 206 207 208 209 515 514 513 512 511 210 211 212 510 213 214 215 216 509 508 217 507 218 506 219 505 220 504 503 502 221 222 501 500 223 499 224 498 497 496 495 494 493 492 225 226 491 227 228 229 490 230 489 488 231 487 232 233 234 486 485 484 483 482 235 481 236 237 480 479 478 238 239 240 477 241 242 243 244 476 245 475 474 473 246 472 471 247 248 470 249 469 250 468 251 252 467 466 253 254 255 465 256 464 257 463 462 258 461 259 460 459 458 457 260 261 262 456 263 264 455 265 266 454 267 268 269 453 452 270 451 271 272 450 449 273 274 275 276 448 277 278 279 280 447 281 282 446 283 284 285 445 444 286 287 443 288 289 290 291 442 441 292 440 293 439 438 294 437 436 435 295 434 433 296 297 432 298 299 431 300 301 302 430 303 429 304 305 428 306 307 308 427 309 426 425 424 310 423 422 311 312 421 420 419 418 313 417 416 314 315 415 316 414 413 317 318 412 319 411 320 410 321 409 408 407 406 405 322 323 324 325 404 403 402 326 327 401 400 399 398 328 329 397 396 395 394 393 392 330 331 391 390 332 333 334 335 389 388 336 337 387 338 386 339 385 384 340 383 382 381 341 380 342 343 344 379 378 377 345 376 375 374 346 347 348 373 349 372 350 351 371 352 353 354 355 370 356 369 357 368 358 367 359 360 366 361 362 363 365 364"
#   print(res == list(map(int,expected.split())), res)
#   print(sol.cherryPickup([[1,1,1,0,0],[0,0,1,0,1],[1,0,1,0,0],[0,0,1,0,0],[0,0,1,1,1]]))
    assert sol.minPathSum([[1,3,1],[1,5,1],[4,2,1]]) == 7
    assert sol.minPathSum([[1,2,3],[4,5,6]]) == 12
    assert sol.minPathSum([[1,3,2],[4,3,1],[5,6,1]]) == 8
    expected = [[[1,1,6],[1,2,5],[1,7],[2,6]],[[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 2], [1, 1, 1, 1, 1, 1, 3], [1, 1, 1, 1, 1, 2, 2], [1, 1, 1, 1, 2, 3], [1, 1, 1, 1, 5], [1, 1, 1, 2, 2, 2], [1, 1, 1, 3, 3], [1, 1, 1, 6], [1, 1, 2, 2, 3], [1, 1, 2, 5], [1, 1, 7], [1, 2, 2, 2, 2], [1, 2, 3, 3], [1, 2, 6], [1, 3, 5], [2, 2, 2, 3], [2, 2, 5], [2, 7], [3, 3, 3], [3, 6]],
[[1, 1, 1, 1], [1, 1, 2], [2, 2]], [[1, 1]], [[1]], [], [[2, 2, 2, 2], [2, 3, 3], [3, 5]], [[2, 2, 3], [7]]]
    tdata = [([10,1,2,7,6,5],8),([2,7,6,3,5,1],9),([1,2],4),([1],2),([1],1),([2],1),([2,3,5],8),([2,3,6,7],7)]
    tdata = [800,[[695,229],[199,149],[443,397],[258,247],[781,667],[350,160],[678,629],[467,166],[500,450],[477,107],[483,151],[792,785],[752,368],[659,623],[316,224],[487,268],[743,206],[552,211],[314,20],[720,196],[421,103],[493,288],[762,24],[528,318],[472,32],[684,502],[641,354],[586,480],[629,54],[611,412],[719,680],[733,42],[549,519],[697,316],[791,634],[546,70],[676,587],[460,58],[605,530],[617,579],[484,89],[571,482],[767,200],[555,547],[771,695],[624,542],[708,551],[432,266],[656,468],[724,317],[423,248],[621,593],[781,399],[535,528],[578,12],[770,549],[576,295],[318,247],[400,372],[465,363],[786,482],[441,398],[577,411],[524,30],[741,540],[459,59],[758,96],[550,89],[402,295],[476,336],[645,346],[750,116],[551,207],[343,226],[568,498],[530,228],[525,84],[507,128],[526,210],[535,381],[635,330],[654,535],[710,275],[397,213],[412,44],[131,70],[508,49],[679,223],[519,11],[626,286],[242,160],[778,199],[606,281],[226,16],[340,46],[578,127],[212,208],[674,343],[778,108],[749,451],[735,105],[544,131],[600,229],[691,314],[608,74],[613,491],[754,500],[722,449],[486,11],[786,70],[212,23],[717,11],[692,410],[503,157],[783,177],[220,215],[419,363],[182,17],[321,54],[711,78],[312,106],[560,101],[501,178],[583,403],[577,9],[595,227],[601,386],[792,619],[550,167],[589,431],[793,243],[395,76],[197,3],[357,6],[763,7],[599,48],[178,92],[325,307],[620,10],[334,117],[556,296],[454,394],[485,236],[140,80],[404,301],[651,58],[504,455],[101,93],[712,42],[559,421],[594,230],[505,98],[719,654],[672,283],[109,73],[556,183],[617,94],[133,100],[771,515],[613,587],[285,50],[579,432],[282,244],[669,527],[783,494],[628,560],[716,661],[177,127],[430,166],[383,159],[746,19],[653,284],[495,243],[376,57],[560,143],[679,198],[751,355],[339,157],[409,140],[729,389],[518,315],[623,352],[651,133],[761,269],[442,44],[379,245],[313,180],[773,583],[291,221],[271,54],[799,44],[200,102],[568,67],[695,167],[327,36],[431,73],[782,167],[611,129],[630,122],[563,497],[697,93],[596,436],[611,131],[627,256],[658,559],[591,419],[193,156],[302,52],[409,33],[405,249],[384,151],[214,142],[558,164],[565,557],[492,445],[681,271],[797,396],[251,195],[784,266],[607,179],[671,30],[752,179],[787,390],[749,532],[618,220],[659,298],[567,134],[229,208],[298,147],[787,459],[572,359],[794,351],[53,14],[646,422],[234,66],[274,255],[744,626],[730,462],[498,428],[573,288],[688,355],[603,25],[191,16],[793,544],[750,682],[415,156],[460,209],[749,85],[269,186],[441,338],[319,278],[505,18],[672,260],[420,233],[493,134],[493,19],[308,302],[582,282],[755,60],[641,626],[669,69],[772,29],[132,111],[666,120],[605,58],[534,252],[636,491],[777,3],[602,368],[533,287],[401,147],[782,669],[517,161],[686,49],[789,639],[776,379],[376,65],[696,545],[423,81],[448,336],[631,605],[501,387],[413,94],[777,563],[661,332],[756,359],[646,36],[650,283],[656,347],[522,7],[383,382],[438,102],[762,305],[650,15],[249,180],[784,467],[763,122],[163,115],[775,734],[166,132],[634,2],[668,584],[767,274],[595,552],[11,7],[693,407],[789,751],[613,556],[715,402],[751,516],[646,199],[625,52],[572,106],[724,332],[617,409],[573,526],[760,18],[382,202],[207,139],[416,392],[672,358],[233,212],[668,22],[765,452],[294,76],[259,47],[593,271],[510,450],[592,132],[770,558],[296,43],[419,86],[752,347],[615,605],[635,554],[794,635],[613,316],[563,61],[770,715],[771,251],[646,582],[423,79],[576,249],[604,97],[767,348],[736,239],[775,56],[619,601],[790,546],[531,384],[507,84],[564,337],[432,310],[600,543],[747,341],[556,392],[661,113],[449,282],[575,288],[637,7],[635,325],[735,574],[574,387],[705,603],[704,15],[684,588],[495,132],[718,223],[517,206],[272,34],[677,416],[788,167],[649,525],[619,427],[541,277],[489,405],[608,259],[603,264],[435,317],[623,26],[544,511],[72,69],[623,17],[600,544],[551,367],[404,52],[324,272],[706,205],[778,446],[341,155],[581,173],[666,192],[588,529],[554,506],[250,39],[772,116],[569,77],[526,132],[563,221],[655,597],[649,224],[57,4],[679,199],[265,157],[380,335],[558,35],[726,388],[763,567],[437,426],[643,103],[773,181],[726,68],[164,50],[717,427],[681,618],[477,172],[697,423],[525,383],[794,132],[149,70],[704,414],[581,139],[678,204],[107,46],[352,68],[645,178],[758,156],[627,365],[331,144],[547,340],[788,36],[633,259],[588,66],[321,102],[528,322],[212,36],[288,179],[434,189],[749,490],[753,508],[784,341],[550,159],[741,206],[758,688],[766,758],[586,70],[657,654],[701,104],[548,184],[613,162],[620,320],[506,430],[517,65],[571,291],[771,517],[796,756],[735,459],[625,367],[759,345],[582,468],[469,73],[790,352],[493,284],[664,567],[342,207],[669,108],[611,182],[764,485],[214,102],[544,202],[713,447],[793,378],[147,129],[407,198],[608,271],[695,667],[680,277],[222,163],[744,527],[280,116],[430,367],[281,228],[688,488],[733,92],[529,190],[750,718],[793,99],[626,169],[486,329],[620,0],[782,460],[329,16],[753,142],[338,172],[518,361],[688,168],[497,490],[484,365],[365,325],[107,98],[622,407],[527,277],[659,74],[552,538],[493,469],[638,147],[304,3],[573,201],[411,169],[719,309],[287,160],[742,175],[573,299],[562,473],[705,328],[261,98],[580,203],[740,26],[418,296],[764,170],[656,89],[724,536],[730,91],[796,290],[735,270],[512,20],[402,246],[46,30],[426,290],[296,57],[725,222],[324,317],[547,0],[661,136],[636,271],[261,56],[750,668],[647,402],[773,390],[677,62],[249,53],[574,4],[393,304],[701,44],[109,66],[275,109],[679,509],[725,21],[409,311],[368,156],[605,514],[538,42],[690,602],[411,343],[424,240],[78,40],[750,273],[367,230],[167,58],[738,200],[634,341],[409,170],[644,373],[741,296],[702,342],[746,233],[411,67],[526,436],[796,438],[647,312],[717,347],[548,54],[725,50],[549,92],[610,294],[668,350],[578,445],[446,93],[727,246],[526,355],[344,246],[145,54],[355,256],[751,46],[454,271],[587,49],[728,79],[627,49],[522,260],[270,250],[491,113],[337,258],[609,470],[387,147],[656,237],[366,357],[160,98],[761,692],[753,627],[718,5],[335,22],[640,78],[687,317],[315,295],[471,93],[481,147],[724,580],[687,177],[409,41],[355,276],[393,366],[770,85],[697,358],[187,115],[671,318],[716,530],[767,140],[566,543],[318,238],[341,336],[648,204],[496,202],[505,191],[360,32],[408,138],[537,489],[668,102],[400,9],[472,3],[727,469],[713,5],[530,292],[465,381],[551,262],[514,227],[394,315],[551,121],[655,402],[755,83],[153,144],[303,60],[766,578],[668,527],[796,391],[692,571],[617,616],[229,154],[798,690],[706,504],[610,569],[655,624],[408,108],[569,463],[461,151],[507,13],[781,314],[780,469],[506,171],[552,312],[189,164],[336,171],[571,432],[688,224],[160,119],[470,311],[663,114],[665,420],[556,492],[709,358],[202,99],[170,149],[340,154],[666,385],[617,383],[502,132],[220,42],[778,393],[444,68],[526,357],[217,7],[597,76],[586,406],[481,44],[486,240],[513,217],[790,447],[275,245],[396,1],[369,224],[485,159],[680,151],[387,312],[721,70],[733,25],[457,216],[798,297],[329,169],[766,212],[286,160],[703,164],[765,77],[620,142],[510,35],[475,400],[784,8],[768,189],[668,328],[697,2],[389,169],[550,223],[514,268],[579,285],[419,53],[318,96],[335,117],[729,27],[694,281],[349,137],[545,221],[679,100],[382,116],[707,140],[62,48],[664,312],[499,369],[547,350],[509,279],[778,76],[186,17],[741,683],[635,531],[441,391],[493,385],[354,218],[304,128],[651,271],[693,360],[613,112],[798,393],[743,190],[115,62],[725,592],[525,233],[621,517],[327,70],[501,358],[504,346],[787,321],[94,74],[729,339],[50,13],[603,265],[163,29],[781,373],[586,459],[797,741],[624,364],[411,277],[360,161],[690,686],[746,639],[553,325],[631,328],[388,330],[619,210],[573,43],[559,100],[210,152],[378,5],[776,447],[615,181],[365,299],[708,310],[718,690],[268,225],[639,90],[318,5],[196,89],[361,184],[762,690],[772,465],[729,721],[541,331],[567,350],[269,58],[656,78],[579,163],[711,223],[282,268],[760,533],[404,280],[473,384],[94,48],[340,12],[727,364],[264,221],[591,487],[514,466],[305,168],[372,248],[639,499],[560,435],[541,142],[462,83],[594,353],[618,485],[95,33],[602,595],[605,289],[715,207],[448,293],[752,170],[641,203],[532,198],[608,13],[707,114],[744,211],[110,3],[298,228],[622,496],[286,26],[683,178],[706,192],[751,358],[486,461],[561,251],[466,193],[342,62],[221,37],[731,325],[205,132],[518,173],[502,261],[640,49],[541,522],[747,110],[756,591],[124,76],[639,603],[765,482],[388,5],[34,12],[514,344],[495,254],[770,751],[730,597],[708,105],[683,586],[528,288],[386,225],[287,26],[649,262],[753,670],[789,85],[632,439],[570,176],[672,652],[445,399],[400,226],[655,522],[469,249],[557,500],[275,6],[397,296],[725,43],[605,533],[425,220],[637,118],[628,215],[654,431],[697,421],[512,121],[237,36],[151,85],[574,217],[320,233],[492,272],[552,220],[739,81],[712,219],[612,590],[410,66],[548,40],[320,211],[381,95],[633,482],[742,535],[704,131],[682,435],[508,48],[435,337],[534,96],[663,653],[283,205],[715,74],[484,376],[585,366],[635,479],[753,719],[793,548],[396,171],[156,112],[575,380],[717,464],[612,576],[569,319],[736,259],[406,227],[711,709],[793,132],[528,295],[592,48],[731,217],[408,299],[373,137],[786,327],[791,166],[712,285],[772,603],[723,338],[531,121],[572,548],[786,167],[670,401],[724,440],[280,229],[497,453],[265,70],[733,144],[689,434],[504,384],[93,64],[563,397],[550,106],[224,198],[372,177],[249,31],[667,372],[263,78],[783,446],[791,59],[438,64],[630,270],[216,160],[704,261],[674,506],[704,23],[378,4],[784,437],[196,118],[681,314],[698,663],[397,274],[499,440],[737,265],[697,625],[139,84],[440,231],[453,150],[266,55],[377,11],[728,60],[431,202],[268,47],[763,123],[347,339],[470,117],[466,298],[344,142],[584,55],[417,175],[439,392],[548,55],[714,701],[643,71],[357,69],[649,459],[789,541],[626,5],[752,619],[711,267],[639,12],[750,364],[620,249],[769,721],[636,97],[233,15],[171,72],[488,421],[251,139],[750,98],[199,64],[768,344],[759,537],[435,154],[425,185],[336,221],[418,395],[390,136],[618,603]]]
    tdata = [5000,[[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11],[11,12],[12,13],[13,14],[14,15],[15,16],[16,17],[17,18],[18,19],[19,20],[20,21],[21,22],[22,23],[23,24],[24,25],[25,26],[26,27],[27,28],[28,29],[29,30],[30,31],[31,32],[32,33],[33,34],[34,35],[35,36],[36,37],[37,38],[38,39],[39,40],[40,41],[41,42],[42,43],[43,44],[44,45],[45,46],[46,47],[47,48],[48,49],[49,50],[50,51],[51,52],[52,53],[53,54],[54,55],[55,56],[56,57],[57,58],[58,59],[59,60],[60,61],[61,62],[62,63],[63,64],[64,65],[65,66],[66,67],[67,68],[68,69],[69,70],[70,71],[71,72],[72,73],[73,74],[74,75],[75,76],[76,77],[77,78],[78,79],[79,80],[80,81],[81,82],[82,83],[83,84],[84,85],[85,86],[86,87],[87,88],[88,89],[89,90],[90,91],[91,92],[92,93],[93,94],[94,95],[95,96],[96,97],[97,98],[98,99],[99,100],[100,101],[101,102],[102,103],[103,104],[104,105],[105,106],[106,107],[107,108],[108,109],[109,110],[110,111],[111,112],[112,113],[113,114],[114,115],[115,116],[116,117],[117,118],[118,119],[119,120],[120,121],[121,122],[122,123],[123,124],[124,125],[125,126],[126,127],[127,128],[128,129],[129,130],[130,131],[131,132],[132,133],[133,134],[134,135],[135,136],[136,137],[137,138],[138,139],[139,140],[140,141],[141,142],[142,143],[143,144],[144,145],[145,146],[146,147],[147,148],[148,149],[149,150],[150,151],[151,152],[152,153],[153,154],[154,155],[155,156],[156,157],[157,158],[158,159],[159,160],[160,161],[161,162],[162,163],[163,164],[164,165],[165,166],[166,167],[167,168],[168,169],[169,170],[170,171],[171,172],[172,173],[173,174],[174,175],[175,176],[176,177],[177,178],[178,179],[179,180],[180,181],[181,182],[182,183],[183,184],[184,185],[185,186],[186,187],[187,188],[188,189],[189,190],[190,191],[191,192],[192,193],[193,194],[194,195],[195,196],[196,197],[197,198],[198,199],[199,200],[200,201],[201,202],[202,203],[203,204],[204,205],[205,206],[206,207],[207,208],[208,209],[209,210],[210,211],[211,212],[212,213],[213,214],[214,215],[215,216],[216,217],[217,218],[218,219],[219,220],[220,221],[221,222],[222,223],[223,224],[224,225],[225,226],[226,227],[227,228],[228,229],[229,230],[230,231],[231,232],[232,233],[233,234],[234,235],[235,236],[236,237],[237,238],[238,239],[239,240],[240,241],[241,242],[242,243],[243,244],[244,245],[245,246],[246,247],[247,248],[248,249],[249,250],[250,251],[251,252],[252,253],[253,254],[254,255],[255,256],[256,257],[257,258],[258,259],[259,260],[260,261],[261,262],[262,263],[263,264],[264,265],[265,266],[266,267],[267,268],[268,269],[269,270],[270,271],[271,272],[272,273],[273,274],[274,275],[275,276],[276,277],[277,278],[278,279],[279,280],[280,281],[281,282],[282,283],[283,284],[284,285],[285,286],[286,287],[287,288],[288,289],[289,290],[290,291],[291,292],[292,293],[293,294],[294,295],[295,296],[296,297],[297,298],[298,299],[299,300],[300,301],[301,302],[302,303],[303,304],[304,305],[305,306],[306,307],[307,308],[308,309],[309,310],[310,311],[311,312],[312,313],[313,314],[314,315],[315,316],[316,317],[317,318],[318,319],[319,320],[320,321],[321,322],[322,323],[323,324],[324,325],[325,326],[326,327],[327,328],[328,329],[329,330],[330,331],[331,332],[332,333],[333,334],[334,335],[335,336],[336,337],[337,338],[338,339],[339,340],[340,341],[341,342],[342,343],[343,344],[344,345],[345,346],[346,347],[347,348],[348,349],[349,350],[350,351],[351,352],[352,353],[353,354],[354,355],[355,356],[356,357],[357,358],[358,359],[359,360],[360,361],[361,362],[362,363],[363,364],[364,365],[365,366],[366,367],[367,368],[368,369],[369,370],[370,371],[371,372],[372,373],[373,374],[374,375],[375,376],[376,377],[377,378],[378,379],[379,380],[380,381],[381,382],[382,383],[383,384],[384,385],[385,386],[386,387],[387,388],[388,389],[389,390],[390,391],[391,392],[392,393],[393,394],[394,395],[395,396],[396,397],[397,398],[398,399],[399,400],[400,401],[401,402],[402,403],[403,404],[404,405],[405,406],[406,407],[407,408],[408,409],[409,410],[410,411],[411,412],[412,413],[413,414],[414,415],[415,416],[416,417],[417,418],[418,419],[419,420],[420,421],[421,422],[422,423],[423,424],[424,425],[425,426],[426,427],[427,428],[428,429],[429,430],[430,431],[431,432],[432,433],[433,434],[434,435],[435,436],[436,437],[437,438],[438,439],[439,440],[440,441],[441,442],[442,443],[443,444],[444,445],[445,446],[446,447],[447,448],[448,449],[449,450],[450,451],[451,452],[452,453],[453,454],[454,455],[455,456],[456,457],[457,458],[458,459],[459,460],[460,461],[461,462],[462,463],[463,464],[464,465],[465,466],[466,467],[467,468],[468,469],[469,470],[470,471],[471,472],[472,473],[473,474],[474,475],[475,476],[476,477],[477,478],[478,479],[479,480],[480,481],[481,482],[482,483],[483,484],[484,485],[485,486],[486,487],[487,488],[488,489],[489,490],[490,491],[491,492],[492,493],[493,494],[494,495],[495,496],[496,497],[497,498],[498,499],[499,500],[500,501],[501,502],[502,503],[503,504],[504,505],[505,506],[506,507],[507,508],[508,509],[509,510],[510,511],[511,512],[512,513],[513,514],[514,515],[515,516],[516,517],[517,518],[518,519],[519,520],[520,521],[521,522],[522,523],[523,524],[524,525],[525,526],[526,527],[527,528],[528,529],[529,530],[530,531],[531,532],[532,533],[533,534],[534,535],[535,536],[536,537],[537,538],[538,539],[539,540],[540,541],[541,542],[542,543],[543,544],[544,545],[545,546],[546,547],[547,548],[548,549],[549,550],[550,551],[551,552],[552,553],[553,554],[554,555],[555,556],[556,557],[557,558],[558,559],[559,560],[560,561],[561,562],[562,563],[563,564],[564,565],[565,566],[566,567],[567,568],[568,569],[569,570],[570,571],[571,572],[572,573],[573,574],[574,575],[575,576],[576,577],[577,578],[578,579],[579,580],[580,581],[581,582],[582,583],[583,584],[584,585],[585,586],[586,587],[587,588],[588,589],[589,590],[590,591],[591,592],[592,593],[593,594],[594,595],[595,596],[596,597],[597,598],[598,599],[599,600],[600,601],[601,602],[602,603],[603,604],[604,605],[605,606],[606,607],[607,608],[608,609],[609,610],[610,611],[611,612],[612,613],[613,614],[614,615],[615,616],[616,617],[617,618],[618,619],[619,620],[620,621],[621,622],[622,623],[623,624],[624,625],[625,626],[626,627],[627,628],[628,629],[629,630],[630,631],[631,632],[632,633],[633,634],[634,635],[635,636],[636,637],[637,638],[638,639],[639,640],[640,641],[641,642],[642,643],[643,644],[644,645],[645,646],[646,647],[647,648],[648,649],[649,650],[650,651],[651,652],[652,653],[653,654],[654,655],[655,656],[656,657],[657,658],[658,659],[659,660],[660,661],[661,662],[662,663],[663,664],[664,665],[665,666],[666,667],[667,668],[668,669],[669,670],[670,671],[671,672],[672,673],[673,674],[674,675],[675,676],[676,677],[677,678],[678,679],[679,680],[680,681],[681,682],[682,683],[683,684],[684,685],[685,686],[686,687],[687,688],[688,689],[689,690],[690,691],[691,692],[692,693],[693,694],[694,695],[695,696],[696,697],[697,698],[698,699],[699,700],[700,701],[701,702],[702,703],[703,704],[704,705],[705,706],[706,707],[707,708],[708,709],[709,710],[710,711],[711,712],[712,713],[713,714],[714,715],[715,716],[716,717],[717,718],[718,719],[719,720],[720,721],[721,722],[722,723],[723,724],[724,725],[725,726],[726,727],[727,728],[728,729],[729,730],[730,731],[731,732],[732,733],[733,734],[734,735],[735,736],[736,737],[737,738],[738,739],[739,740],[740,741],[741,742],[742,743],[743,744],[744,745],[745,746],[746,747],[747,748],[748,749],[749,750],[750,751],[751,752],[752,753],[753,754],[754,755],[755,756],[756,757],[757,758],[758,759],[759,760],[760,761],[761,762],[762,763],[763,764],[764,765],[765,766],[766,767],[767,768],[768,769],[769,770],[770,771],[771,772],[772,773],[773,774],[774,775],[775,776],[776,777],[777,778],[778,779],[779,780],[780,781],[781,782],[782,783],[783,784],[784,785],[785,786],[786,787],[787,788],[788,789],[789,790],[790,791],[791,792],[792,793],[793,794],[794,795],[795,796],[796,797],[797,798],[798,799],[799,800],[800,801],[801,802],[802,803],[803,804],[804,805],[805,806],[806,807],[807,808],[808,809],[809,810],[810,811],[811,812],[812,813],[813,814],[814,815],[815,816],[816,817],[817,818],[818,819],[819,820],[820,821],[821,822],[822,823],[823,824],[824,825],[825,826],[826,827],[827,828],[828,829],[829,830],[830,831],[831,832],[832,833],[833,834],[834,835],[835,836],[836,837],[837,838],[838,839],[839,840],[840,841],[841,842],[842,843],[843,844],[844,845],[845,846],[846,847],[847,848],[848,849],[849,850],[850,851],[851,852],[852,853],[853,854],[854,855],[855,856],[856,857],[857,858],[858,859],[859,860],[860,861],[861,862],[862,863],[863,864],[864,865],[865,866],[866,867],[867,868],[868,869],[869,870],[870,871],[871,872],[872,873],[873,874],[874,875],[875,876],[876,877],[877,878],[878,879],[879,880],[880,881],[881,882],[882,883],[883,884],[884,885],[885,886],[886,887],[887,888],[888,889],[889,890],[890,891],[891,892],[892,893],[893,894],[894,895],[895,896],[896,897],[897,898],[898,899],[899,900],[900,901],[901,902],[902,903],[903,904],[904,905],[905,906],[906,907],[907,908],[908,909],[909,910],[910,911],[911,912],[912,913],[913,914],[914,915],[915,916],[916,917],[917,918],[918,919],[919,920],[920,921],[921,922],[922,923],[923,924],[924,925],[925,926],[926,927],[927,928],[928,929],[929,930],[930,931],[931,932],[932,933],[933,934],[934,935],[935,936],[936,937],[937,938],[938,939],[939,940],[940,941],[941,942],[942,943],[943,944],[944,945],[945,946],[946,947],[947,948],[948,949],[949,950],[950,951],[951,952],[952,953],[953,954],[954,955],[955,956],[956,957],[957,958],[958,959],[959,960],[960,961],[961,962],[962,963],[963,964],[964,965],[965,966],[966,967],[967,968],[968,969],[969,970],[970,971],[971,972],[972,973],[973,974],[974,975],[975,976],[976,977],[977,978],[978,979],[979,980],[980,981],[981,982],[982,983],[983,984],[984,985],[985,986],[986,987],[987,988],[988,989],[989,990],[990,991],[991,992],[992,993],[993,994],[994,995],[995,996],[996,997],[997,998],[998,999],[999,1000],[1000,1001],[1001,1002],[1002,1003],[1003,1004],[1004,1005],[1005,1006],[1006,1007],[1007,1008],[1008,1009],[1009,1010],[1010,1011],[1011,1012],[1012,1013],[1013,1014],[1014,1015],[1015,1016],[1016,1017],[1017,1018],[1018,1019],[1019,1020],[1020,1021],[1021,1022],[1022,1023],[1023,1024],[1024,1025],[1025,1026],[1026,1027],[1027,1028],[1028,1029],[1029,1030],[1030,1031],[1031,1032],[1032,1033],[1033,1034],[1034,1035],[1035,1036],[1036,1037],[1037,1038],[1038,1039],[1039,1040],[1040,1041],[1041,1042],[1042,1043],[1043,1044],[1044,1045],[1045,1046],[1046,1047],[1047,1048],[1048,1049],[1049,1050],[1050,1051],[1051,1052],[1052,1053],[1053,1054],[1054,1055],[1055,1056],[1056,1057],[1057,1058],[1058,1059],[1059,1060],[1060,1061],[1061,1062],[1062,1063],[1063,1064],[1064,1065],[1065,1066],[1066,1067],[1067,1068],[1068,1069],[1069,1070],[1070,1071],[1071,1072],[1072,1073],[1073,1074],[1074,1075],[1075,1076],[1076,1077],[1077,1078],[1078,1079],[1079,1080],[1080,1081],[1081,1082],[1082,1083],[1083,1084],[1084,1085],[1085,1086],[1086,1087],[1087,1088],[1088,1089],[1089,1090],[1090,1091],[1091,1092],[1092,1093],[1093,1094],[1094,1095],[1095,1096],[1096,1097],[1097,1098],[1098,1099],[1099,1100],[1100,1101],[1101,1102],[1102,1103],[1103,1104],[1104,1105],[1105,1106],[1106,1107],[1107,1108],[1108,1109],[1109,1110],[1110,1111],[1111,1112],[1112,1113],[1113,1114],[1114,1115],[1115,1116],[1116,1117],[1117,1118],[1118,1119],[1119,1120],[1120,1121],[1121,1122],[1122,1123],[1123,1124],[1124,1125],[1125,1126],[1126,1127],[1127,1128],[1128,1129],[1129,1130],[1130,1131],[1131,1132],[1132,1133],[1133,1134],[1134,1135],[1135,1136],[1136,1137],[1137,1138],[1138,1139],[1139,1140],[1140,1141],[1141,1142],[1142,1143],[1143,1144],[1144,1145],[1145,1146],[1146,1147],[1147,1148],[1148,1149],[1149,1150],[1150,1151],[1151,1152],[1152,1153],[1153,1154],[1154,1155],[1155,1156],[1156,1157],[1157,1158],[1158,1159],[1159,1160],[1160,1161],[1161,1162],[1162,1163],[1163,1164],[1164,1165],[1165,1166],[1166,1167],[1167,1168],[1168,1169],[1169,1170],[1170,1171],[1171,1172],[1172,1173],[1173,1174],[1174,1175],[1175,1176],[1176,1177],[1177,1178],[1178,1179],[1179,1180],[1180,1181],[1181,1182],[1182,1183],[1183,1184],[1184,1185],[1185,1186],[1186,1187],[1187,1188],[1188,1189],[1189,1190],[1190,1191],[1191,1192],[1192,1193],[1193,1194],[1194,1195],[1195,1196],[1196,1197],[1197,1198],[1198,1199],[1199,1200],[1200,1201],[1201,1202],[1202,1203],[1203,1204],[1204,1205],[1205,1206],[1206,1207],[1207,1208],[1208,1209],[1209,1210],[1210,1211],[1211,1212],[1212,1213],[1213,1214],[1214,1215],[1215,1216],[1216,1217],[1217,1218],[1218,1219],[1219,1220],[1220,1221],[1221,1222],[1222,1223],[1223,1224],[1224,1225],[1225,1226],[1226,1227],[1227,1228],[1228,1229],[1229,1230],[1230,1231],[1231,1232],[1232,1233],[1233,1234],[1234,1235],[1235,1236],[1236,1237],[1237,1238],[1238,1239],[1239,1240],[1240,1241],[1241,1242],[1242,1243],[1243,1244],[1244,1245],[1245,1246],[1246,1247],[1247,1248],[1248,1249],[1249,1250],[1250,1251],[1251,1252],[1252,1253],[1253,1254],[1254,1255],[1255,1256],[1256,1257],[1257,1258],[1258,1259],[1259,1260],[1260,1261],[1261,1262],[1262,1263],[1263,1264],[1264,1265],[1265,1266],[1266,1267],[1267,1268],[1268,1269],[1269,1270],[1270,1271],[1271,1272],[1272,1273],[1273,1274],[1274,1275],[1275,1276],[1276,1277],[1277,1278],[1278,1279],[1279,1280],[1280,1281],[1281,1282],[1282,1283],[1283,1284],[1284,1285],[1285,1286],[1286,1287],[1287,1288],[1288,1289],[1289,1290],[1290,1291],[1291,1292],[1292,1293],[1293,1294],[1294,1295],[1295,1296],[1296,1297],[1297,1298],[1298,1299],[1299,1300],[1300,1301],[1301,1302],[1302,1303],[1303,1304],[1304,1305],[1305,1306],[1306,1307],[1307,1308],[1308,1309],[1309,1310],[1310,1311],[1311,1312],[1312,1313],[1313,1314],[1314,1315],[1315,1316],[1316,1317],[1317,1318],[1318,1319],[1319,1320],[1320,1321],[1321,1322],[1322,1323],[1323,1324],[1324,1325],[1325,1326],[1326,1327],[1327,1328],[1328,1329],[1329,1330],[1330,1331],[1331,1332],[1332,1333],[1333,1334],[1334,1335],[1335,1336],[1336,1337],[1337,1338],[1338,1339],[1339,1340],[1340,1341],[1341,1342],[1342,1343],[1343,1344],[1344,1345],[1345,1346],[1346,1347],[1347,1348],[1348,1349],[1349,1350],[1350,1351],[1351,1352],[1352,1353],[1353,1354],[1354,1355],[1355,1356],[1356,1357],[1357,1358],[1358,1359],[1359,1360],[1360,1361],[1361,1362],[1362,1363],[1363,1364],[1364,1365],[1365,1366],[1366,1367],[1367,1368],[1368,1369],[1369,1370],[1370,1371],[1371,1372],[1372,1373],[1373,1374],[1374,1375],[1375,1376],[1376,1377],[1377,1378],[1378,1379],[1379,1380],[1380,1381],[1381,1382],[1382,1383],[1383,1384],[1384,1385],[1385,1386],[1386,1387],[1387,1388],[1388,1389],[1389,1390],[1390,1391],[1391,1392],[1392,1393],[1393,1394],[1394,1395],[1395,1396],[1396,1397],[1397,1398],[1398,1399],[1399,1400],[1400,1401],[1401,1402],[1402,1403],[1403,1404],[1404,1405],[1405,1406],[1406,1407],[1407,1408],[1408,1409],[1409,1410],[1410,1411],[1411,1412],[1412,1413],[1413,1414],[1414,1415],[1415,1416],[1416,1417],[1417,1418],[1418,1419],[1419,1420],[1420,1421],[1421,1422],[1422,1423],[1423,1424],[1424,1425],[1425,1426],[1426,1427],[1427,1428],[1428,1429],[1429,1430],[1430,1431],[1431,1432],[1432,1433],[1433,1434],[1434,1435],[1435,1436],[1436,1437],[1437,1438],[1438,1439],[1439,1440],[1440,1441],[1441,1442],[1442,1443],[1443,1444],[1444,1445],[1445,1446],[1446,1447],[1447,1448],[1448,1449],[1449,1450],[1450,1451],[1451,1452],[1452,1453],[1453,1454],[1454,1455],[1455,1456],[1456,1457],[1457,1458],[1458,1459],[1459,1460],[1460,1461],[1461,1462],[1462,1463],[1463,1464],[1464,1465],[1465,1466],[1466,1467],[1467,1468],[1468,1469],[1469,1470],[1470,1471],[1471,1472],[1472,1473],[1473,1474],[1474,1475],[1475,1476],[1476,1477],[1477,1478],[1478,1479],[1479,1480],[1480,1481],[1481,1482],[1482,1483],[1483,1484],[1484,1485],[1485,1486],[1486,1487],[1487,1488],[1488,1489],[1489,1490],[1490,1491],[1491,1492],[1492,1493],[1493,1494],[1494,1495],[1495,1496],[1496,1497],[1497,1498],[1498,1499],[1499,1500],[1500,1501],[1501,1502],[1502,1503],[1503,1504],[1504,1505],[1505,1506],[1506,1507],[1507,1508],[1508,1509],[1509,1510],[1510,1511],[1511,1512],[1512,1513],[1513,1514],[1514,1515],[1515,1516],[1516,1517],[1517,1518],[1518,1519],[1519,1520],[1520,1521],[1521,1522],[1522,1523],[1523,1524],[1524,1525],[1525,1526],[1526,1527],[1527,1528],[1528,1529],[1529,1530],[1530,1531],[1531,1532],[1532,1533],[1533,1534],[1534,1535],[1535,1536],[1536,1537],[1537,1538],[1538,1539],[1539,1540],[1540,1541],[1541,1542],[1542,1543],[1543,1544],[1544,1545],[1545,1546],[1546,1547],[1547,1548],[1548,1549],[1549,1550],[1550,1551],[1551,1552],[1552,1553],[1553,1554],[1554,1555],[1555,1556],[1556,1557],[1557,1558],[1558,1559],[1559,1560],[1560,1561],[1561,1562],[1562,1563],[1563,1564],[1564,1565],[1565,1566],[1566,1567],[1567,1568],[1568,1569],[1569,1570],[1570,1571],[1571,1572],[1572,1573],[1573,1574],[1574,1575],[1575,1576],[1576,1577],[1577,1578],[1578,1579],[1579,1580],[1580,1581],[1581,1582],[1582,1583],[1583,1584],[1584,1585],[1585,1586],[1586,1587],[1587,1588],[1588,1589],[1589,1590],[1590,1591],[1591,1592],[1592,1593],[1593,1594],[1594,1595],[1595,1596],[1596,1597],[1597,1598],[1598,1599],[1599,1600],[1600,1601],[1601,1602],[1602,1603],[1603,1604],[1604,1605],[1605,1606],[1606,1607],[1607,1608],[1608,1609],[1609,1610],[1610,1611],[1611,1612],[1612,1613],[1613,1614],[1614,1615],[1615,1616],[1616,1617],[1617,1618],[1618,1619],[1619,1620],[1620,1621],[1621,1622],[1622,1623],[1623,1624],[1624,1625],[1625,1626],[1626,1627],[1627,1628],[1628,1629],[1629,1630],[1630,1631],[1631,1632],[1632,1633],[1633,1634],[1634,1635],[1635,1636],[1636,1637],[1637,1638],[1638,1639],[1639,1640],[1640,1641],[1641,1642],[1642,1643],[1643,1644],[1644,1645],[1645,1646],[1646,1647],[1647,1648],[1648,1649],[1649,1650],[1650,1651],[1651,1652],[1652,1653],[1653,1654],[1654,1655],[1655,1656],[1656,1657],[1657,1658],[1658,1659],[1659,1660],[1660,1661],[1661,1662],[1662,1663],[1663,1664],[1664,1665],[1665,1666],[1666,1667],[1667,1668],[1668,1669],[1669,1670],[1670,1671],[1671,1672],[1672,1673],[1673,1674],[1674,1675],[1675,1676],[1676,1677],[1677,1678],[1678,1679],[1679,1680],[1680,1681],[1681,1682],[1682,1683],[1683,1684],[1684,1685],[1685,1686],[1686,1687],[1687,1688],[1688,1689],[1689,1690],[1690,1691],[1691,1692],[1692,1693],[1693,1694],[1694,1695],[1695,1696],[1696,1697],[1697,1698],[1698,1699],[1699,1700],[1700,1701],[1701,1702],[1702,1703],[1703,1704],[1704,1705],[1705,1706],[1706,1707],[1707,1708],[1708,1709],[1709,1710],[1710,1711],[1711,1712],[1712,1713],[1713,1714],[1714,1715],[1715,1716],[1716,1717],[1717,1718],[1718,1719],[1719,1720],[1720,1721],[1721,1722],[1722,1723],[1723,1724],[1724,1725],[1725,1726],[1726,1727],[1727,1728],[1728,1729],[1729,1730],[1730,1731],[1731,1732],[1732,1733],[1733,1734],[1734,1735],[1735,1736],[1736,1737],[1737,1738],[1738,1739],[1739,1740],[1740,1741],[1741,1742],[1742,1743],[1743,1744],[1744,1745],[1745,1746],[1746,1747],[1747,1748],[1748,1749],[1749,1750],[1750,1751],[1751,1752],[1752,1753],[1753,1754],[1754,1755],[1755,1756],[1756,1757],[1757,1758],[1758,1759],[1759,1760],[1760,1761],[1761,1762],[1762,1763],[1763,1764],[1764,1765],[1765,1766],[1766,1767],[1767,1768],[1768,1769],[1769,1770],[1770,1771],[1771,1772],[1772,1773],[1773,1774],[1774,1775],[1775,1776],[1776,1777],[1777,1778],[1778,1779],[1779,1780],[1780,1781],[1781,1782],[1782,1783],[1783,1784],[1784,1785],[1785,1786],[1786,1787],[1787,1788],[1788,1789],[1789,1790],[1790,1791],[1791,1792],[1792,1793],[1793,1794],[1794,1795],[1795,1796],[1796,1797],[1797,1798],[1798,1799],[1799,1800],[1800,1801],[1801,1802],[1802,1803],[1803,1804],[1804,1805],[1805,1806],[1806,1807],[1807,1808],[1808,1809],[1809,1810],[1810,1811],[1811,1812],[1812,1813],[1813,1814],[1814,1815],[1815,1816],[1816,1817],[1817,1818],[1818,1819],[1819,1820],[1820,1821],[1821,1822],[1822,1823],[1823,1824],[1824,1825],[1825,1826],[1826,1827],[1827,1828],[1828,1829],[1829,1830],[1830,1831],[1831,1832],[1832,1833],[1833,1834],[1834,1835],[1835,1836],[1836,1837],[1837,1838],[1838,1839],[1839,1840],[1840,1841],[1841,1842],[1842,1843],[1843,1844],[1844,1845],[1845,1846],[1846,1847],[1847,1848],[1848,1849],[1849,1850],[1850,1851],[1851,1852],[1852,1853],[1853,1854],[1854,1855],[1855,1856],[1856,1857],[1857,1858],[1858,1859],[1859,1860],[1860,1861],[1861,1862],[1862,1863],[1863,1864],[1864,1865],[1865,1866],[1866,1867],[1867,1868],[1868,1869],[1869,1870],[1870,1871],[1871,1872],[1872,1873],[1873,1874],[1874,1875],[1875,1876],[1876,1877],[1877,1878],[1878,1879],[1879,1880],[1880,1881],[1881,1882],[1882,1883],[1883,1884],[1884,1885],[1885,1886],[1886,1887],[1887,1888],[1888,1889],[1889,1890],[1890,1891],[1891,1892],[1892,1893],[1893,1894],[1894,1895],[1895,1896],[1896,1897],[1897,1898],[1898,1899],[1899,1900],[1900,1901],[1901,1902],[1902,1903],[1903,1904],[1904,1905],[1905,1906],[1906,1907],[1907,1908],[1908,1909],[1909,1910],[1910,1911],[1911,1912],[1912,1913],[1913,1914],[1914,1915],[1915,1916],[1916,1917],[1917,1918],[1918,1919],[1919,1920],[1920,1921],[1921,1922],[1922,1923],[1923,1924],[1924,1925],[1925,1926],[1926,1927],[1927,1928],[1928,1929],[1929,1930],[1930,1931],[1931,1932],[1932,1933],[1933,1934],[1934,1935],[1935,1936],[1936,1937],[1937,1938],[1938,1939],[1939,1940],[1940,1941],[1941,1942],[1942,1943],[1943,1944],[1944,1945],[1945,1946],[1946,1947],[1947,1948],[1948,1949],[1949,1950],[1950,1951],[1951,1952],[1952,1953],[1953,1954],[1954,1955],[1955,1956],[1956,1957],[1957,1958],[1958,1959],[1959,1960],[1960,1961],[1961,1962],[1962,1963],[1963,1964],[1964,1965],[1965,1966],[1966,1967],[1967,1968],[1968,1969],[1969,1970],[1970,1971],[1971,1972],[1972,1973],[1973,1974],[1974,1975],[1975,1976],[1976,1977],[1977,1978],[1978,1979],[1979,1980],[1980,1981],[1981,1982],[1982,1983],[1983,1984],[1984,1985],[1985,1986],[1986,1987],[1987,1988],[1988,1989],[1989,1990],[1990,1991],[1991,1992],[1992,1993],[1993,1994],[1994,1995],[1995,1996],[1996,1997],[1997,1998],[1998,1999],[1999,2000],[2000,2001],[2001,2002],[2002,2003],[2003,2004],[2004,2005],[2005,2006],[2006,2007],[2007,2008],[2008,2009],[2009,2010],[2010,2011],[2011,2012],[2012,2013],[2013,2014],[2014,2015],[2015,2016],[2016,2017],[2017,2018],[2018,2019],[2019,2020],[2020,2021],[2021,2022],[2022,2023],[2023,2024],[2024,2025],[2025,2026],[2026,2027],[2027,2028],[2028,2029],[2029,2030],[2030,2031],[2031,2032],[2032,2033],[2033,2034],[2034,2035],[2035,2036],[2036,2037],[2037,2038],[2038,2039],[2039,2040],[2040,2041],[2041,2042],[2042,2043],[2043,2044],[2044,2045],[2045,2046],[2046,2047],[2047,2048],[2048,2049],[2049,2050],[2050,2051],[2051,2052],[2052,2053],[2053,2054],[2054,2055],[2055,2056],[2056,2057],[2057,2058],[2058,2059],[2059,2060],[2060,2061],[2061,2062],[2062,2063],[2063,2064],[2064,2065],[2065,2066],[2066,2067],[2067,2068],[2068,2069],[2069,2070],[2070,2071],[2071,2072],[2072,2073],[2073,2074],[2074,2075],[2075,2076],[2076,2077],[2077,2078],[2078,2079],[2079,2080],[2080,2081],[2081,2082],[2082,2083],[2083,2084],[2084,2085],[2085,2086],[2086,2087],[2087,2088],[2088,2089],[2089,2090],[2090,2091],[2091,2092],[2092,2093],[2093,2094],[2094,2095],[2095,2096],[2096,2097],[2097,2098],[2098,2099],[2099,2100],[2100,2101],[2101,2102],[2102,2103],[2103,2104],[2104,2105],[2105,2106],[2106,2107],[2107,2108],[2108,2109],[2109,2110],[2110,2111],[2111,2112],[2112,2113],[2113,2114],[2114,2115],[2115,2116],[2116,2117],[2117,2118],[2118,2119],[2119,2120],[2120,2121],[2121,2122],[2122,2123],[2123,2124],[2124,2125],[2125,2126],[2126,2127],[2127,2128],[2128,2129],[2129,2130],[2130,2131],[2131,2132],[2132,2133],[2133,2134],[2134,2135],[2135,2136],[2136,2137],[2137,2138],[2138,2139],[2139,2140],[2140,2141],[2141,2142],[2142,2143],[2143,2144],[2144,2145],[2145,2146],[2146,2147],[2147,2148],[2148,2149],[2149,2150],[2150,2151],[2151,2152],[2152,2153],[2153,2154],[2154,2155],[2155,2156],[2156,2157],[2157,2158],[2158,2159],[2159,2160],[2160,2161],[2161,2162],[2162,2163],[2163,2164],[2164,2165],[2165,2166],[2166,2167],[2167,2168],[2168,2169],[2169,2170],[2170,2171],[2171,2172],[2172,2173],[2173,2174],[2174,2175],[2175,2176],[2176,2177],[2177,2178],[2178,2179],[2179,2180],[2180,2181],[2181,2182],[2182,2183],[2183,2184],[2184,2185],[2185,2186],[2186,2187],[2187,2188],[2188,2189],[2189,2190],[2190,2191],[2191,2192],[2192,2193],[2193,2194],[2194,2195],[2195,2196],[2196,2197],[2197,2198],[2198,2199],[2199,2200],[2200,2201],[2201,2202],[2202,2203],[2203,2204],[2204,2205],[2205,2206],[2206,2207],[2207,2208],[2208,2209],[2209,2210],[2210,2211],[2211,2212],[2212,2213],[2213,2214],[2214,2215],[2215,2216],[2216,2217],[2217,2218],[2218,2219],[2219,2220],[2220,2221],[2221,2222],[2222,2223],[2223,2224],[2224,2225],[2225,2226],[2226,2227],[2227,2228],[2228,2229],[2229,2230],[2230,2231],[2231,2232],[2232,2233],[2233,2234],[2234,2235],[2235,2236],[2236,2237],[2237,2238],[2238,2239],[2239,2240],[2240,2241],[2241,2242],[2242,2243],[2243,2244],[2244,2245],[2245,2246],[2246,2247],[2247,2248],[2248,2249],[2249,2250],[2250,2251],[2251,2252],[2252,2253],[2253,2254],[2254,2255],[2255,2256],[2256,2257],[2257,2258],[2258,2259],[2259,2260],[2260,2261],[2261,2262],[2262,2263],[2263,2264],[2264,2265],[2265,2266],[2266,2267],[2267,2268],[2268,2269],[2269,2270],[2270,2271],[2271,2272],[2272,2273],[2273,2274],[2274,2275],[2275,2276],[2276,2277],[2277,2278],[2278,2279],[2279,2280],[2280,2281],[2281,2282],[2282,2283],[2283,2284],[2284,2285],[2285,2286],[2286,2287],[2287,2288],[2288,2289],[2289,2290],[2290,2291],[2291,2292],[2292,2293],[2293,2294],[2294,2295],[2295,2296],[2296,2297],[2297,2298],[2298,2299],[2299,2300],[2300,2301],[2301,2302],[2302,2303],[2303,2304],[2304,2305],[2305,2306],[2306,2307],[2307,2308],[2308,2309],[2309,2310],[2310,2311],[2311,2312],[2312,2313],[2313,2314],[2314,2315],[2315,2316],[2316,2317],[2317,2318],[2318,2319],[2319,2320],[2320,2321],[2321,2322],[2322,2323],[2323,2324],[2324,2325],[2325,2326],[2326,2327],[2327,2328],[2328,2329],[2329,2330],[2330,2331],[2331,2332],[2332,2333],[2333,2334],[2334,2335],[2335,2336],[2336,2337],[2337,2338],[2338,2339],[2339,2340],[2340,2341],[2341,2342],[2342,2343],[2343,2344],[2344,2345],[2345,2346],[2346,2347],[2347,2348],[2348,2349],[2349,2350],[2350,2351],[2351,2352],[2352,2353],[2353,2354],[2354,2355],[2355,2356],[2356,2357],[2357,2358],[2358,2359],[2359,2360],[2360,2361],[2361,2362],[2362,2363],[2363,2364],[2364,2365],[2365,2366],[2366,2367],[2367,2368],[2368,2369],[2369,2370],[2370,2371],[2371,2372],[2372,2373],[2373,2374],[2374,2375],[2375,2376],[2376,2377],[2377,2378],[2378,2379],[2379,2380],[2380,2381],[2381,2382],[2382,2383],[2383,2384],[2384,2385],[2385,2386],[2386,2387],[2387,2388],[2388,2389],[2389,2390],[2390,2391],[2391,2392],[2392,2393],[2393,2394],[2394,2395],[2395,2396],[2396,2397],[2397,2398],[2398,2399],[2399,2400],[2400,2401],[2401,2402],[2402,2403],[2403,2404],[2404,2405],[2405,2406],[2406,2407],[2407,2408],[2408,2409],[2409,2410],[2410,2411],[2411,2412],[2412,2413],[2413,2414],[2414,2415],[2415,2416],[2416,2417],[2417,2418],[2418,2419],[2419,2420],[2420,2421],[2421,2422],[2422,2423],[2423,2424],[2424,2425],[2425,2426],[2426,2427],[2427,2428],[2428,2429],[2429,2430],[2430,2431],[2431,2432],[2432,2433],[2433,2434],[2434,2435],[2435,2436],[2436,2437],[2437,2438],[2438,2439],[2439,2440],[2440,2441],[2441,2442],[2442,2443],[2443,2444],[2444,2445],[2445,2446],[2446,2447],[2447,2448],[2448,2449],[2449,2450],[2450,2451],[2451,2452],[2452,2453],[2453,2454],[2454,2455],[2455,2456],[2456,2457],[2457,2458],[2458,2459],[2459,2460],[2460,2461],[2461,2462],[2462,2463],[2463,2464],[2464,2465],[2465,2466],[2466,2467],[2467,2468],[2468,2469],[2469,2470],[2470,2471],[2471,2472],[2472,2473],[2473,2474],[2474,2475],[2475,2476],[2476,2477],[2477,2478],[2478,2479],[2479,2480],[2480,2481],[2481,2482],[2482,2483],[2483,2484],[2484,2485],[2485,2486],[2486,2487],[2487,2488],[2488,2489],[2489,2490],[2490,2491],[2491,2492],[2492,2493],[2493,2494],[2494,2495],[2495,2496],[2496,2497],[2497,2498],[2498,2499],[2499,2500],[2500,2501],[2501,2502],[2502,2503],[2503,2504],[2504,2505],[2505,2506],[2506,2507],[2507,2508],[2508,2509],[2509,2510],[2510,2511],[2511,2512],[2512,2513],[2513,2514],[2514,2515],[2515,2516],[2516,2517],[2517,2518],[2518,2519],[2519,2520],[2520,2521],[2521,2522],[2522,2523],[2523,2524],[2524,2525],[2525,2526],[2526,2527],[2527,2528],[2528,2529],[2529,2530],[2530,2531],[2531,2532],[2532,2533],[2533,2534],[2534,2535],[2535,2536],[2536,2537],[2537,2538],[2538,2539],[2539,2540],[2540,2541],[2541,2542],[2542,2543],[2543,2544],[2544,2545],[2545,2546],[2546,2547],[2547,2548],[2548,2549],[2549,2550],[2550,2551],[2551,2552],[2552,2553],[2553,2554],[2554,2555],[2555,2556],[2556,2557],[2557,2558],[2558,2559],[2559,2560],[2560,2561],[2561,2562],[2562,2563],[2563,2564],[2564,2565],[2565,2566],[2566,2567],[2567,2568],[2568,2569],[2569,2570],[2570,2571],[2571,2572],[2572,2573],[2573,2574],[2574,2575],[2575,2576],[2576,2577],[2577,2578],[2578,2579],[2579,2580],[2580,2581],[2581,2582],[2582,2583],[2583,2584],[2584,2585],[2585,2586],[2586,2587],[2587,2588],[2588,2589],[2589,2590],[2590,2591],[2591,2592],[2592,2593],[2593,2594],[2594,2595],[2595,2596],[2596,2597],[2597,2598],[2598,2599],[2599,2600],[2600,2601],[2601,2602],[2602,2603],[2603,2604],[2604,2605],[2605,2606],[2606,2607],[2607,2608],[2608,2609],[2609,2610],[2610,2611],[2611,2612],[2612,2613],[2613,2614],[2614,2615],[2615,2616],[2616,2617],[2617,2618],[2618,2619],[2619,2620],[2620,2621],[2621,2622],[2622,2623],[2623,2624],[2624,2625],[2625,2626],[2626,2627],[2627,2628],[2628,2629],[2629,2630],[2630,2631],[2631,2632],[2632,2633],[2633,2634],[2634,2635],[2635,2636],[2636,2637],[2637,2638],[2638,2639],[2639,2640],[2640,2641],[2641,2642],[2642,2643],[2643,2644],[2644,2645],[2645,2646],[2646,2647],[2647,2648],[2648,2649],[2649,2650],[2650,2651],[2651,2652],[2652,2653],[2653,2654],[2654,2655],[2655,2656],[2656,2657],[2657,2658],[2658,2659],[2659,2660],[2660,2661],[2661,2662],[2662,2663],[2663,2664],[2664,2665],[2665,2666],[2666,2667],[2667,2668],[2668,2669],[2669,2670],[2670,2671],[2671,2672],[2672,2673],[2673,2674],[2674,2675],[2675,2676],[2676,2677],[2677,2678],[2678,2679],[2679,2680],[2680,2681],[2681,2682],[2682,2683],[2683,2684],[2684,2685],[2685,2686],[2686,2687],[2687,2688],[2688,2689],[2689,2690],[2690,2691],[2691,2692],[2692,2693],[2693,2694],[2694,2695],[2695,2696],[2696,2697],[2697,2698],[2698,2699],[2699,2700],[2700,2701],[2701,2702],[2702,2703],[2703,2704],[2704,2705],[2705,2706],[2706,2707],[2707,2708],[2708,2709],[2709,2710],[2710,2711],[2711,2712],[2712,2713],[2713,2714],[2714,2715],[2715,2716],[2716,2717],[2717,2718],[2718,2719],[2719,2720],[2720,2721],[2721,2722],[2722,2723],[2723,2724],[2724,2725],[2725,2726],[2726,2727],[2727,2728],[2728,2729],[2729,2730],[2730,2731],[2731,2732],[2732,2733],[2733,2734],[2734,2735],[2735,2736],[2736,2737],[2737,2738],[2738,2739],[2739,2740],[2740,2741],[2741,2742],[2742,2743],[2743,2744],[2744,2745],[2745,2746],[2746,2747],[2747,2748],[2748,2749],[2749,2750],[2750,2751],[2751,2752],[2752,2753],[2753,2754],[2754,2755],[2755,2756],[2756,2757],[2757,2758],[2758,2759],[2759,2760],[2760,2761],[2761,2762],[2762,2763],[2763,2764],[2764,2765],[2765,2766],[2766,2767],[2767,2768],[2768,2769],[2769,2770],[2770,2771],[2771,2772],[2772,2773],[2773,2774],[2774,2775],[2775,2776],[2776,2777],[2777,2778],[2778,2779],[2779,2780],[2780,2781],[2781,2782],[2782,2783],[2783,2784],[2784,2785],[2785,2786],[2786,2787],[2787,2788],[2788,2789],[2789,2790],[2790,2791],[2791,2792],[2792,2793],[2793,2794],[2794,2795],[2795,2796],[2796,2797],[2797,2798],[2798,2799],[2799,2800],[2800,2801],[2801,2802],[2802,2803],[2803,2804],[2804,2805],[2805,2806],[2806,2807],[2807,2808],[2808,2809],[2809,2810],[2810,2811],[2811,2812],[2812,2813],[2813,2814],[2814,2815],[2815,2816],[2816,2817],[2817,2818],[2818,2819],[2819,2820],[2820,2821],[2821,2822],[2822,2823],[2823,2824],[2824,2825],[2825,2826],[2826,2827],[2827,2828],[2828,2829],[2829,2830],[2830,2831],[2831,2832],[2832,2833],[2833,2834],[2834,2835],[2835,2836],[2836,2837],[2837,2838],[2838,2839],[2839,2840],[2840,2841],[2841,2842],[2842,2843],[2843,2844],[2844,2845],[2845,2846],[2846,2847],[2847,2848],[2848,2849],[2849,2850],[2850,2851],[2851,2852],[2852,2853],[2853,2854],[2854,2855],[2855,2856],[2856,2857],[2857,2858],[2858,2859],[2859,2860],[2860,2861],[2861,2862],[2862,2863],[2863,2864],[2864,2865],[2865,2866],[2866,2867],[2867,2868],[2868,2869],[2869,2870],[2870,2871],[2871,2872],[2872,2873],[2873,2874],[2874,2875],[2875,2876],[2876,2877],[2877,2878],[2878,2879],[2879,2880],[2880,2881],[2881,2882],[2882,2883],[2883,2884],[2884,2885],[2885,2886],[2886,2887],[2887,2888],[2888,2889],[2889,2890],[2890,2891],[2891,2892],[2892,2893],[2893,2894],[2894,2895],[2895,2896],[2896,2897],[2897,2898],[2898,2899],[2899,2900],[2900,2901],[2901,2902],[2902,2903],[2903,2904],[2904,2905],[2905,2906],[2906,2907],[2907,2908],[2908,2909],[2909,2910],[2910,2911],[2911,2912],[2912,2913],[2913,2914],[2914,2915],[2915,2916],[2916,2917],[2917,2918],[2918,2919],[2919,2920],[2920,2921],[2921,2922],[2922,2923],[2923,2924],[2924,2925],[2925,2926],[2926,2927],[2927,2928],[2928,2929],[2929,2930],[2930,2931],[2931,2932],[2932,2933],[2933,2934],[2934,2935],[2935,2936],[2936,2937],[2937,2938],[2938,2939],[2939,2940],[2940,2941],[2941,2942],[2942,2943],[2943,2944],[2944,2945],[2945,2946],[2946,2947],[2947,2948],[2948,2949],[2949,2950],[2950,2951],[2951,2952],[2952,2953],[2953,2954],[2954,2955],[2955,2956],[2956,2957],[2957,2958],[2958,2959],[2959,2960],[2960,2961],[2961,2962],[2962,2963],[2963,2964],[2964,2965],[2965,2966],[2966,2967],[2967,2968],[2968,2969],[2969,2970],[2970,2971],[2971,2972],[2972,2973],[2973,2974],[2974,2975],[2975,2976],[2976,2977],[2977,2978],[2978,2979],[2979,2980],[2980,2981],[2981,2982],[2982,2983],[2983,2984],[2984,2985],[2985,2986],[2986,2987],[2987,2988],[2988,2989],[2989,2990],[2990,2991],[2991,2992],[2992,2993],[2993,2994],[2994,2995],[2995,2996],[2996,2997],[2997,2998],[2998,2999],[2999,3000],[3000,3001],[3001,3002],[3002,3003],[3003,3004],[3004,3005],[3005,3006],[3006,3007],[3007,3008],[3008,3009],[3009,3010],[3010,3011],[3011,3012],[3012,3013],[3013,3014],[3014,3015],[3015,3016],[3016,3017],[3017,3018],[3018,3019],[3019,3020],[3020,3021],[3021,3022],[3022,3023],[3023,3024],[3024,3025],[3025,3026],[3026,3027],[3027,3028],[3028,3029],[3029,3030],[3030,3031],[3031,3032],[3032,3033],[3033,3034],[3034,3035],[3035,3036],[3036,3037],[3037,3038],[3038,3039],[3039,3040],[3040,3041],[3041,3042],[3042,3043],[3043,3044],[3044,3045],[3045,3046],[3046,3047],[3047,3048],[3048,3049],[3049,3050],[3050,3051],[3051,3052],[3052,3053],[3053,3054],[3054,3055],[3055,3056],[3056,3057],[3057,3058],[3058,3059],[3059,3060],[3060,3061],[3061,3062],[3062,3063],[3063,3064],[3064,3065],[3065,3066],[3066,3067],[3067,3068],[3068,3069],[3069,3070],[3070,3071],[3071,3072],[3072,3073],[3073,3074],[3074,3075],[3075,3076],[3076,3077],[3077,3078],[3078,3079],[3079,3080],[3080,3081],[3081,3082],[3082,3083],[3083,3084],[3084,3085],[3085,3086],[3086,3087],[3087,3088],[3088,3089],[3089,3090],[3090,3091],[3091,3092],[3092,3093],[3093,3094],[3094,3095],[3095,3096],[3096,3097],[3097,3098],[3098,3099],[3099,3100],[3100,3101],[3101,3102],[3102,3103],[3103,3104],[3104,3105],[3105,3106],[3106,3107],[3107,3108],[3108,3109],[3109,3110],[3110,3111],[3111,3112],[3112,3113],[3113,3114],[3114,3115],[3115,3116],[3116,3117],[3117,3118],[3118,3119],[3119,3120],[3120,3121],[3121,3122],[3122,3123],[3123,3124],[3124,3125],[3125,3126],[3126,3127],[3127,3128],[3128,3129],[3129,3130],[3130,3131],[3131,3132],[3132,3133],[3133,3134],[3134,3135],[3135,3136],[3136,3137],[3137,3138],[3138,3139],[3139,3140],[3140,3141],[3141,3142],[3142,3143],[3143,3144],[3144,3145],[3145,3146],[3146,3147],[3147,3148],[3148,3149],[3149,3150],[3150,3151],[3151,3152],[3152,3153],[3153,3154],[3154,3155],[3155,3156],[3156,3157],[3157,3158],[3158,3159],[3159,3160],[3160,3161],[3161,3162],[3162,3163],[3163,3164],[3164,3165],[3165,3166],[3166,3167],[3167,3168],[3168,3169],[3169,3170],[3170,3171],[3171,3172],[3172,3173],[3173,3174],[3174,3175],[3175,3176],[3176,3177],[3177,3178],[3178,3179],[3179,3180],[3180,3181],[3181,3182],[3182,3183],[3183,3184],[3184,3185],[3185,3186],[3186,3187],[3187,3188],[3188,3189],[3189,3190],[3190,3191],[3191,3192],[3192,3193],[3193,3194],[3194,3195],[3195,3196],[3196,3197],[3197,3198],[3198,3199],[3199,3200],[3200,3201],[3201,3202],[3202,3203],[3203,3204],[3204,3205],[3205,3206],[3206,3207],[3207,3208],[3208,3209],[3209,3210],[3210,3211],[3211,3212],[3212,3213],[3213,3214],[3214,3215],[3215,3216],[3216,3217],[3217,3218],[3218,3219],[3219,3220],[3220,3221],[3221,3222],[3222,3223],[3223,3224],[3224,3225],[3225,3226],[3226,3227],[3227,3228],[3228,3229],[3229,3230],[3230,3231],[3231,3232],[3232,3233],[3233,3234],[3234,3235],[3235,3236],[3236,3237],[3237,3238],[3238,3239],[3239,3240],[3240,3241],[3241,3242],[3242,3243],[3243,3244],[3244,3245],[3245,3246],[3246,3247],[3247,3248],[3248,3249],[3249,3250],[3250,3251],[3251,3252],[3252,3253],[3253,3254],[3254,3255],[3255,3256],[3256,3257],[3257,3258],[3258,3259],[3259,3260],[3260,3261],[3261,3262],[3262,3263],[3263,3264],[3264,3265],[3265,3266],[3266,3267],[3267,3268],[3268,3269],[3269,3270],[3270,3271],[3271,3272],[3272,3273],[3273,3274],[3274,3275],[3275,3276],[3276,3277],[3277,3278],[3278,3279],[3279,3280],[3280,3281],[3281,3282],[3282,3283],[3283,3284],[3284,3285],[3285,3286],[3286,3287],[3287,3288],[3288,3289],[3289,3290],[3290,3291],[3291,3292],[3292,3293],[3293,3294],[3294,3295],[3295,3296],[3296,3297],[3297,3298],[3298,3299],[3299,3300],[3300,3301],[3301,3302],[3302,3303],[3303,3304],[3304,3305],[3305,3306],[3306,3307],[3307,3308],[3308,3309],[3309,3310],[3310,3311],[3311,3312],[3312,3313],[3313,3314],[3314,3315],[3315,3316],[3316,3317],[3317,3318],[3318,3319],[3319,3320],[3320,3321],[3321,3322],[3322,3323],[3323,3324],[3324,3325],[3325,3326],[3326,3327],[3327,3328],[3328,3329],[3329,3330],[3330,3331],[3331,3332],[3332,3333],[3333,3334],[3334,3335],[3335,3336],[3336,3337],[3337,3338],[3338,3339],[3339,3340],[3340,3341],[3341,3342],[3342,3343],[3343,3344],[3344,3345],[3345,3346],[3346,3347],[3347,3348],[3348,3349],[3349,3350],[3350,3351],[3351,3352],[3352,3353],[3353,3354],[3354,3355],[3355,3356],[3356,3357],[3357,3358],[3358,3359],[3359,3360],[3360,3361],[3361,3362],[3362,3363],[3363,3364],[3364,3365],[3365,3366],[3366,3367],[3367,3368],[3368,3369],[3369,3370],[3370,3371],[3371,3372],[3372,3373],[3373,3374],[3374,3375],[3375,3376],[3376,3377],[3377,3378],[3378,3379],[3379,3380],[3380,3381],[3381,3382],[3382,3383],[3383,3384],[3384,3385],[3385,3386],[3386,3387],[3387,3388],[3388,3389],[3389,3390],[3390,3391],[3391,3392],[3392,3393],[3393,3394],[3394,3395],[3395,3396],[3396,3397],[3397,3398],[3398,3399],[3399,3400],[3400,3401],[3401,3402],[3402,3403],[3403,3404],[3404,3405],[3405,3406],[3406,3407],[3407,3408],[3408,3409],[3409,3410],[3410,3411],[3411,3412],[3412,3413],[3413,3414],[3414,3415],[3415,3416],[3416,3417],[3417,3418],[3418,3419],[3419,3420],[3420,3421],[3421,3422],[3422,3423],[3423,3424],[3424,3425],[3425,3426],[3426,3427],[3427,3428],[3428,3429],[3429,3430],[3430,3431],[3431,3432],[3432,3433],[3433,3434],[3434,3435],[3435,3436],[3436,3437],[3437,3438],[3438,3439],[3439,3440],[3440,3441],[3441,3442],[3442,3443],[3443,3444],[3444,3445],[3445,3446],[3446,3447],[3447,3448],[3448,3449],[3449,3450],[3450,3451],[3451,3452],[3452,3453],[3453,3454],[3454,3455],[3455,3456],[3456,3457],[3457,3458],[3458,3459],[3459,3460],[3460,3461],[3461,3462],[3462,3463],[3463,3464],[3464,3465],[3465,3466],[3466,3467],[3467,3468],[3468,3469],[3469,3470],[3470,3471],[3471,3472],[3472,3473],[3473,3474],[3474,3475],[3475,3476],[3476,3477],[3477,3478],[3478,3479],[3479,3480],[3480,3481],[3481,3482],[3482,3483],[3483,3484],[3484,3485],[3485,3486],[3486,3487],[3487,3488],[3488,3489],[3489,3490],[3490,3491],[3491,3492],[3492,3493],[3493,3494],[3494,3495],[3495,3496],[3496,3497],[3497,3498],[3498,3499],[3499,3500],[3500,3501],[3501,3502],[3502,3503],[3503,3504],[3504,3505],[3505,3506],[3506,3507],[3507,3508],[3508,3509],[3509,3510],[3510,3511],[3511,3512],[3512,3513],[3513,3514],[3514,3515],[3515,3516],[3516,3517],[3517,3518],[3518,3519],[3519,3520],[3520,3521],[3521,3522],[3522,3523],[3523,3524],[3524,3525],[3525,3526],[3526,3527],[3527,3528],[3528,3529],[3529,3530],[3530,3531],[3531,3532],[3532,3533],[3533,3534],[3534,3535],[3535,3536],[3536,3537],[3537,3538],[3538,3539],[3539,3540],[3540,3541],[3541,3542],[3542,3543],[3543,3544],[3544,3545],[3545,3546],[3546,3547],[3547,3548],[3548,3549],[3549,3550],[3550,3551],[3551,3552],[3552,3553],[3553,3554],[3554,3555],[3555,3556],[3556,3557],[3557,3558],[3558,3559],[3559,3560],[3560,3561],[3561,3562],[3562,3563],[3563,3564],[3564,3565],[3565,3566],[3566,3567],[3567,3568],[3568,3569],[3569,3570],[3570,3571],[3571,3572],[3572,3573],[3573,3574],[3574,3575],[3575,3576],[3576,3577],[3577,3578],[3578,3579],[3579,3580],[3580,3581],[3581,3582],[3582,3583],[3583,3584],[3584,3585],[3585,3586],[3586,3587],[3587,3588],[3588,3589],[3589,3590],[3590,3591],[3591,3592],[3592,3593],[3593,3594],[3594,3595],[3595,3596],[3596,3597],[3597,3598],[3598,3599],[3599,3600],[3600,3601],[3601,3602],[3602,3603],[3603,3604],[3604,3605],[3605,3606],[3606,3607],[3607,3608],[3608,3609],[3609,3610],[3610,3611],[3611,3612],[3612,3613],[3613,3614],[3614,3615],[3615,3616],[3616,3617],[3617,3618],[3618,3619],[3619,3620],[3620,3621],[3621,3622],[3622,3623],[3623,3624],[3624,3625],[3625,3626],[3626,3627],[3627,3628],[3628,3629],[3629,3630],[3630,3631],[3631,3632],[3632,3633],[3633,3634],[3634,3635],[3635,3636],[3636,3637],[3637,3638],[3638,3639],[3639,3640],[3640,3641],[3641,3642],[3642,3643],[3643,3644],[3644,3645],[3645,3646],[3646,3647],[3647,3648],[3648,3649],[3649,3650],[3650,3651],[3651,3652],[3652,3653],[3653,3654],[3654,3655],[3655,3656],[3656,3657],[3657,3658],[3658,3659],[3659,3660],[3660,3661],[3661,3662],[3662,3663],[3663,3664],[3664,3665],[3665,3666],[3666,3667],[3667,3668],[3668,3669],[3669,3670],[3670,3671],[3671,3672],[3672,3673],[3673,3674],[3674,3675],[3675,3676],[3676,3677],[3677,3678],[3678,3679],[3679,3680],[3680,3681],[3681,3682],[3682,3683],[3683,3684],[3684,3685],[3685,3686],[3686,3687],[3687,3688],[3688,3689],[3689,3690],[3690,3691],[3691,3692],[3692,3693],[3693,3694],[3694,3695],[3695,3696],[3696,3697],[3697,3698],[3698,3699],[3699,3700],[3700,3701],[3701,3702],[3702,3703],[3703,3704],[3704,3705],[3705,3706],[3706,3707],[3707,3708],[3708,3709],[3709,3710],[3710,3711],[3711,3712],[3712,3713],[3713,3714],[3714,3715],[3715,3716],[3716,3717],[3717,3718],[3718,3719],[3719,3720],[3720,3721],[3721,3722],[3722,3723],[3723,3724],[3724,3725],[3725,3726],[3726,3727],[3727,3728],[3728,3729],[3729,3730],[3730,3731],[3731,3732],[3732,3733],[3733,3734],[3734,3735],[3735,3736],[3736,3737],[3737,3738],[3738,3739],[3739,3740],[3740,3741],[3741,3742],[3742,3743],[3743,3744],[3744,3745],[3745,3746],[3746,3747],[3747,3748],[3748,3749],[3749,3750],[3750,3751],[3751,3752],[3752,3753],[3753,3754],[3754,3755],[3755,3756],[3756,3757],[3757,3758],[3758,3759],[3759,3760],[3760,3761],[3761,3762],[3762,3763],[3763,3764],[3764,3765],[3765,3766],[3766,3767],[3767,3768],[3768,3769],[3769,3770],[3770,3771],[3771,3772],[3772,3773],[3773,3774],[3774,3775],[3775,3776],[3776,3777],[3777,3778],[3778,3779],[3779,3780],[3780,3781],[3781,3782],[3782,3783],[3783,3784],[3784,3785],[3785,3786],[3786,3787],[3787,3788],[3788,3789],[3789,3790],[3790,3791],[3791,3792],[3792,3793],[3793,3794],[3794,3795],[3795,3796],[3796,3797],[3797,3798],[3798,3799],[3799,3800],[3800,3801],[3801,3802],[3802,3803],[3803,3804],[3804,3805],[3805,3806],[3806,3807],[3807,3808],[3808,3809],[3809,3810],[3810,3811],[3811,3812],[3812,3813],[3813,3814],[3814,3815],[3815,3816],[3816,3817],[3817,3818],[3818,3819],[3819,3820],[3820,3821],[3821,3822],[3822,3823],[3823,3824],[3824,3825],[3825,3826],[3826,3827],[3827,3828],[3828,3829],[3829,3830],[3830,3831],[3831,3832],[3832,3833],[3833,3834],[3834,3835],[3835,3836],[3836,3837],[3837,3838],[3838,3839],[3839,3840],[3840,3841],[3841,3842],[3842,3843],[3843,3844],[3844,3845],[3845,3846],[3846,3847],[3847,3848],[3848,3849],[3849,3850],[3850,3851],[3851,3852],[3852,3853],[3853,3854],[3854,3855],[3855,3856],[3856,3857],[3857,3858],[3858,3859],[3859,3860],[3860,3861],[3861,3862],[3862,3863],[3863,3864],[3864,3865],[3865,3866],[3866,3867],[3867,3868],[3868,3869],[3869,3870],[3870,3871],[3871,3872],[3872,3873],[3873,3874],[3874,3875],[3875,3876],[3876,3877],[3877,3878],[3878,3879],[3879,3880],[3880,3881],[3881,3882],[3882,3883],[3883,3884],[3884,3885],[3885,3886],[3886,3887],[3887,3888],[3888,3889],[3889,3890],[3890,3891],[3891,3892],[3892,3893],[3893,3894],[3894,3895],[3895,3896],[3896,3897],[3897,3898],[3898,3899],[3899,3900],[3900,3901],[3901,3902],[3902,3903],[3903,3904],[3904,3905],[3905,3906],[3906,3907],[3907,3908],[3908,3909],[3909,3910],[3910,3911],[3911,3912],[3912,3913],[3913,3914],[3914,3915],[3915,3916],[3916,3917],[3917,3918],[3918,3919],[3919,3920],[3920,3921],[3921,3922],[3922,3923],[3923,3924],[3924,3925],[3925,3926],[3926,3927],[3927,3928],[3928,3929],[3929,3930],[3930,3931],[3931,3932],[3932,3933],[3933,3934],[3934,3935],[3935,3936],[3936,3937],[3937,3938],[3938,3939],[3939,3940],[3940,3941],[3941,3942],[3942,3943],[3943,3944],[3944,3945],[3945,3946],[3946,3947],[3947,3948],[3948,3949],[3949,3950],[3950,3951],[3951,3952],[3952,3953],[3953,3954],[3954,3955],[3955,3956],[3956,3957],[3957,3958],[3958,3959],[3959,3960],[3960,3961],[3961,3962],[3962,3963],[3963,3964],[3964,3965],[3965,3966],[3966,3967],[3967,3968],[3968,3969],[3969,3970],[3970,3971],[3971,3972],[3972,3973],[3973,3974],[3974,3975],[3975,3976],[3976,3977],[3977,3978],[3978,3979],[3979,3980],[3980,3981],[3981,3982],[3982,3983],[3983,3984],[3984,3985],[3985,3986],[3986,3987],[3987,3988],[3988,3989],[3989,3990],[3990,3991],[3991,3992],[3992,3993],[3993,3994],[3994,3995],[3995,3996],[3996,3997],[3997,3998],[3998,3999],[3999,4000],[4000,4001],[4001,4002],[4002,4003],[4003,4004],[4004,4005],[4005,4006],[4006,4007],[4007,4008],[4008,4009],[4009,4010],[4010,4011],[4011,4012],[4012,4013],[4013,4014],[4014,4015],[4015,4016],[4016,4017],[4017,4018],[4018,4019],[4019,4020],[4020,4021],[4021,4022],[4022,4023],[4023,4024],[4024,4025],[4025,4026],[4026,4027],[4027,4028],[4028,4029],[4029,4030],[4030,4031],[4031,4032],[4032,4033],[4033,4034],[4034,4035],[4035,4036],[4036,4037],[4037,4038],[4038,4039],[4039,4040],[4040,4041],[4041,4042],[4042,4043],[4043,4044],[4044,4045],[4045,4046],[4046,4047],[4047,4048],[4048,4049],[4049,4050],[4050,4051],[4051,4052],[4052,4053],[4053,4054],[4054,4055],[4055,4056],[4056,4057],[4057,4058],[4058,4059],[4059,4060],[4060,4061],[4061,4062],[4062,4063],[4063,4064],[4064,4065],[4065,4066],[4066,4067],[4067,4068],[4068,4069],[4069,4070],[4070,4071],[4071,4072],[4072,4073],[4073,4074],[4074,4075],[4075,4076],[4076,4077],[4077,4078],[4078,4079],[4079,4080],[4080,4081],[4081,4082],[4082,4083],[4083,4084],[4084,4085],[4085,4086],[4086,4087],[4087,4088],[4088,4089],[4089,4090],[4090,4091],[4091,4092],[4092,4093],[4093,4094],[4094,4095],[4095,4096],[4096,4097],[4097,4098],[4098,4099],[4099,4100],[4100,4101],[4101,4102],[4102,4103],[4103,4104],[4104,4105],[4105,4106],[4106,4107],[4107,4108],[4108,4109],[4109,4110],[4110,4111],[4111,4112],[4112,4113],[4113,4114],[4114,4115],[4115,4116],[4116,4117],[4117,4118],[4118,4119],[4119,4120],[4120,4121],[4121,4122],[4122,4123],[4123,4124],[4124,4125],[4125,4126],[4126,4127],[4127,4128],[4128,4129],[4129,4130],[4130,4131],[4131,4132],[4132,4133],[4133,4134],[4134,4135],[4135,4136],[4136,4137],[4137,4138],[4138,4139],[4139,4140],[4140,4141],[4141,4142],[4142,4143],[4143,4144],[4144,4145],[4145,4146],[4146,4147],[4147,4148],[4148,4149],[4149,4150],[4150,4151],[4151,4152],[4152,4153],[4153,4154],[4154,4155],[4155,4156],[4156,4157],[4157,4158],[4158,4159],[4159,4160],[4160,4161],[4161,4162],[4162,4163],[4163,4164],[4164,4165],[4165,4166],[4166,4167],[4167,4168],[4168,4169],[4169,4170],[4170,4171],[4171,4172],[4172,4173],[4173,4174],[4174,4175],[4175,4176],[4176,4177],[4177,4178],[4178,4179],[4179,4180],[4180,4181],[4181,4182],[4182,4183],[4183,4184],[4184,4185],[4185,4186],[4186,4187],[4187,4188],[4188,4189],[4189,4190],[4190,4191],[4191,4192],[4192,4193],[4193,4194],[4194,4195],[4195,4196],[4196,4197],[4197,4198],[4198,4199],[4199,4200],[4200,4201],[4201,4202],[4202,4203],[4203,4204],[4204,4205],[4205,4206],[4206,4207],[4207,4208],[4208,4209],[4209,4210],[4210,4211],[4211,4212],[4212,4213],[4213,4214],[4214,4215],[4215,4216],[4216,4217],[4217,4218],[4218,4219],[4219,4220],[4220,4221],[4221,4222],[4222,4223],[4223,4224],[4224,4225],[4225,4226],[4226,4227],[4227,4228],[4228,4229],[4229,4230],[4230,4231],[4231,4232],[4232,4233],[4233,4234],[4234,4235],[4235,4236],[4236,4237],[4237,4238],[4238,4239],[4239,4240],[4240,4241],[4241,4242],[4242,4243],[4243,4244],[4244,4245],[4245,4246],[4246,4247],[4247,4248],[4248,4249],[4249,4250],[4250,4251],[4251,4252],[4252,4253],[4253,4254],[4254,4255],[4255,4256],[4256,4257],[4257,4258],[4258,4259],[4259,4260],[4260,4261],[4261,4262],[4262,4263],[4263,4264],[4264,4265],[4265,4266],[4266,4267],[4267,4268],[4268,4269],[4269,4270],[4270,4271],[4271,4272],[4272,4273],[4273,4274],[4274,4275],[4275,4276],[4276,4277],[4277,4278],[4278,4279],[4279,4280],[4280,4281],[4281,4282],[4282,4283],[4283,4284],[4284,4285],[4285,4286],[4286,4287],[4287,4288],[4288,4289],[4289,4290],[4290,4291],[4291,4292],[4292,4293],[4293,4294],[4294,4295],[4295,4296],[4296,4297],[4297,4298],[4298,4299],[4299,4300],[4300,4301],[4301,4302],[4302,4303],[4303,4304],[4304,4305],[4305,4306],[4306,4307],[4307,4308],[4308,4309],[4309,4310],[4310,4311],[4311,4312],[4312,4313],[4313,4314],[4314,4315],[4315,4316],[4316,4317],[4317,4318],[4318,4319],[4319,4320],[4320,4321],[4321,4322],[4322,4323],[4323,4324],[4324,4325],[4325,4326],[4326,4327],[4327,4328],[4328,4329],[4329,4330],[4330,4331],[4331,4332],[4332,4333],[4333,4334],[4334,4335],[4335,4336],[4336,4337],[4337,4338],[4338,4339],[4339,4340],[4340,4341],[4341,4342],[4342,4343],[4343,4344],[4344,4345],[4345,4346],[4346,4347],[4347,4348],[4348,4349],[4349,4350],[4350,4351],[4351,4352],[4352,4353],[4353,4354],[4354,4355],[4355,4356],[4356,4357],[4357,4358],[4358,4359],[4359,4360],[4360,4361],[4361,4362],[4362,4363],[4363,4364],[4364,4365],[4365,4366],[4366,4367],[4367,4368],[4368,4369],[4369,4370],[4370,4371],[4371,4372],[4372,4373],[4373,4374],[4374,4375],[4375,4376],[4376,4377],[4377,4378],[4378,4379],[4379,4380],[4380,4381],[4381,4382],[4382,4383],[4383,4384],[4384,4385],[4385,4386],[4386,4387],[4387,4388],[4388,4389],[4389,4390],[4390,4391],[4391,4392],[4392,4393],[4393,4394],[4394,4395],[4395,4396],[4396,4397],[4397,4398],[4398,4399],[4399,4400],[4400,4401],[4401,4402],[4402,4403],[4403,4404],[4404,4405],[4405,4406],[4406,4407],[4407,4408],[4408,4409],[4409,4410],[4410,4411],[4411,4412],[4412,4413],[4413,4414],[4414,4415],[4415,4416],[4416,4417],[4417,4418],[4418,4419],[4419,4420],[4420,4421],[4421,4422],[4422,4423],[4423,4424],[4424,4425],[4425,4426],[4426,4427],[4427,4428],[4428,4429],[4429,4430],[4430,4431],[4431,4432],[4432,4433],[4433,4434],[4434,4435],[4435,4436],[4436,4437],[4437,4438],[4438,4439],[4439,4440],[4440,4441],[4441,4442],[4442,4443],[4443,4444],[4444,4445],[4445,4446],[4446,4447],[4447,4448],[4448,4449],[4449,4450],[4450,4451],[4451,4452],[4452,4453],[4453,4454],[4454,4455],[4455,4456],[4456,4457],[4457,4458],[4458,4459],[4459,4460],[4460,4461],[4461,4462],[4462,4463],[4463,4464],[4464,4465],[4465,4466],[4466,4467],[4467,4468],[4468,4469],[4469,4470],[4470,4471],[4471,4472],[4472,4473],[4473,4474],[4474,4475],[4475,4476],[4476,4477],[4477,4478],[4478,4479],[4479,4480],[4480,4481],[4481,4482],[4482,4483],[4483,4484],[4484,4485],[4485,4486],[4486,4487],[4487,4488],[4488,4489],[4489,4490],[4490,4491],[4491,4492],[4492,4493],[4493,4494],[4494,4495],[4495,4496],[4496,4497],[4497,4498],[4498,4499],[4499,4500],[4500,4501],[4501,4502],[4502,4503],[4503,4504],[4504,4505],[4505,4506],[4506,4507],[4507,4508],[4508,4509],[4509,4510],[4510,4511],[4511,4512],[4512,4513],[4513,4514],[4514,4515],[4515,4516],[4516,4517],[4517,4518],[4518,4519],[4519,4520],[4520,4521],[4521,4522],[4522,4523],[4523,4524],[4524,4525],[4525,4526],[4526,4527],[4527,4528],[4528,4529],[4529,4530],[4530,4531],[4531,4532],[4532,4533],[4533,4534],[4534,4535],[4535,4536],[4536,4537],[4537,4538],[4538,4539],[4539,4540],[4540,4541],[4541,4542],[4542,4543],[4543,4544],[4544,4545],[4545,4546],[4546,4547],[4547,4548],[4548,4549],[4549,4550],[4550,4551],[4551,4552],[4552,4553],[4553,4554],[4554,4555],[4555,4556],[4556,4557],[4557,4558],[4558,4559],[4559,4560],[4560,4561],[4561,4562],[4562,4563],[4563,4564],[4564,4565],[4565,4566],[4566,4567],[4567,4568],[4568,4569],[4569,4570],[4570,4571],[4571,4572],[4572,4573],[4573,4574],[4574,4575],[4575,4576],[4576,4577],[4577,4578],[4578,4579],[4579,4580],[4580,4581],[4581,4582],[4582,4583],[4583,4584],[4584,4585],[4585,4586],[4586,4587],[4587,4588],[4588,4589],[4589,4590],[4590,4591],[4591,4592],[4592,4593],[4593,4594],[4594,4595],[4595,4596],[4596,4597],[4597,4598],[4598,4599],[4599,4600],[4600,4601],[4601,4602],[4602,4603],[4603,4604],[4604,4605],[4605,4606],[4606,4607],[4607,4608],[4608,4609],[4609,4610],[4610,4611],[4611,4612],[4612,4613],[4613,4614],[4614,4615],[4615,4616],[4616,4617],[4617,4618],[4618,4619],[4619,4620],[4620,4621],[4621,4622],[4622,4623],[4623,4624],[4624,4625],[4625,4626],[4626,4627],[4627,4628],[4628,4629],[4629,4630],[4630,4631],[4631,4632],[4632,4633],[4633,4634],[4634,4635],[4635,4636],[4636,4637],[4637,4638],[4638,4639],[4639,4640],[4640,4641],[4641,4642],[4642,4643],[4643,4644],[4644,4645],[4645,4646],[4646,4647],[4647,4648],[4648,4649],[4649,4650],[4650,4651],[4651,4652],[4652,4653],[4653,4654],[4654,4655],[4655,4656],[4656,4657],[4657,4658],[4658,4659],[4659,4660],[4660,4661],[4661,4662],[4662,4663],[4663,4664],[4664,4665],[4665,4666],[4666,4667],[4667,4668],[4668,4669],[4669,4670],[4670,4671],[4671,4672],[4672,4673],[4673,4674],[4674,4675],[4675,4676],[4676,4677],[4677,4678],[4678,4679],[4679,4680],[4680,4681],[4681,4682],[4682,4683],[4683,4684],[4684,4685],[4685,4686],[4686,4687],[4687,4688],[4688,4689],[4689,4690],[4690,4691],[4691,4692],[4692,4693],[4693,4694],[4694,4695],[4695,4696],[4696,4697],[4697,4698],[4698,4699],[4699,4700],[4700,4701],[4701,4702],[4702,4703],[4703,4704],[4704,4705],[4705,4706],[4706,4707],[4707,4708],[4708,4709],[4709,4710],[4710,4711],[4711,4712],[4712,4713],[4713,4714],[4714,4715],[4715,4716],[4716,4717],[4717,4718],[4718,4719],[4719,4720],[4720,4721],[4721,4722],[4722,4723],[4723,4724],[4724,4725],[4725,4726],[4726,4727],[4727,4728],[4728,4729],[4729,4730],[4730,4731],[4731,4732],[4732,4733],[4733,4734],[4734,4735],[4735,4736],[4736,4737],[4737,4738],[4738,4739],[4739,4740],[4740,4741],[4741,4742],[4742,4743],[4743,4744],[4744,4745],[4745,4746],[4746,4747],[4747,4748],[4748,4749],[4749,4750],[4750,4751],[4751,4752],[4752,4753],[4753,4754],[4754,4755],[4755,4756],[4756,4757],[4757,4758],[4758,4759],[4759,4760],[4760,4761],[4761,4762],[4762,4763],[4763,4764],[4764,4765],[4765,4766],[4766,4767],[4767,4768],[4768,4769],[4769,4770],[4770,4771],[4771,4772],[4772,4773],[4773,4774],[4774,4775],[4775,4776],[4776,4777],[4777,4778],[4778,4779],[4779,4780],[4780,4781],[4781,4782],[4782,4783],[4783,4784],[4784,4785],[4785,4786],[4786,4787],[4787,4788],[4788,4789],[4789,4790],[4790,4791],[4791,4792],[4792,4793],[4793,4794],[4794,4795],[4795,4796],[4796,4797],[4797,4798],[4798,4799],[4799,4800],[4800,4801],[4801,4802],[4802,4803],[4803,4804],[4804,4805],[4805,4806],[4806,4807],[4807,4808],[4808,4809],[4809,4810],[4810,4811],[4811,4812],[4812,4813],[4813,4814],[4814,4815],[4815,4816],[4816,4817],[4817,4818],[4818,4819],[4819,4820],[4820,4821],[4821,4822],[4822,4823],[4823,4824],[4824,4825],[4825,4826],[4826,4827],[4827,4828],[4828,4829],[4829,4830],[4830,4831],[4831,4832],[4832,4833],[4833,4834],[4834,4835],[4835,4836],[4836,4837],[4837,4838],[4838,4839],[4839,4840],[4840,4841],[4841,4842],[4842,4843],[4843,4844],[4844,4845],[4845,4846],[4846,4847],[4847,4848],[4848,4849],[4849,4850],[4850,4851],[4851,4852],[4852,4853],[4853,4854],[4854,4855],[4855,4856],[4856,4857],[4857,4858],[4858,4859],[4859,4860],[4860,4861],[4861,4862],[4862,4863],[4863,4864],[4864,4865],[4865,4866],[4866,4867],[4867,4868],[4868,4869],[4869,4870],[4870,4871],[4871,4872],[4872,4873],[4873,4874],[4874,4875],[4875,4876],[4876,4877],[4877,4878],[4878,4879],[4879,4880],[4880,4881],[4881,4882],[4882,4883],[4883,4884],[4884,4885],[4885,4886],[4886,4887],[4887,4888],[4888,4889],[4889,4890],[4890,4891],[4891,4892],[4892,4893],[4893,4894],[4894,4895],[4895,4896],[4896,4897],[4897,4898],[4898,4899],[4899,4900],[4900,4901],[4901,4902],[4902,4903],[4903,4904],[4904,4905],[4905,4906],[4906,4907],[4907,4908],[4908,4909],[4909,4910],[4910,4911],[4911,4912],[4912,4913],[4913,4914],[4914,4915],[4915,4916],[4916,4917],[4917,4918],[4918,4919],[4919,4920],[4920,4921],[4921,4922],[4922,4923],[4923,4924],[4924,4925],[4925,4926],[4926,4927],[4927,4928],[4928,4929],[4929,4930],[4930,4931],[4931,4932],[4932,4933],[4933,4934],[4934,4935],[4935,4936],[4936,4937],[4937,4938],[4938,4939],[4939,4940],[4940,4941],[4941,4942],[4942,4943],[4943,4944],[4944,4945],[4945,4946],[4946,4947],[4947,4948],[4948,4949],[4949,4950],[4950,4951],[4951,4952],[4952,4953],[4953,4954],[4954,4955],[4955,4956],[4956,4957],[4957,4958],[4958,4959],[4959,4960],[4960,4961],[4961,4962],[4962,4963],[4963,4964],[4964,4965],[4965,4966],[4966,4967],[4967,4968],[4968,4969],[4969,4970],[4970,4971],[4971,4972],[4972,4973],[4973,4974],[4974,4975],[4975,4976],[4976,4977],[4977,4978],[4978,4979],[4979,4980],[4980,4981],[4981,4982],[4982,4983],[4983,4984],[4984,4985],[4985,4986],[4986,4987],[4987,4988],[4988,4989],[4989,4990],[4990,4991],[4991,4992],[4992,4993],[4993,4994],[4994,4995],[4995,4996],[4996,4997],[4997,4998],[4998,4999]]]
    # for i in range(5):
    # res = sol.findNumberOfLIS([3,1,2])
    # res = sol.findNumberOfLIS([2,2,2,2,2])
    # res = sol.findNumberOfLIS([1,2,4,3,5,4,7,2])
    # res = sol.canPartitionKSubsets([605,454,322,218,8,19,651,2220,175,710,2666,350,252,2264,327,1843],4)
    # res = sol.canFinish(3,[[0,1],[1,0]])
    # res = sol.canFinish(3,[[1,0],[0,2],[2,1]])
    # res = sol.canFinish(8,[[1,0],[2,6],[1,7],[6,4],[7,0],[0,5]])
    # res = sol.findMinHeightTrees(3,[[0,1],[0,2]])
    # res = sol.findMinHeightTrees(*tdata)
    # res = sol.pacificAtlantic([[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]])
    assert sol.searchRange([5, 7, 7, 8, 8, 10],8) == [3,4]
    assert sol.searchRange([5, 17, 100, 111],3) == [-1,-1]
    assert sol.searchMatrix2([[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]],5) == True
    assert sol.searchMatrix2([[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]],20) == False
    assert sol.searchRange([],0) == [-1,-1]
    assert sol.searchRange([5,7,7,8,8,10],8) == [3,4]
    assert sol.searchRange([5,7,7,8,8,10],6) == [-1,-1]
    # res = sol.searchMatrix([[1,3,5,7],[10,11,16,20],[23,30,34,50]],3)
    # res = sol.search([1,2,1],2)
    # tdata = ["iibazqymhssjxnaurfsddydbwftetatwyosdcnnujzjzbpdeozpwpaixxghuyrlwrxdsyvpgoycuumxtryrqpgfmgyodbvtmybzkaqoobecgosekwvcyyfzuaiweqjnwfssjmbrcfkppdclmqpayziyzwhktkxlmhjgjjrkdvfseatybsltvlsklnxxywzcgvvqxrnkkccfvbwpjfnpqrkfqgoajiksmtmogouhivogjuriyfwushmnyqcrdkkfkrhfieaujweckpjjtzlitjhghukqllqttszuzrnwyafumimkiljsatlklrwwdbxcaunklenvcbgq","quitwlnicl"]
    # tdata = ["cnhnczmccqouqadqtnmjjzl","nm"]
    # tdata = ["tizglvnpwadrarimeufxexveuxwrvaatjjkssklxnentjuyumobfuyixktaqztiodzoliiwppdqvplwrnrbapiddjptzlkzysmdijvisrxpsuxvmgzovkcekddgzhxwcisxapnwwszhlageihuxgewuicxiafrrceswnqiuujzruaegsfxwapmkxchfvmrhuaqnumtjilygfhdwxbevciazptagpdouknowqdpvbqqhebfiorzngqswdfxczzgxfnmacrpwgmhvgolotlfvobosdvwptomodjqemdjbexcqjzbslwrjqrecfgltkefoizfgyrhoruvaovolifrwvsgcuaftjjurufxydvhmsuttiuylsfwfbhfynlmjvrpsrvfudfqnlbhhsfsjocyoadxhfobudlgygaykvxcmlhvonkxvoycrukbucdplorwhpvazbmqhhiorpdmzvzavomruxemiwuurdmmkqztwiyfytjvzltnriaypqckiecgkyosyhdqhpiicfwyncmcocatdtjdrrqomlidoglzewffvcbggcbptawnovoziqnedvlbkrravyzpthenvikrrmriwovruucbdymufjveymajnvheqfrcnffdfnomyfjydsbousktbtlwpbxcfznxarnubwbaowmahkscrtifvrntjjyhpevxxsqcbpqlvgcpcxdskgklcsxjvoqybjurprlnhvikumrbdfdpqxijqsdbkjsykjekrehcawsqetimvyewwxfcchvjmwsenvnpxqtyeilwdmskarrpaomhhseguozehmrmiybifmfppjayngcniyiagbmtrajgxdewahvintzwjskweoardirqnljiaolcwzhuadtcpjkxhdgladruonehzjhvqdfeoqmwmpokbiwrubldtklomsxdsjjcnynqpcqblnsnexzvbpovooxwdgftnaxbszrlxpmnjrfmgmogeichqpgvrayawcnqruduneqksbnoasjcwoistelhuublbcmqgjliawqkkzgvocdpecitonnpprxtpssoectcycvlzhqzliuhnvxzgwrywhbikkjfpgpnljqrfdtuxmocsezzpgjudgmmwxupivfypbrxakoxropyioqhtqfqiiosjhvooddnclwvmjftelcnfzxhtcefnzsusvttdlkciinxdaibfvubmnigthlpbapgixdltnoiffnqiystaodlwiqvldxrgucibsusfasswagevqitlosqxtertfoidbffvoeptcwlaoeujigjurxkatfeouduqsqjenbtjtsoydgfqyipmcqokdlfowuexrlexjudlisndlvkwbulhevojwcjccozjjtzwjlpqoxecivazokcxgoxmchrisvtptsfdcroogjaexismjqwjrykytuthrvzeiiitxctkbcdcxsylkshrhfdzsxylghicenofniexzfmfmmntkynsnrjjzozlzizanemvpbzjemxpigviqbqhvhinyjpowboyowwyxdwutuuheqvzhvwvbxyedpssqlpyjhznemclizqnmshtktjwzxdnagtfpwxfesubqtpyllbwqgzeyslpdfvtrddwvwjhsbjzvtwcjcyxmissxxjihcscyndramqucqylktufczepwkylruyjgrodpsldhxtmleogheudvoivxowmsuphghmulfzyudxrnfcdafpgmsmzjypuitpjyxfhkbnyzgnuylfenwiyjrarzmkmdtiwpeeqfwpnjhspzrzvgbsvmdpnclunmoqwfkbwpmolqvdmqkahclvhxtozotpsqbnjjtqzlwbvomqbprmzlsxjoxjcamubqwwfccmjlevpcgkjzrgswiibedvsffqecnzvcjqfcxeqyfuibpawvomjipphmectdwsvhlmghrxttwwuxfmdcdawaesdygngkyblfryduloaocpnxzlssguujmqblufiunozgjimsxuulaqrdrmdcmmgrrqineccootgzojbxguajvwlcmmlivyzcvwdesyuacaffsotclwsycwrvlssritecbicfmdgzdiicmkcbopvqjugrihbpkxliakuggncfojumzplpiwjfpzlkrgtupykimfmtdkyzpiywfvpubgwpzgsijdrhnapuzppqwmjkdncmsrzidnrnqddohmbrwojqatrhsuopryisnmjyiwnbabmlqubywqaiowtaimpjkgsyqsamnouglustpcteeciiacebwinelocxhjgrlgtptfzaxanidwioqsxeyzsgmiiizstcbybxzpkzdirhlgqbnxwuyrfoilkluihrwwjcggxduxgttroczucrpjnudjtjamceprrehibvttnybckjdyxejpjwukgubdknjqzigxfvxrgacthcryzdwkmiquznifqkbhuhqbfvddxgmuybrdehwcitwpummdauylabxnixgzotbxxwdzhivmisqjazecqhtpleshqjmgyhkihikettlsndjauqwhyeeukojwqiggprcpzicbvgyhasrqcxbdrakmdchtqjisiznuwkhzxyzyqzsypuzqopyznemqyrrvrkytzelqofpebvszvfxizwveboxtanhbkiujnpakyjxoixfyvtpdtwjqredevgysvxtpmjublzksgfbfaiqpjcuwcytatztkwvkvwukuggqlxtqdtzkegsdjoubufcxmnphdudphlprcizhudedgccqxexjvgttjyffkqxdsdgimwcotlteijibxdvdeaxcnwojrfyaiiwwtltzbgexavgadabnhekmmvxyvchmzgyccxzagypsvkypveetypcjmafkhdyvriicprptukcwaxnaosrprvtkqlpchtvedkcakixxqwxdshiiqvatrhjxaurwtpyedpwuunneyylgsbyhefxibsjvjtipjsjfkqxhzuxvyugkgopwprumuheoxzbgvfrftlprwmrjfdbcpidnowgjslapskibjrjuethuwydvrzcnnzbppvkqgikdknwpumfaluccgmltmjgfjxvstjaobjxvvploxhrywhbycamasoyfpnfcakhjgujnywyjhazjyokcwcrkeoxechuailpviskfjjrnwubndayhleptuyfowrlyrneafuqiylmfsmvzvmbyobymbzyemqnbxarnrlbqsokclvliaznnsnvxsoqyvqdcdgurknpfuqalxtenehrjwmiffeiyxwxbumrvobmoecozroncffyzopecsvwlngcjbhrhfpqlbvydevpplfflnehnvkrzdjxffzcqbwdpodgyurzhuuqgfwtrsigricbfkxpsjhlcuxahpvmaimabcdjopyxphykduphqbzhuvwibgbfsqfnwnoqwwyshdthcsmzzqexowrbpuqozjfypqjbwtqdahfcwbtcasnsyfrlseqsdjmuxgcgahvfyudkchptzedkfvvsocjcomumfauxorcsyutcgjqtgyvrkteixgovpsisgmpwffizcpzkfhhmpkwmsrmjkqinarobkqctsmqxppefadlgqbtainshlsdstsxkjubzgfcccqmqtqhqwryjwktlhzbhdqvpbzyiabvkxhyntrswoaxzqilrppeyzjngmwoxnugrlfrftiysgukxumnlwsucaqecpyuzyzghnernvrciacwchqqqdeigglktpikizxvyrzcpvfrehjdmagwwpztfpmmqgoxdwmwonixjeomsmuhfbhckkedqilidmlqtplruwdsgpflydzwrlrqjlprpqouvrdzzicfvevaiamhjutpaocknpuxotcvwhbcfzthbbjwdbrhfivmqyyevfzkzamdkmsawxbyjngttwfoldnazzqjabywvxjpuyvtylwjbpxwbqhybtfiwpolgeeelbtpafwudfwouudrbbfbbssaqcbtfarqvwicijorgfbzjykfdbdxatpumbnbrnvudxhwevomvipapbtuhbovxfwfhjlcjglydissiawiqkqgbunuhdpevqccrrqycsywratpnuuccdrznqzfwceotztuzzdyhtosrqudtkhscjtfvskilocrgwavaflxfgbevpfuhqttmetognhgkswflvbwgcraorzmamktsytqxtnegtywvslyvjxipfuheuuduxkunppkmlbnjmtcuaqquxofmjlrsmlzuznzgrqdovzxbrvkaxlfpyuxsvwuohjewkbxntbqbxlxkvdxthvihhvmyuogxhqtxddjvcufrddstecbtmmeunkurguegiygqjsdclrsypeobwdutywohybvwbunbdnizddvtuxhoasubnampccqpukhnphzlczrhzqfqgcgqaavevacjqfdjywiocsvgtzugpmnbodgwpyydkifmbfnvhogyzcidkfzfsppjrmuxqverlwqvetgfifhuwvzizthgkfyigszxgbmsixsuyizhjfbrvbshebdrlunvyniezhrwgywjurjyrqraeoucirgtbukwysfdcdrwxjrqintjdheebylmottpfrjoadhcqxrnlmlaxoqkkylamvplmyydvcehmpcvrysfreytulqfhsrjvvpgsvnwfoisbxvtigvoxezkxxhnecenqopsqkklfalglhleescnnldxnrzrmxsqgmamhnsbmdzgoppcmgyvsrpxcfeihpszxnksznhxlkbmewdydxppbimofrebpflgptpddchiquovfbfwwslmpitjticvrtlibgvdtqxpmwfahxdhzstijznzqgyblfvnftejchqtqtvfnshttbldrwyfcxowwxvzqeellrcwgtynoypwrrtzenyfdumxzqfrqzoiyrxpakakbeykspipzaunxapexnsplojrbhpujgavntapfhdeomndnnvpxriurqhmautsfswyqoozizevbrkxubbwrsbycmjhgrxveecwveszbqgkbcdquoqwutsszmqbppdywvfqlavbnxcyowximjqnntdgyatyfscnufmmabpsozltvivkghgrsqihnivdgiluhxkcoerrwdocxaqghsarormtqvhwvsyhbbdwemoddqfxcjozkfqkxtodtoqbwrwszlgrfokjsoqiyczxadnsnhvcobrpnhfxfqpqctgahiniwjmqkxsimwxcjdcvmuslhrevfixovocupeqfvgoyhvoyzgqrgfuqkegrodshqgbpcdscieajehzvtwolzlaibkfqmgyncaxngcgoaiezzcfegkdhdqiswldvcnaupdqopsiesaietwyroyeqaivumcpztedvogffzioygeymoydcvshwptushayrebmdxxkpejxcgueucnxfilfuktrwkabgqllexcgczwwhnvafvjcnzbxcumtbqxjekodxrqrnezpuicmyznvinqcalcbjszmhkclpzvsabfyshxyjhdxeivfbyhdyhuuecmfexyypqrrdearknqcltlljadmdzditbmehrckxpaznivyjlgyihnbxzhvluiwerudguipbeshncltvhvgkijhrbohhxmzawsdjpguoeskzzbytijoioadoirjhrrlyzpucsxkrmjdsfbvgkhdlajeooyxxnkruwpuemzbghdwprajxrijcvuzzymdltvaqwmcxtecumgycvbrasqfhiimezgkzndjskveefkalffxzaybeevtdkddqxhpsemwgihjwpdhvenrghigpphyrnopqshufrmyfixecazukulkegkmuwhsfazkeoygqnsdmwovroiztmndpovbovohqoqurhplqlhrdxqgajgyapcsumifyfmlwpgarlpnpdbtofafvwcohnijhklppmarbcymsukiiluvgyhusaudwnuzbdsjjslnmgnxhwnolnhhbwvkonhmhnvlrzazlkcmvqappmzcnrmyypsvxdslxrgsrcqwsuawklfnywkraencxgnqnxpvoawhdmsxakxcuzedssmgdyrvimbiaipgsujmwufabecmjpyglqtjbliugbripyiuvbraoyjktdbtzytyzqpthhfrzbghnjcthzxltzynbxtqlksscfqbimmnhjilkcrvzneeuykmqibiofkigkwpimpawcstawyzwfxsvcapmyzoeqzuwnkdzvacblyfseykgwncmlanmyuqakupbjndoxmbmoihwzilszttiujdvzqitukvzkobceovszzscwyiccngnbpbiksvfqkofqsjpuqpfmwqlozftqproekcyfdnriphlgdihzavoowhcpuavnzrvsphacvfduwhlqpfldewmlabpdksvujhajwbpwifrxxrolmwcxyzygoadrzbdsyytfbmcbiitwspshqqrfqgasegsqclnnyymxqbqetrwvxfceilcgxuodkbwznstnmqtogtcrmmxcqlwcnmdojdyfyeobgpzbkeansuxhlivvgbpvhnxprnohonlybqcvwcjouygalvkxlyqizvyxzralmqzimkieazcncezqhnkctnadbeiagmsxudignwygycjbrkltnqzcskvhasesulpvfnfcosvwbpoyhkkimnggxxqvnodklnzgeotxlihbjdoctkkbwrqsrcatkeeqpfabbdwdijltlzmurzsomheisvadnnbaczwordozqejwcwgvtbmhwregiztqfexhvbwuadcdpdltvuxkltldpefjcksabxxjapznubaxtvdrfpnlurynzkbdipqinxuvdiyidmesdhnptghiodcmwlhwololxpusaahxuimjmwjrrwfwgoonkvdotigheuqzbuphsdlnukoolwhvoqkwcagucwwipyqaztvtzvssxhmydouujlihaljtenypjbzlaaufcpxmgxxafosdtvslusumyktxaxebmwnyohjosbkiytnctmoyysbsrhzfkbqjaoopqadpxmafoujerjatnedjjzohhgvbsmmhtqogsdidiihmzluphxntpkrzbuybpjjgmvzcuxqyvmwmsnwdqbrsffgvzqlzkrikhuejebofvpmgejefjgqidtacinwvpndvgwunsnoycbqbevwudfsdreybibtvnitcivyiclhmsxqcifecaloqzjvziegtwlyerreuokxkjildifawjzbtasqkrhokkbdzkkxirkytfgzbqoroavyedewpzaovuymebduiwtryhwvxwrfbkpvuhkiedjsdcjevremfxbmkxffbxoyyossgtbgdrykugjzzpvolwbxvyleluyeavjtxygiarxxpbbtffepiwzsgohalrhhprjwuralfxntmdcyggppdmfpvnwiofyapzqxuomoxebtauxqmllghrslymdkbskwatpfusyjjpeohseqizuzdnsubrjosztgujiojvpsnvxgyubancdevwrljesodfuxftppycvdwotkqshcofkwmhnmamabsdcuxgbaqgerkhuvrdllmlvkkhgffknpnlhdbfgtvkfumjyfcapgbeadybguwxgcuhwfvirybsyueqliskvekfapcumnkkkqvncjzffhjvlcgfhfvdjnmwvogxhqqaxbjfwkmbjhbdnwsdeecuhoybilbqzbnvfdnzlxlvsfutnqbddlubhlmasmukqgbypototabfenmbnrtyjdsfbghoqqicxxxlpdmmspzrykskisvucprelqrnhwrbclzbdbkqqdrdlfjqspalqvowqrspwyrqihcvrviltmntmqozghnhgosshzyademfkqoqwtetkmyenlkwykubumuenvsqplgdneexwxnppobrdsflfpyuzgfsjhtukbkxltfciqwvfwhbxwnvufcxridopcnzlbqjnxlvszahhirucyhuhybaubgnpqlmdlgmxnxcnjvpmhudaneucqlistxxxxvnosgoqqvbitrfublihhugrpukizejnifrnurntvhpcwpkeggljewllrmzlorrspfrthaldauaawqglnvxuiinhfewajlugovwofuzjmgaacyagjxaahtcyhpxwtilahnxbqtfrhyjznehmxbtxnfxlpjfwsuhiedpulcwqnnlljltnlbmfybvlrozeunlooxmwuwfyqlxprelqdxsjtndkfkwzhjpctscxcafnqqfalgttxzmzxsfavgsdxxjpemvexifgisdagskwytgrhxxsrafnkqcqflzyzmaftoakhsnqtiheodrowqyovnxgynzkhrggstbfchfkswsnoazcywxcrvmzcidwecacekyezamhnglejmfkelrjmvgcmuqslpibqicayeprjlaxrjjceagstjddmsmgacpxcvgujpqxxetuvuhwrdnysjkeuiumwhofryrfpeqzuiwdolahskxjwykmoeskookbswrovgdvtlgajhzfythkckamyavudnicgzmausgjpctseaybvvzhzvgyigodnlgwnrnfkoqsljtzlbftqmtyrlsztmvwvewudzefjeprkoxfhenhwzxndgbwcjsjabukfvirebfqhwmdyzblezzvwckqmeyprihxqexfmqlifrsudcxhwqpdcyztepdzxtcshjhuokcweiisydappcrqepvfagmodcsmtaqhygnudjzxbkpsybtajdgmyevlbqmvmggpzppeljmosulscfbodcfcqcezmffctengycejcdzbyikculifpepsajkwnhurckexhtvhjlwhlefwcvqjvgcxihvrbqhnjmfmvnoqellzrrsdeqovykvvgadmdnxkhmmhygfdnvcfsdapdchyaqnassqygohdiefihjgdtddkqdefsiiometxswovxapdemlfugnwblqvecgfjujhnpuluexkzgcfaduwilkcjconfigswbepmfcyfznjgoqsyuqaixucbzowkavtxksraawgvwexilodqumcjbpmzfxxdnlthinmsjclumkswytwkbqhigciswalzechgiboxtudylshylpjzqgctpnmpbpuvupweofqejhvcgzikwtczgidodecssoxawzdbknhchazdiodqrcjvaohipqrcmkpezamextlpbiqijrviwcitdmmrlnzogykwtgsxuhlrkfwutobjpvgrglujorevntvuqrpijnqpwfnfczptpruuepeymlirkzubmrrwmedxqeicffjinslbozvoekfoyamuqtexlsfdkxtbmidxpafdvfwxvkfecbllqqtlaxxjeiincuwoonxtypzhvgzouqpnkwqihmpaxivlgfccvdyahljjpvxygcjbvgsnclojkmurymjvfhxgbagdrybiclxdutdfelwcalfhnkxhhoiryecqhsgztoizxoukjxcqekhvoudjbcsduwgtzqonoopkpvasdcaqhitwrkwxqqaurzwogbwfqjkzmesonmqhzdugxylsqytzwhxmxybziiogxktrzbgelwnepctthhnzowkzpvhcdvrfwyffmhsqqzajakkduqvtzkgessupocvtrugvbffcqgkogyosrhfepphaittyhnszcypinmkehomdukcvinjjvuooweuyswawoeqpdzghqbvzyhtwzdgwsqqkscoskavzmsjftiwrsyuibfolpuvookzcspgvitycjtzklltfgedrcgirolinjjmzilezshjioybnevwkvgihqiwlcdgpowqhejttordzvfvemiivqwgcfqimcjeuihodxwycarbqjoanpxyciyrioslpkuhqiewjqipgfxngumnrykvjulsrmwwspbznqutnnghzvaxuxwcxnirzakypnkdkrsyqhgwsjwqntbptrlnhunpgskwf","tgpuesgnuxwgzzlomkgbswrwvpscidszzpsuhwjpuylgxrujoqhdomvdwitxkyshnemycdhvecygxcesyvnmqucokatzsyuahhic"]
    # res = sol.minWindow(*tdata)
    # res = sol.isPalindromeK("eeccccbebaeeabebccceea")
    # res = sol.isPalindromeK("abca")
    # tdata = [0,1,0,3,12]
    # res = sol.moveZeroes(tdata)
    # print(tdata)
    # res = sol.isMonotonic([6,5,4,4])
    # res = sol.isPalindromeK("deeee")
    # res = sol.isPalindromeK("aguokepatgbnvfqmgmlcupuufxoohdfpgjdmysgvhmvffcnqxjjxqncffvmhvgsymdjgpfdhooxfuupuculmgmqfvnbgtapekouga")
    # res = sol.isPalindrome("A man, a plan, a canal: Panama")
    # res = sol.isPalindrome("Marge, let's \"[went].\" I await {news} telegram.")
    # res = sol.firstUniqChar("leetcode")
    assert sol.findMin([4,5,6,7,0,1,]) == 0
    assert sol.findMin([2,2,2,0,1]) == 0
    assert sol.findMin([3,3,1,3]) == 1
    # res = sol.findPeakElement([1,2])
    # res = sol.longestIncreasingPath([[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]])
    # res = sol.peakIndexInMountainArray([18,29,38,59,98,100,99,98,90])
    # res = sol.merge([[1,3],[2,6],[8,10],[15,18]])
    # res = sol.eraseOverlapIntervals([[1,100],[11,22],[1,11],[2,12]])
    # res = sol.findMinHeightTrees(3,[[0,1],[0,2]])
    # res = sol.findMinHeightTrees(4,[[1,0],[1,2],[1,3]])
    # res = sol.findMinHeightTrees(11,[[0,1],[0,2],[2,3],[0,4],[2,5],[5,6],[3,7],[6,8],[8,9],[9,10]])
    # res = sol.findMinHeightTrees(6,[[3,0],[3,1],[3,2],[3,4],[5,4]])
    # res = sol.findMinHeightTrees(4,[[1,0],[1,2],[1,3]])
    # res = sol.canFinish(*tdata)
    # res = sol.canFinish(7,[[1,0],[0,3],[0,2],[3,2],[2,5],[4,5],[5,6],[2,4]])
    # res = sol.canFinish(3,[[2,1],[1,0]])
    # res = sol.canFinish(3,[[0,1],[0,2],[1,2]])
    # res = sol.canPartitionKSubsets([4,3,2,3,5,2,1],4)
    # res = sol.canPartitionKSubsets([2,2,2,2,3,4,5],4)
    # res = sol.findNumberOfLIS([0,1,0,3,2,3])
    assert sol.lengthOfLIS([4,10,4,3,8,9]) == 3
    #   res = sol.lengthOfLIS([10,9,2,5,3,7,101,18])
    #   res = sol.lengthOfLIS([7,7,7,7,7,7,7])
    # res = sol.coinChange(*([1,2,5],11))
    res = sol.maxEnvelopes([[2,100],[3,200],[4,300],[5,500],[5,400],[5,250],[6,370],[6,360],[7,380]])
    # res = sol.coinChangeDP(*([1, 2, 3], 5))
    # res = sol.coinChange(*([186,419,83,408],6249))
    res = sol.coinChangeBFS(*([186,419,83,408],6249))
    # print([[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]])
    # expected = "qoobecgosekwvcyyfzuaiweqjnwfssjmbrcfkppdclmqpayziyzwhktkxlmhjgjjrkdvfseatybsltvlsklnxxywzcgvvqxrnkkccfvbwpjfnpqrkfqgoajiksmtmogouhivogjuriyfwushmnyqcrdkkfkrhfieaujweckpjjtzlitjhghukqllqttszuzrnwyafumimkiljsatlklrwwdbxcaunkl"
    # expected = "tpikizxvyrzcpvfrehjdmagwwpztfpmmqgoxdwmwonixjeomsmuhfbhckkedqilidmlqtplruwdsgpflydzwrlrqjlprpqouvrdzzicfvevaiamhjutpaocknpuxotcvwhbcfzthbbjwdbrhfivmqyyevfzkzamdkmsawxbyjngttwfoldnazzqjabywvxjpuyvtylwjbpxwbqhybtfiwpolgeeelbtpafwudfwouudrbbfbbssaqcbtfarqvwicijorgfbzjykfdbdxatpumbnbrnvudxhwevomvipapbtuhbovxfwfhjlcjglydissiawiqkqgbunuhdpevqccrrqycsywratpnuuccdrznqzfwceotztuzzdyhtosrqudtkhscjtfvskilocrgwavaflxfgbevpfuhqttmetognhgkswflvbwgcraorzmamktsytqxtnegtywvslyvjxipfuheuuduxkunppkmlbnjmtcuaqquxofmjlrsmlzuznzgrqdovzxbrvkaxlfpyuxsvwuohjewkbxntbqbxlxkvdxthvihhvmyuogxhqtxddjvcufrddstecbtmmeunkurguegiygqjsdclrsypeobwdutywohybvwbunbdnizddvtuxhoasubnampccqpukhnphzlczrhzqfqgcgqaavevacjqfdjywiocsvgtzugpmnbodgwpyydkifmbfnvhogyzcidkfzfsppjrmuxqverlwqvetgfifhuwvzizthgkfyigszxgbmsixsuyizhjfbrvbshebdrlunvyniezhrwgywjurjyrqraeoucirgtbukwysfdcdrwxjrqintjdheebylmottpfrjoadhcqxrnlmlaxoqkkylamvplmyydvcehmpcvrysfreytulqfhsrjvvpgsvnwfoisbxvtigvoxezkxxhnecenqopsqkklfalglhleescnnldxnrzrmxsqgmamhnsbmdzgoppcmgyvsrpxcfeihpszxnksznhxlk"
    # print(expected)
    # print(res)
    # td = "a"*1000
    # res = sol.countSubstrings("aba")
    # print(res)
    # res = sol.countBits(5)
    # res = sol.isSubtree(sol.deserialize("3,4,5,1,2,#,#,0"), sol.deserialize("4,1,2"))
    # res = sol.canPartition([1,2,5])
    # res = sol.canPartition([100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,99,97])
    # print(res)
        # check(set((tuple(sorted(l)) for l in expected[i])), set((tuple(sorted(l)) for l in res)))
    # break
    # res = sol.canPartition([1,2,5])
    # res = sol.canPartition([100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,99,97])
    # print(res)
        # check(set((tuple(sorted(l)) for l in expected[i])), set((tuple(sorted(l)) for l in res)))
    # break
    td = [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]
    assert sol.isValidSudoku(td) == True
    # sol.solveSudoku(td)
    # print(td)
    # print(sol.solveNQueens(4))
    # break
  # commands = [(sol.push, 4),(sol.push, 0),(sol.push, 9),(sol.push, 3),(sol.push, 4),(sol.push, 2),(sol.pop,),(sol.push, 6),(sol.pop,),(sol.push, 1),(sol.pop,),(sol.push, 1),(sol.pop,),(sol.push, 4),(sol.pop,),(sol.pop,),(sol.pop,),(sol.pop,),(sol.pop,),(sol.pop,)]
  # for cmd in commands:
  #     if cmd[0] == sol.push:
  #         cmd[0](cmd[1])
  #     else:
  #         print(cmd[0]())
    a = "6,2,8,0,4,7,9,#,3,3,5"
    deser = sol.deserialize(a)
    assert sol.lowestCommonAncestorBST(deser, deser.left, deser.right).val == 6
    assert sol.lowestCommonAncestorBST(deser, deser.left, deser.left.right).val == 2
    a = "2,1"
    deser = sol.deserialize(a)
    assert sol.lowestCommonAncestorBST(deser, deser, deser.left).val == 2
    a = "15,20,34,35,5,14,16,26,#,25,23,#,30,3,36,#,#,7,24,11,32,#,#,21,#,#,#,29,4,9,#,33,13,#,#,#,#,22,31,#,27,19,1,#,12,18,6,#,#,#,2,#,#,#,#,10,#,#,#,#,8,#,28,#,#,#,#,#,17"
    deser = sol.deserialize(a)
    assert sol.lca(deser, 33, 5) == 5
    a = "16,23,9,#,1,22,2,25,19,6,13,#,24,14,#,30,4,26,29,#,#,#,#,#,3,#,8,#,#,12,18,28,#,10,#,5,#,17,11,21,7,#,#,#,20,#,#,#,#,#,15,#,#,#,#,#,#,#,27,#,#"
    deser = sol.deserialize(a)
    assert sol.lca(deser, 32, 24) == -1
    a = "3,5,1,6,2,0,8,#,#,7,4"
    deser = sol.deserialize(a)
    assert sol.lowestCommonAncestorBT(deser, deser.left, deser.right).val == 3
    a = "3,5,1,6,2,0,8,#,#,7,4"
    deser = sol.deserialize(a)
    assert sol.lowestCommonAncestorBT(deser, deser.left, deser.left.right.right).val == 5
    node9 = TreeNode(9)
    assert sol.lowestCommonAncestorBT(deser, node9, deser.right) == None
    # print(sol.lowestCommonAncestor(deser, deser.left, deser.right).val)
    # print(sol.compress(["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","0", "0", "b","b","c","c","c"]))
    # print(sol.compress(["a","a","b","b","c","c","c"]))
    tdata = [[1,2,3],[4,5,6],[7,8,9]]
    for r in tdata: print(r)
    sol.rotateAC(tdata)
    for r in tdata: print(r)
    tdata = [[1,2,3,4],[5,0,7,8],[0,10,11,12],[13,14,15,0]]
    tdata = [[-4,-2147483648,6,-7,0],[-8,6,-8,-6,0],[2147483647,2,-9,-6,-10]]
    # res = sol.setZeroes(tdata)
    # break
  # a = [3,2,1,6,0,5]
  # bst = BinarySearchTree(a)
  # maxRoot = sol.constructMaximumBinaryTree(a)
  # minRoot = sol.constructMinimumBinaryTree(a)
  # print(inOrder(maxRoot), preOrder(minRoot), sol.MinBinaryTreeSort(a), inOrder(bst.root))
  # a = "1,3,2,5,3,#,9"
  # a = "1,3,2,5"
  # a = "1,1,1,1,#,#,1,1,#,#,1"
  # deser = sol.deserialize(a)
  # print(sol.widthOfBinaryTree(deser))
  # td = "tndsewnllhrtwsvxenkscbivijfqnysamckzoyfnapuotmdexzkkrpmppttficzerdndssuveompqkemtbwbodrhwsfpbmkafpwyedpcowruntvymxtyyejqtajkcjakghtdwmuygecjncxzcxezgecrxonnszmqmecgvqqkdagvaaucewelchsmebikscciegzoiamovdojrmmwgbxeygibxxltemfgpogjkhobmhwquizuwvhfaiavsxhiknysdghcawcrphaykyashchyomklvghkyabxatmrkmrfsppfhgrwywtlxebgzmevefcqquvhvgounldxkdzndwybxhtycmlybhaaqvodntsvfhwcuhvuccwcsxelafyzushjhfyklvghpfvknprfouevsxmcuhiiiewcluehpmzrjzffnrptwbuhnyahrbzqvirvmffbxvrmynfcnupnukayjghpusewdwrbkhvjnveuiionefmnfxao"
  # td = "aab"
  # print(sol.reorganizeString(td))
  # print(sol.findClosestElements([0,0,1,2,3,3,4,7,7,8], 3, 5))
  # print(sol.findClosestElements([1,1,1,10,10,10], 1, 9))
  # bst = "3,1,4,#,2"
  # deser = sol.deserialize(bst)
  # print(sol.kthSmallestRe(deser, 1),sol.kthSmallestIt(deser, 1))
  # bst = "5,3,6,2,4,#,#,1"
  # deser = sol.deserialize(bst)
  # print(sol.kthSmallestRe(deser, 3),sol.kthSmallestIt(deser, 3))
  # deser = sol.deserialize("1")
  # print(sol.serialize(deser),sol.serialize0(deser))
  # deser = sol.deserialize("1,2")
  # print(sol.serialize(deser),sol.serialize0(deser))
  # deser = sol.deserialize("1,-2,-3,1,3,#,#,-2,#,-1")
  # print(deser.val,deser.left.val,deser.right.val)
  # deser0 = sol.deserialize0("1,-2,-3,1,3,#,#,-2,#,-1")
  # print(deser0.val,deser0.left.val,deser0.right.val)
  # ser = sol.serialize(deser)
  # ser0 = sol.serialize0(deser0)
  # print(ser)
  # print(ser0)
  # assert(ser == ser0)
  # s = '4,-7,-3,#,#,-9,-3,9,-7,-4,#,6,#,-6,-6,#,#,0,6,5,#,9,#,#,-1,-4,#,#,#,-2'
  # print(sol.serialize0(sol.deserialize0(s)))
  # 1, -2, -3, 1, 3, None, None, -2, None, -1
  # root_1 = TreeNode(1)
  # root_1.left = TreeNode(-2)
  # root_1.right = TreeNode(-3)
  # root_1.left.left = TreeNode(1)
  # root_1.left.right = TreeNode(3)
  # root_1.left.left.left = TreeNode(-2)
  # root_1.left.right.left = TreeNode(-1)
    root_1 = sol.deserialize('1,-2,-3,1,3,#,#,-2,#,-1')
    expected_1 = 3
    output_1 = sol.maxPathSum(root_1)
  # check(expected_1, output_1)
  # print(sol.serialize0(root_1))

  # root_1 = TreeNode(-10)
  # root_1.left = TreeNode(9)
  # root_1.right = TreeNode(20)
  # root_1.right.left = TreeNode(15)
  # root_1.right.right = TreeNode(7)
    root_1 = sol.deserialize('-10,9,20,#,#,15,7')
    expected_1 = 42
    output_1 = sol.maxPathSum(root_1)
  # check(expected_1, output_1)
  # print(sol.serialize0(root_1))

  # root_1 = TreeNode(5)
  # root_1.left = TreeNode(4)
  # root_1.right = TreeNode(8)
  # root_1.left.left = TreeNode(11)
  # root_1.left.left.left = TreeNode(7)
  # root_1.left.left.right = TreeNode(2)
  # root_1.right.left = TreeNode(13)
  # root_1.right.right = TreeNode(4)
  # root_1.right.right.left = TreeNode(5)
  # root_1.right.right.right = TreeNode(1)
    root_1 = sol.deserialize('5,4,8,11,#,13,4,7,2,#,#,5,1')
    expected_1 = [[5,4,11,2],[5,8,4,5]]
    output_1 = sol.pathSumL(root_1,22)
    check(expected_1, output_1)
    print(sol.serialize(root_1), output_1)
  # check(expected_1, output_1)
  # print(sol.serialize0(root_1))

  # root_1 = TreeNode(10)
  # root_1.left = TreeNode(5)
  # root_1.right = TreeNode(-3)
  # root_1.left.left = TreeNode(3)
  # root_1.left.right = TreeNode(2)
  # root_1.left.left.left = TreeNode(3)
  # root_1.left.left.right = TreeNode(2)
  # root_1.left.right.right = TreeNode(1)
  # root_1.right.right = TreeNode(11)
    root_1 = sol.deserialize('10,5,-3,3,2,#,11,3,2,#,1')
    expected_1 = 3
    output_1 = sol.pathSumL(root_1,8)
  # check(expected_1, output_1)
    print(sol.serialize(root_1),output_1)
  # check(expected_1, output_1)
  # print(sol.serialize0(root_1))

  # root_1 = TreeNode(8)
  # root_1.left = TreeNode(3)
  # root_1.right = TreeNode(10)
  # root_1.left.left = TreeNode(1)
  # root_1.left.right = TreeNode(6)
  # root_1.left.right.left = TreeNode(4)
  # root_1.left.right.right = TreeNode(7)
  # root_1.right.right = TreeNode(14)
  # root_1.right.right.left = TreeNode(13)
    root_1 = sol.deserialize('8,3,10,1,6,#,14,#,#,4,7,13')
    expected_1 = 4
    output_1 = visible_nodes(root_1)
  # check(expected_1, output_1)
  # print(sol.serialize0(root_1))

  # root_2 = TreeNode(10)
  # root_2.left = TreeNode(8)
  # root_2.right = TreeNode(15)
  # root_2.left.left = TreeNode(4)
  # root_2.left.left.right = TreeNode(5)
  # root_2.left.left.right.right = TreeNode(6)
  # root_2.right.left =TreeNode(14)
  # root_2.right.right = TreeNode(16)
    root_2 = sol.deserialize('10,8,15,4,#,14,16,#,5,#,#,#,#,#,6')
    expected_2 = 5
    output_2 = visible_nodes(root_2)
  # check(expected_2, output_2)
  # print(sol.serialize0(root_2))
    root_3 = sol.deserialize("460,3871,4698,8399,504,4421,7515,#,4167,5727,#,#,3096,434,7389,2667,5661,1969,7815,4292,3006,9750,6693,#,6906")
    expected_3 = [[8399,2667,1969],[3871,4167,5727],[460,504,4421,5661,7815,4292,9750],[4698,3096,434],[7515,3006,6693],[7389],[6906]]
    ans = sol.verticalTraversalPreOrder(root_3)
    assert ans == expected_3
    root_3 = sol.deserialize("8958,4812,4370,6034,360,1572,#,3487,1216,6218,1380,5258,#,3672,8645,#,2187,#,1297,3176,#,3587,2400,1220,7530,#,2082,1443,9407,#,292,1847,2077,#,5338,9868,4001,7072,2556,#,8544,8970,#,5152,#,1775,4173,#,882,3578,#,974,#,5034,9473,#,591,4155,2372,5071,4575,200,2029,9440,180,5631,#,3699,5880,3383,#,2191,2635,#,5388,5544,179,8450,2441,#,254,7012,224,6142,6696,4063,#,2673,2005")
    expected_3 = [[5071],[7072],[1220,4575,200],[3672,2556],[3487,7530,2029,9440,5631,3699,5544],[6034,8645,3587,8544,8970,5152,3578,5034],[4812,1216,6218,5258,2082,1443,1847,5338,9868,180,5880,3383,179,8450,254,7012,6142,4063],[8958,360,1572,2187,1297,3176,2400,1775,974,9473,591,4155],[4370,1380,9407,292,2077,4001,2191,2441,224,6696,2673],[4173,882,2372 ],[2635,5388,2005]]
    ans = sol.verticalTraversalPreOrder(root_3)
    assert ans == expected_3
    root_3 = sol.deserialize("6,3,7,2,5,#,9")
    expected_3 = [[2],[3],[6,5],[7],[9]]
    assert sol.verticalTraversal(root_3) == expected_3
    root_3 = sol.deserialize("1,2,3,4,5")
    expected_3 = [[4],[2],[1,5],[3]]
    assert sol.verticalTraversal(root_3) == expected_3
    root_3 = sol.deserialize("3,9,20,#,#,15,7")
    expected_3 = [[9],[3,15],[20],[7]]
    assert sol.verticalTraversal(root_3) == expected_3
    root_3 = sol.deserialize("1,2,3,4,6,5,7")
    expected_3 = [[4],[2],[1,5,6],[3],[7]]
    assert sol.verticalTraversal(root_3) == expected_3
    root_3 = sol.deserialize("1,2,3,4,5,6,7")
    expected_3 = [[4],[2],[1,5,6],[3],[7]]
    assert sol.verticalTraversal(root_3) == expected_3
    root_3 = sol.deserialize("3,1,4,0,2,2")
    expected_3 = [[0], [1], [3, 2, 2], [4]]
    assert sol.verticalTraversal(root_3) == expected_3
#73 15 20 34 35 5 14 16 26 -1 25 23 -1 30 3 36 -1 -1 7 24 11 32 -1 -1 21 -1 -1 -1 29 4 9 -1 33 13 -1 -1 -1 -1 22 31 -1 27 19 1 -1 12 18 6 -1 -1 -1 2 -1 -1 -1 -1 10 -1 -1 -1 -1 8 -1 28 -1 -1 -1 -1 -1 17 -1 -1 -1 -1

