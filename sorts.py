import os, sys, re, math
from random import randint
from timeit import timeit
from timeit import repeat
from functools import cmp_to_key
import heapq
class Solution:
# https://github.com/tuvo1106/python_sorting_algorithms
# https://leetcode.com/problems/largest-number/discuss/53298/Python-different-solutions-(bubble-insertion-selection-merge-quick-sorts).
    def largestNumber(self, nums):
        if not any(nums): return "0"
        compare = lambda a, b: 1 if a+b < b+a else -1 if a+b > b+a else 0
        return "".join(sorted(map(str, nums), key=cmp_to_key(compare)))

    def compare(self, n1, n2):
        return str(n1) + str(n2) > str(n2) + str(n1)

    # bubble sort
    def largestNumber2(self, nums):
        for i in range(len(nums), 0, -1):
            for j in range(i-1):
                if not self.compare(nums[j], nums[j+1]):
                    nums[j], nums[j+1] = nums[j+1], nums[j]
        return str(int("".join(map(str, nums))))

    # selection sort
    def largestNumber3(self, nums):
        for i in range(len(nums), 0, -1):
            tmp = 0
            for j in range(i):
                if not self.compare(nums[j], nums[tmp]):
                    tmp = j
            nums[tmp], nums[i-1] = nums[i-1], nums[tmp]
        return str(int("".join(map(str, nums))))
        
    # insertion sort
    def largestNumber4(self, nums):
        for i in range(len(nums)):
            pos, cur = i, nums[i]
            while pos > 0 and not self.compare(nums[pos-1], cur):
                nums[pos] = nums[pos-1]  # move one-step forward
                pos -= 1
            nums[pos] = cur
        return str(int("".join(map(str, nums))))
    
    # merge sort        
    def largestNumber5(self, nums):
        nums = self.mergeSort(nums, 0, len(nums)-1)
        return str(int("".join(map(str, nums))))
        
    def mergeSort(self, nums, l, r):
        if l > r: return 
        if l == r: return [nums[l]]
        mid = l + (r-l)//2
        left = self.mergeSort(nums, l, mid)
        right = self.mergeSort(nums, mid+1, r)
        return self.merge(left, right)
        
    def merge(self, l1, l2):
        res, i, j = [], 0, 0
        while i < len(l1) and j < len(l2):
            if not self.compare(l1[i], l2[j]):
                res.append(l2[j])
                j += 1
            else:
                res.append(l1[i])
                i += 1
        res.extend(l1[i:] or l2[j:])
        return res
        
    # quick sort, in-place
    def largestNumber(self, nums):
        self.quickSort(nums, 0, len(nums)-1)
        return str(int("".join(map(str, nums)))) 
    
    def quickSort(self, nums, l, r):
        if l >= r:
            return 
        pos = self.partition(nums, l, r)
        self.quickSort(nums, l, pos-1)
        self.quickSort(nums, pos+1, r)
        
    def partition(self, nums, l, r):
        low = l
        while l < r:
            if self.compare(nums[l], nums[r]): # str(nums[l]) + str(nums[r]) > str(nums[r]) + str(nums[l]) 
                nums[l], nums[low] = nums[low], nums[l] # compare == 1, low/l moves in-sync
                low += 1 # move low to pivot index
            l += 1 # compare next left value to pivot right value
        nums[low], nums[r] = nums[r], nums[low] # nums[low] gets the pivot value
        return low
sol = Solution()
print(sol.largestNumber5([3,30,34,5,9]))

def countSwaps(a):
    count = 0
    for i in range(len(a)):
        for j in range(len(a)-1):
            if a[j] > a[j + 1]: #N**2
                a[j], a[j + 1] = a[j+1], a[j]
                count += 1
    print(f"Array is sorted in {count} swaps.")  
    print(f"First Element: {a[0]}")  
    print(f"Last Element: {a[-1]}")

class Player:
    def __init__(self, name, score):
        self.name = name
        self.score = score
        
    def __repr__(self):
        return self.name+'_'+self.score
        
    def comparator(a, b):
        if a.score > b.score:
            return -1
        elif a.score < b.score:
            return 1
        elif a.name > b.name:
            return 1
        elif a.name < b.name:
            return -1
        else:
            return 0

def mergeSort(a):
  if len(a) > 1:
    mid = len(a)//2
    l = a[:mid]
    r = a[mid:]
    mergeSort(l)
    mergeSort(r)

    i = j = k = 0
    while i < len(l) and j < len(r):
      if l[i] < r[j]:
        a[k] = l[i]
        i += 1
      else:
        a[k] = r[j]
        j += 1
      k += 1
    while i < len(l):
      a[k] = l[i]
      k += 1
      i += 1
    while j < len(r):
      a[k] = r[j]
      j += 1
      k += 1

def insertSort2(a):
    n = len(a)
    for i in range(1,n):
        j = i - 1
        while j > 0 and a[i] < a[j]:
            a[i],a[j] = a[j],a[i]
            i -= 1
            j -= 1
        if a[j] > a[i]:
            a[i],a[j] = a[j],a[i]
        print(*a)
def quickSort(a):
    if len(a) < 2: return a
    l, r, e = [], [], a[:1]
    pV = e[0]
    for v in a[1:]:
        if v < pV: l.append(v)
        elif v > pV: r.append(v)
        else: e.append(v)
    result = quickSort(l) + e + quickSort(r)
    print(' '.join(map(str, result)))
    return result

def partition(a, low, high):
  j = low-1; pivot = a[high]
  for i in range(low,high):
      if a[i] < pivot: # move smaller numbers to low end, after j index
        j += 1; a[i], a[j] = a[j], a[i]
  j += 1
  a[j], a[high] = a[high], a[j] # swap pivot value to its position
  return j # pivot position, left (right) partition values < (>=) than pivot value 
def quickSort2(a, low, high):
  if low < high:
      pi = partition(a, low, high)
      quickSort2(a, low, pi-1)
      quickSort2(a, pi+1, high)

def maxheapify(a, n, i): #logN
    l = 2*i + 1; r = l + 1
    largest = i
    if l < n and a[l] > a[largest]:
        largest = l
    if r < n and a[r] > a[largest]:
        largest = r
    if largest != i: # swap i value with larger of left or right
        a[i], a[largest] = a[largest], a[i]
        maxheapify(a, n, largest)

def maxheappush(a, n):
    pass

def maxheappop(a):
    if len(a) == 0: raise IndexError
    a0 = a[0] # max value
    if len(a) == 1: a.pop()
    else:
        a[0] = a.pop() # move last to root
        maxheapify(a,len(a),0) # heapify root
    return a0

def nlargest(k, a):
    nl = []
    for i in range(k):
        nl.append(maxheappop(a))
    return nl

def maxheap(a): #NlogN
    n = len(a)
    for i in range(n//2-1, -1, -1):
        maxheapify(a, n, i) # heapify from leaves to root

def heapsort(a): #NlogN
    n = len(a)
    for i in range(n//2-1, -1, -1):
        maxheapify(a, n, i)
    for i in range(n-1, 0, -1): #sorting: move max to end, heapify 0 for n-2 elements
        a[i], a[0] = a[0], a[i]
        maxheapify(a,i,0)

def heapqsort(iterable):
    h = iterable[:]
    heapq.heapify(h)
    return [heapq.heappop(h) for i in range(len(h))]
class Solution:
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
        return end - begin + 1
    # https://www.youtube.com/watch?v=uQ_YsvOuXRY https://github.com/shreya367/InterviewBit/blob/master/Array/Repeat%20and%20missing%20number%20array
    def repeatMissingNumbers(A):
        aSum, a2Sum = sum(A), sum((n*n for n in A))
        nSum, n2Sum = sum(range(1,len(A)+1)), sum((n*n for n in range(1,len(A)+1)))
        x, y = aSum - nSum, a2Sum - n2Sum
        a = (x + y/x)/2
        return [a, a-x]

def subUnsort(A):
    lenA = len(A)
    if lenA < 2: return 0
    if lenA < 3: return 2 if A[0] > A[1] else 0 
    i = -1
    maxVal = -sys.maxsize
    k = 0
    j = 1
    for j in range(1,lenA):
        if A[j] < A[j-1] or A[j] < maxVal:
            if i == -1:
                i = j-1
            while i > 0: # grow left, as A[j] is new min
                if A[i-1] <= A[j]: break
                i -= 1
            k = j # k points to new min
            maxVal = max(maxVal, A[j-1])
    return 0 if [i,j] == [-1,lenA-1] else k-i+1
    # return [-1] if [i,j] == [0,len(A)-1] else [i,j]
# https://realpython.com/sorting-algorithms-python/
def run_sorting_algorithm(algorithm, array):
    # Set up the context and prepare the call to the specified
    # algorithm using the supplied array. Only import the
    # algorithm function if it's not the built-in `sorted()`.
    setup_code = f"from __main__ import {algorithm}" \
        if algorithm != "sorted" else ""

    stmt = f"{algorithm}({array})"

    # Execute the code ten different times and return the time
    # in seconds that each execution took
    times = repeat(setup=setup_code, stmt=stmt, repeat=3, number=10)

    # Finally, display the name of the algorithm and the
    # minimum time it took to run
    print(f"Algorithm: {algorithm}. Minimum execution time: {min(times)}")
ARRAY_LENGTH = 1000
if __name__ == '__main__':
    sol = Solution()
    print(sol.subUnsort([-1,-1,-1,-1,-1]))
    print(sol.subUnsort([5,4,3,2,1]))
    print(sol.subUnsort([2,3,3,2,4]))
    print(sol.subUnsort([3,2,3,2,4]))
    print(sol.subUnsort([1, 3, 2, 4, 5]))
    array = [randint(0, 1000) for i in range(ARRAY_LENGTH)]
    run_sorting_algorithm(algorithm="sorted", array=array)
    insertSort2([1, 4, 3, 5, 6, 2])
    print(' '.join(map(str, quickSort([4, 5, 3, 7, 2]))))
    print(' '.join(map(str, quickSort([5, 8, 1, 3, 7, 9, 2]))))
    a = [5, 8, 1, 3, 7, 9, 2]
    b = a[:]
    c = a[:]
    d = a[:]
    e = a[:]
    maxheap(d)
    heapsort(a)
    quickSort2(b, 0, 6)
    mergeSort(c)
    print(heapqsort(a), a, b, c, quickSort([5, 8, 1, 3, 7, 9, 2]), nlargest(5,d))
    assert(a == b == c)
    countSwaps(e)
    print(e)

'''n = int(input())
data = []
for i in range(n):
    name, score = input().split()
    score = int(score)
    player = Player(name, score)
    data.append(player)
    
data = sorted(data, key=cmp_to_key(Player.comparator))
for i in data:
    print(i.name, i.score)
'''
