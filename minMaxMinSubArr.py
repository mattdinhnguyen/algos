# https://www.hackerrank.com/challenges/angry-children/problem
# given a list of integers, , and a single integer . You must create an array of length  from elements of  such that its unfairness is minimized.
def maxMin(k, arr):
    sArr = sorted(arr)
    minMaxMin = float("Inf")
    for i in range(k-1,len(arr)):
        minMaxMin = min(minMaxMin, sArr[i]-sArr[i-k+1])
    return minMaxMin
    
if __name__ == '__main__':

    arr,k = [10,100,300,200,1000,20,30],3

    print(maxMin(k, arr))

