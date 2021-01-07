# Python program for implementation of MergeSort 

# Merges two subarrays of arr[]. 
# First subarray is arr[l..m] 
# Second subarray is arr[m+1..r] 
def merge(arr, l, m, r): 
	L = arr[l:m+1] 
	R = arr[m+1:r+1]
	i = j = 0	 # Initial index of first/second subarray 
	k = l	 # Initial index of merged subarray 
	n1 = m - l + 1
	n2 = r - m
	while i < n1 and j < n2 : 
		if L[i] <= R[j]: 
			arr[k] = L[i] 
			i += 1
		else: 
			arr[k] = R[j] 
			j += 1
		k += 1
	# Copy the remaining elements of L[], if there 
	# are any 
	while i < n1: 
		arr[k] = L[i] 
		i += 1
		k += 1
	# Copy the remaining elements of R[], if there 
	# are any 
	while j < n2: 
		arr[k] = R[j] 
		j += 1
		k += 1

# l is for left index and r is right index of the 
# sub-array of arr to be sorted 
def mergeSort(arr,l,r): 
	if l < r:
		m = (l+r)//2
		mergeSort(arr, l, m)
		mergeSort(arr, m+1, r) 
		merge(arr, l, m, r) 

# Driver code to test above 
arr = [12, 11, 13, 5, 6, 7] 
print("Given array is", arr) 
mergeSort(arr,0,len(arr)-1) 
print ("Sorted array is", arr) 
