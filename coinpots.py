
# A & B to pick coin pots to get the most coins

from collections import deque

# dq = deque([1,2,3,4,5,6,7,8,9,10,11])
def max_coin(coin, start, end ):
  if start == end:
    return coin[start]

	# if we're left with only two pots, choose one with maximum coins
  if start + 1 == end:
    return max(coin[start], coin[end])

  a = coin[start] + min( max_coin( coin, start+2,end ), max_coin( coin, start+1,end-1 ) )
  b = coin[end] + min( max_coin( coin, start+1,end-1 ), max_coin( coin, start,end-2 ) )
	
  return max(a,b)

DP_map = dict()
def dp_max_coin(coin, start, end):
  if start == end:
    return coin[start]

  # if we're left with only two pots, choose one with maximum coins
  if start + 1 == end:
    return max(coin[start], coin[end])

  if (start, end) in DP_map:
    return DP_map[(start, end)]

  a = coin[start] + min( dp_max_coin( coin, start+2,end ), dp_max_coin( coin, start+1,end-1 ) )
  b = coin[end] + min( dp_max_coin( coin, start+1,end-1 ), dp_max_coin( coin, start,end-2 ) )

  ret = max(a, b)
  DP_map[(start, end)] = ret
  return ret

# Recursive function to maximize the number of coins collected by a player,
# assuming that opponent also plays optimally
def optimalStrategy(coin, i, j):

	# base case: one pot left, only one choice possible
	if i == j:
		return coin[i]

	# if we're left with only two pots, choose one with maximum coins
	if i + 1 == j:
		return max(coin[i], coin[j])

	# if player chooses front pot i, opponent is left to choose from [i+1, j].
	# 1. if opponent chooses front pot i+1, recur for [i+2, j]
	# 2. if opponent chooses rear pot j, recur for [i+1, j-1]

	start = coin[i] + min(optimalStrategy(coin, i + 2, j),
						  optimalStrategy(coin, i + 1, j - 1))

	# if player chooses rear pot j, opponent is left to choose from [i, j-1].
	# 1. if opponent chooses front pot i, recur for [i+1, j-1]
	# 2. if opponent chooses rear pot j-1, recur for [i, j-2]

	end = coin[j] + min(optimalStrategy(coin, i + 1, j - 1),
						optimalStrategy(coin, i, j - 2))

	# return maximum of two choices
	return max(start, end)


# Pots of Gold Game using Dynamic Programming
if __name__ == '__main__':

	# pots of gold (even number) arranged in a line
  coin = [4, 6, 2, 3]
  # coin = [1,2,3,4,5,6,7,8,9,10,11]
  # coin = [1, 100, 60, 20, 1]

  print("Maximum coins collected by player is", 
		  optimalStrategy(coin, 0, len(coin) - 1),max_coin(coin,0,len(coin)-1),dp_max_coin(coin,0,len(coin)-1))
