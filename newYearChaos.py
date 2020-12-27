'''It's New Year's Day and everyone's in line for the Wonderland rollercoaster ride!
There are a number of people queued up, and each person wears a sticker indicating their initial position in the queue.
Initial positions increment by  from  at the front of the line to  at the back.

Any person in the queue can bribe the person directly in front of them to swap positions.
If two people swap positions, they still wear the same sticker denoting their original places in line. One person can bribe at most two others.
For example, if  and  bribes , the queue will look like this: .

Fascinated by this chaotic queue, you decide you must know the minimum number of bribes that took place to get the queue into its current state!

Function Description

Complete the function minimumBribes in the editor below. It must print an integer representing the minimum number of bribes necessary,
or Too chaotic if the line configuration is not possible.

minimumBribes has the following parameter(s):

q: an array of integers
Input Format

The first line contains an integer , the number of test cases.

Each of the next  pairs of lines are as follows:
- The first line contains an integer , the number of people in the queue
- The second line has  space-separated integers describing the final state of the queue.

Constraints

Subtasks

For  score 
For  score 

Output Format

Print an integer denoting the minimum number of bribes needed to get the queue into its final state.
Print Too chaotic if the state is invalid, i.e. it requires a person to have bribed more than  people.
space/time n/n
'''
def minimumBribes1(Q):
    #
    # initialize the number of moves
    moves = 0 
    #
    # decrease Q by 1 to make index-matching more intuitive
    # so that our values go from 0 to N-1, just like our
    # indices.  (Not necessary but makes it easier to
    # understand.)
    Q = [P-1 for P in Q]
    #
    # Loop through each person (P) in the queue (Q)
    for i,P in enumerate(Q):
        # i is the current position of P, while P is the
        # original position of P.
        #
        # First check if any P is more than two ahead of 
        # its original position
        if P - i > 2:
            print("Too chaotic")
            return
        #
        # From here on out, we don't care if P has moved
        # forwards, it is better to count how many times
        # P has RECEIVED a bribe, by looking at who is
        # ahead of P.  P's original position is the value
        # of P.

        # Anyone who bribed P cannot get to higher than
        # one position in front if P's original position,
        # so we need to look from one position in front
        # of P's original position to one in front of P's
        # current position, and see how many of those 
        # positions in Q contain a number large than P.

        # In other words we will look from P-1 to i-1,
        # which in Python is range(P-1,i-1+1), or simply
        # range(P-1,i).  To make sure we don't try an
        # index less than zero, replace P-1 with
        # max(P-1,0)
        for j in range(max(P-1,0),i):
            if Q[j] > P:
                moves += 1
    return moves
def minimumBribes0(q):
    minBrides = 0
    riderPos = dict()
    for i,n in enumerate(q):
        if n-i > 3:
            return "Too chaotic"
        riderPos[n] = i
    for rider in range(len(q),0,-1):
        brides = rider - riderPos[rider] - 1
        if brides > 0:
            minBrides += brides
            pos = riderPos[rider]
            if brides > 1:
                q[pos], q[pos+1], q[pos+2] = q[pos+1], q[pos+2], q[pos]
                riderPos[q[pos+1]] -= 1
                riderPos[q[pos]] -= 1
            else:
                q[pos], q[pos+1] = q[pos+1], q[pos]
                riderPos[q[pos]] -= 1
            riderPos[rider] += brides

    return minBrides

def minimumBribes(q):
    minBribes = 0
    q = [v-1 for v in q]
    for i,p in enumerate(q):
        if p-i > 2:
            return("Too chaotic")
        for j in range(max(0,p-1),i):
            if q[j] > p:
                minBribes += 1
    return minBribes

# 1 2 5 3 7 8 6 4
if __name__ == '__main__':
    fptr = open("newYearChaos.ut", "r")
    t = int(fptr.readline())
    for t_itr in range(t):
        n = int(fptr.readline())

        q = list(map(int, fptr.readline().rstrip().split()))
        print(list(range(len(q))))
        print([v-1 for v in q])
        print(minimumBribes(q),minimumBribes1(q),minimumBribes0(q))
        print(q)
        break
