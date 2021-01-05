ENQUEUE = 1
DEQUEUE = 2
PRINT = 3

if __name__ == '__main__':
    head = 0
    que = []
    t = int(input())

    for _ in range(t):
        s = input()
        nums = [int(n) for n in s.split()]
        if nums[0] == ENQUEUE:
            que.append(nums[1])
        elif nums[0] == DEQUEUE:
            head += 1
        elif nums[0] == PRINT:
            print(que[head])
