import sys


def DFS(index, cities, count, listF):

    listF[index] = False

    count[0] +=1

    for i in range(len(cities[index])):

        if listF[cities[index][i]]:

            DFS(cities[index][i],cities,count, listF)



def roadsAndLibraries(n, c_lib, c_road, ct):

    listF = [True]*(n+1)

    cost = 0

    for i in range (1,n+1):

        if (listF[i]):

            count = [0]

            DFS(i, ct, count, listF)

            cost += c_lib + c_road * (count[0] - 1)

    return cost


if __name__ == "__main__":

    data = open("roadsNlibraries.ut")

    q = int(data.readline())

    for a0 in range(q):

        n, m, c_lib, c_road = data.readline().strip().split(' ')

        n, m, c_lib, c_road = [int(n), int(m), int(c_lib), int(c_road)]

        if (c_lib<=c_road):

            result=  c_lib * n

            for cities_i in range(m):

                cities_t = [int(cities_temp) for cities_temp in data.readline().strip().split(' ')]

        else:

            ct = []

            for i in range(n+1):
                item = []
                ct.append(item)

            for cities_i in range(m):

                cities_t = [int(cities_temp) for cities_temp in data.readline().strip().split(' ')]

                ct[cities_t[1]].append(cities_t[0])
                ct[cities_t[0]].append(cities_t[1])

            result = roadsAndLibraries(n, c_lib, c_road, ct)

        print(result)


