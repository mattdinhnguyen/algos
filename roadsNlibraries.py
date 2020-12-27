def roadsAndLibraries(n, c_lib, c_road, roads):

    willVisit = [True]*(n+1) # cities to visit

    def DFS(city, stax):
        for ct in roads[city]:
            if willVisit[ct]:
                willVisit[ct] = False
                stax.append(ct)

    cost = 0
    for ct in range (1,n+1):
        if (willVisit[ct]):
            cicc = 0 # cities (in) connected (components)
            if ct in roads:
                stax = [ct]
                willVisit[ct] = False
                while stax:
                    DFS(stax.pop(), stax)
                    cicc += 1
            else:
                willVisit[ct] = False
                cicc += 1
            cost += c_lib + c_road * (cicc-1)

    return cost

if __name__ == "__main__":

    data = open("roadsNlibraries.ut","r")

    q = int(data.readline())

    for _ in range(q):
        n, m, c_lib, c_road = list(map(int, data.readline().strip().split()))
        roads = {}

        if (c_lib <= c_road):
            result=  c_lib * n
            for _ in range(m):
                data.readline()

        else:
            for _ in range(m):
                u,v = tuple(map(int, data.readline().strip().split()))
                roads.setdefault(u,[]).append(v)
                roads.setdefault(v,[]).append(u)

            result = roadsAndLibraries(n, c_lib, c_road, roads)

        print(result)
