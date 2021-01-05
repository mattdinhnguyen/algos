
def triangles(edges):
    vMap = dict()
    triangles = 0
    for e in edges:
        vMap.setdefault(e[0],[]).append(e[1])
        vMap.setdefault(e[1],[]).append(e[0])
    for vs in vMap.values():
        for v in vs:
            triangles += len(set(vMap[v]) & set(vs))

    return triangles/6

print(triangles([(0,1),(2,1),(0,2),(4,1)]))
print(triangles([(0,1),(2,1),(0,2),(4,1),(4,2)]))
print(triangles([(0,1),(2,1),(0,2),(4,1),(4,2),(4,0)]))