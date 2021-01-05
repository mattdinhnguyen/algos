def addTimeIntervals0(tIntervals):
    timeline = [0]*24
    for s,e in tIntervals:
        if timeline[s] < e:
            timeline[s] = e
    for s in range(24):
        e = timeline[s]
        if e:
            for i in range(s+1,e+1):
                if timeline[i] > e:
                    timeline[s] = timeline[i]
                timeline[i] = 0
    return sum([e-s for s,e in enumerate(timeline) if e])

def addTimeIntervals(tIntervals):
    hours = set()
    for first,last in tIntervals:
        i = first
        while i < last:
            # print(i)
            hours.add(i)
            i += 1
    return len(hours)

if __name__ == '__main__':
    tdata = [[[1,4],[6,8],[4,6],[10,15]],[(1,4),(2,3)],[(4,6),(1,2)],[(1,4),(6,8),(2,4),(7,9),(10,15)]]
    for td in tdata:
        print(addTimeIntervals(td))
