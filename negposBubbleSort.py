def negPosBubble(negpos):
    negs = []
    pos = []
    for n in negpos:
        if n < 0:
            negs.append(n)
        else:
            pos.append(n)        
    return negs+pos

def test(A,Z):
    print(negPosBubble(A))
    assert negPosBubble(A) == Z

test([-1, 1, 3, -2, 2], [-1, -2, 1, 3, 2])