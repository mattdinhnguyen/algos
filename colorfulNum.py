'''
Given Number : 3245
Output : Colorful
Number 3245 can be broken into parts like 3 2 4 5 32 24 45 324 245.
this number is a colorful number, since product of every digit of a sub-sequence are different.
That is, 3 2 4 5 (3*2)=6 (2*4)=8 (4*5)=20, (3*2*4)= 24 (2*4*5)= 40

Given Number : 326
Output : Not Colorful.
326 is not a colorful number as it generates 3 2 6 (3*2)=6 (2*6)=12.
'''
def allSubStrings(s):
    return [s[i:j] for i in range(len(s)) 
          for j in range(i + 1, len(s) + 1)]


def isColorful(i):
    found = set()
    sss = allSubStrings(str(i))
    for ss in sss:
        p = 1
        for i in list(ss):
            p *= int(i)
        if p in found:
            print(ss)
            return "NotColorful"
        found.add(p)
    return "Colorful"

print(allSubStrings("326"))
print(allSubStrings("3245"))
print(isColorful(326))
print(isColorful(3245))
