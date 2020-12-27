from collections import deque
from typing import List, Deque, Dict
from copy import deepcopy

class Member:
    def __init__(self, id: int, name: str, friends: List[int] = []) -> None:
        self.id = id
        self.name = name
        self.friends = friends

    def AddFriends(self, friends: List[int]):
        self.friends += friends

    def SuggestingFriends(self):
        pass

class PathNode:
    def __init__(self, member: Member, previous) -> None:
	    self.member = member
	    self.previousNode = previous

    def Collapse(self, startsWithRoot: bool) -> Deque[Member]:
        path: Deque[Member] = deque()
        node = self
        while (node):
            if (startsWithRoot):
                path.append(node.member)
            else:
                path.appendleft(node.member) 
            node = node.previousNode

        return path

class BFSData:
    def __init__(self, root: Member) -> None:
        sourcePath: PathNode = PathNode(root, None)
        self.toVisit: Deque[PathNode] = deque([sourcePath])
        self.visited: Dict[int, PathNode] = {root.id : sourcePath}

    def isFinished(self) -> bool:
        return len(self.toVisit) == 0

def searchLevel(members: Dict[int, Member], primary: BFSData , secondary: BFSData) -> Member:
	# We only want to search one level at a time. Count how many nodes are currently 
	# in the primary's level and only do that many nodes. We continue to add nodes to the end.
    _toVisit: Deque[PathNode] = deepcopy(primary.toVisit)
    for pathNode in _toVisit:
        memberId = pathNode.member.id
        if memberId in secondary.visited:
            return pathNode.member

        member = pathNode.member
        for friendId in member.friends:
            if friendId not in primary.visited:
                friend = members.get(friendId)
                next = PathNode(friend, pathNode)
                primary.visited[friendId] = next
                primary.toVisit.append(next)
    return None

# Merge paths where searches met at the connection.
def mergePaths(bfs1: BFSData, bfs2: BFSData, connection: int) -> Deque[Member]:
	# end1 -> source, end2 -> dest 
    end1: PathNode = bfs1.visited.get(connection)
    end2: PathNode = bfs2.visited.get(connection)

    pathOne: Deque[Member] = end1.Collapse(False)
    pathTwo: Deque[Member] = end2.Collapse(True)

    pathTwo.popleft() # remove connection
    pathOne.extend(pathTwo) # add second path

    return pathOne

def findPathBiBFS(member: Dict[int, Member], source: int, destination: int) -> Deque[Member]:
	sourceData: BFSData = BFSData(member.get(source))
	destData: BFSData = BFSData(member.get(destination))

	while not sourceData.isFinished() and not destData.isFinished():
            collision: Member = searchLevel(member, sourceData, destData)
            if collision:
                return mergePaths(sourceData, destData, collision.id)

            # Search out from destination
            collision = searchLevel(member, destData, sourceData)
            if collision:
                return mergePaths(sourceData, destData, collision.id)

	return None


def AddFriendship(first: Member, friend: Member) -> None:
    first.AddFriends([friend.id])
    friend.AddFriends([first.id])

john = Member(1,"John",[2])
chi = Member(4, "Chi", [])
members: Dict[int, Member] = {1 : john, 2 : Member(2,"Matthew",[1,3]), 3 : Member(3,"Ly", [2,4]), 4 : chi}
AddFriendship(john, chi)

print(list(map(lambda m: m.name, findPathBiBFS(members, 1, 4))))
