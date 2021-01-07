from collections import defaultdict
from typing import List, Dict, Set, Tuple
from heapq import heappush, heappop

class Graph():
    def __init__(self) -> None:
        """
        self.edges is a dict of all possible next nodes
        e.g. {'X': ['A', 'B', 'C', 'E'], ...}
        self.weights has all the weights between two nodes,
        with the two nodes as a tuple as the key
        e.g. {('X', 'A'): 7, ('X', 'B'): 2, ...}
        """
        self.edges: Dict[str, List[str]] = defaultdict(list)
        self.weights: Dict[Tuple[str,str], int] = {}
   
    def add_edge(self, from_node: str, to_node: str, weight: int) -> None:
        # Note: assumes edges are bi-directional
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.weights[(from_node, to_node)] = weight
        self.weights[(to_node, from_node)] = weight

def dijsktra(graph: Graph, initial: str, end: str):
    # shortest paths is a dict of nodes, whose value is a tuple of (previous node, weight)
    shortest_paths: Dict[str, Tuple[str,int]] = {initial: (None, 0)}
    current_node: str = initial
    visited: Set[str] = set()
    
    while current_node != end:
        visited.add(current_node)
        destinations = graph.edges[current_node]
        weight_to_current_node = shortest_paths[current_node][1]

        for next_node in destinations:
            weight = graph.weights[(current_node, next_node)] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)
        
        next_destinations = []
        for node in shortest_paths:
            if node not in visited:
                heappush(next_destinations, (shortest_paths[node][1], node))
        if not next_destinations:
            return "Route Not Possible"
        # next node is the destination with the lowest weight
        current_node = heappop(next_destinations)[1]
    
    # Work back through destinations in shortest path
    path = []
    while current_node is not None:
        path.append(current_node)
        current_node = shortest_paths[current_node][0]
    # Reverse path
    return path[::-1]

edges = [
    ('X', 'A', 7),
    ('X', 'B', 2),
    ('X', 'C', 3),
    ('X', 'E', 4),
    ('A', 'B', 3),
    ('A', 'D', 4),
    ('B', 'D', 4),
    ('B', 'H', 5),
    ('C', 'L', 2),
    ('D', 'F', 1),
    ('F', 'H', 3),
    ('G', 'H', 2),
    ('G', 'Y', 2),
    ('I', 'J', 6),
    ('I', 'K', 4),
    ('I', 'L', 4),
    ('J', 'L', 1),
    ('K', 'Y', 5),
]

graph: Graph = Graph()
for edge in edges:
    graph.add_edge(*edge)

print(dijsktra(graph, 'X', 'Y'))
