import math
from collections import defaultdict
# https://downey.io/notes/omscs/cs6515/dynamic-programming-shortests-paths/
# bellman-ford shortest paths algorithm

graph = { 
          'S': ['A'],
          'A': ['B'],
          'B': ['C', 'E'],
          'C': ['A', 'D'],
          'D': [],
          'E': []
        }

vertex_to_idx = { 
          'S': 0,
          'A': 1,
          'B': 2,
          'C': 3,
          'D': 4,
          'E': 5
        }

weights = { 
          'S': {'A': 6},
          'A': {'B': 3},
          'B': {'C': 4, 'E': 5},
          'C': {'A': -3, 'D': 3},
          'D': {},
          'E': {}
        }

num_edges = 0
num_vertices = 0
reversed_graph = defaultdict(list)
for vertex, edges in graph.items():
  num_edges += len(edges)
  num_vertices += 1
  for edge in edges:
    # Reversing the graph so we can later find which vertices directly lead to a particular vertex
    reversed_graph[edge].append(vertex)

d = [[math.inf for x in range(num_vertices)] for y in range(num_edges + 1)] 
d[0][vertex_to_idx['S']] = 0

for i in range(1, num_edges):
  for z in graph.keys():
    z_idx = vertex_to_idx[z]

    # Initialize the shortest path to z to the
    # path found in the previous subproblem.
    # Only update if new paths are shorter
    d[i][z_idx] = d[i-1][z_idx]

    # The reversed graph lets us find which vertexes
    # immediately lead to z
    for y in reversed_graph[z]:
      y_idx = vertex_to_idx[y]
      d[i][z_idx] = min(d[i][z_idx], d[i-1][y_idx] + weights[y][z])

print()
print("The memoization table (with headers):")
header_row = ['TBD'] * num_vertices
for vertex, idx in vertex_to_idx.items():
  header_row[idx] = vertex

print(header_row)
for i in range(0, num_edges):
  print(d[i])

print()
print("Final shortest paths to each vertex:")
print(d[num_edges - 1])
