<<<<<<< HEAD
import heapq

def greedy_best_first(graph, heuristic, start, goal):
    pq = [(heuristic[start], [start])]
    visited = set()

    while pq:
        h_value, path = heapq.heappop(pq)
        node = path[-1]

        if node == goal:
            return path

        if node not in visited:
            visited.add(node)

            for neighbor, cost in graph[node]:
                new_path = list(path)
                new_path.append(neighbor)
                heapq.heappush(pq, (heuristic[neighbor], new_path))

    return None


# Example weighted graph
graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('D', 2), ('E', 5)],
    'C': [('F', 1)],
    'D': [],
    'E': [],
    'F': [('G', 3)],
    'G': []
}

# Heuristic values (estimated distance to G)
heuristic = {
    'A': 7,
    'B': 6,
    'C': 2,
    'D': 5,
    'E': 4,
    'F': 1,
    'G': 0
}

result = greedy_best_first(graph, heuristic, 'A', 'G')
print("Path found:", result)
=======

>>>>>>> 975aa8059aedf4aa40d9d54e5b4d2dd51312d189
