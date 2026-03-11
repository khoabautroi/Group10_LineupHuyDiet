<<<<<<< HEAD
import heapq

def a_star(graph, heuristic, start, goal):
    # Priority queue: (f_cost, g_cost, path)
    pq = [(heuristic[start], 0, [start])]
    visited = set()

    while pq:
        f_cost, g_cost, path = heapq.heappop(pq)
        node = path[-1]

        if node == goal:
            return path, g_cost

        if node not in visited:
            visited.add(node)

            for neighbor, cost in graph[node]:
                new_g = g_cost + cost
                new_f = new_g + heuristic[neighbor]
                new_path = list(path)
                new_path.append(neighbor)

                heapq.heappush(pq, (new_f, new_g, new_path))

    return None


# Example graph
graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('D', 2), ('E', 5)],
    'C': [('F', 1)],
    'D': [],
    'E': [],
    'F': [('G', 3)],
    'G': []
}

heuristic = {
    'A': 7,
    'B': 6,
    'C': 2,
    'D': 5,
    'E': 4,
    'F': 1,
    'G': 0
}

result = a_star(graph, heuristic, 'A', 'G')
print("Path and cost:", result)
=======

>>>>>>> 975aa8059aedf4aa40d9d54e5b4d2dd51312d189
