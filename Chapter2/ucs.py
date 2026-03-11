import heapq

def ucs(graph, start, goal):
    # Priority queue (cost, path)
    pq = [(0, [start])]
    visited = set()

    while pq:
        cost, path = heapq.heappop(pq)
        node = path[-1]

        if node == goal:
            return path, cost

        if node not in visited:
            visited.add(node)

            for neighbor, weight in graph[node]:
                new_cost = cost + weight
                new_path = list(path)
                new_path.append(neighbor)
                heapq.heappush(pq, (new_cost, new_path))

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

result = ucs(graph, 'A', 'G')
print("Path and cost:", result)
