<<<<<<< HEAD
from collections import deque

def bfs(graph, start, goal):
    # Queue for BFS
    queue = deque([[start]])
    
    # Visited set to avoid revisiting nodes
    visited = set()

    while queue:
        # Get first path from queue
        path = queue.popleft()
        node = path[-1]

        # If goal found
        if node == goal:
            return path

        if node not in visited:
            visited.add(node)

            # Explore neighbors
            for neighbor in graph[node]:
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)

    return None


# Example graph (Adjacency List)
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': [],
    'F': ['G'],
    'G': []
}

# Run BFS
result = bfs(graph, 'A', 'D')
print("Path found:", result)
=======

>>>>>>> 975aa8059aedf4aa40d9d54e5b4d2dd51312d189
