<<<<<<< HEAD
def dfs(graph, start, goal):
    # Stack for DFS
    stack = [[start]]
    visited = set()

    while stack:
        # Take last path (LIFO)
        path = stack.pop()
        node = path[-1]

        # Goal test
        if node == goal:
            return path

        if node not in visited:
            visited.add(node)

            # Add neighbors to stack
            for neighbor in graph[node]:
                new_path = list(path)
                new_path.append(neighbor)
                stack.append(new_path)

    return None


# Example graph
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': [],
    'F': ['G'],
    'G': []
}

result = dfs(graph, 'A', 'G')
print("Path found:", result)
=======

>>>>>>> 975aa8059aedf4aa40d9d54e5b4d2dd51312d189
