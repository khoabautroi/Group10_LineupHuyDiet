import numpy as np
import matplotlib.pyplot as plt

def make_knapsack_instance(num_items=20, capacity=47, seed=47):
    rng = np.random.default_rng(seed)
    weights = rng.integers(1, 20, num_items)
    values = rng.integers(10, 100, num_items)
    return weights, values, capacity

class GA_Knapsack:
    def __init__(self, weights, values, capacity,
                 pop_size=100, generations=1000, mutation_rate=0.8, seed=None):
        self.weights = np.array(weights)
        self.values = np.array(values)
        self.capacity = capacity

        self.num_items = len(weights)
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate

        if seed is not None:
            np.random.seed(seed)

        self.population = np.random.randint(0, 2, (pop_size, self.num_items))
        self.best_history = []

    def fitness(self, individual):
        total_weight = np.sum(individual * self.weights)
        total_value = np.sum(individual * self.values)
        if total_weight > self.capacity:
            return 0
        return total_value

    def evaluate(self):
        return np.array([self.fitness(ind) for ind in self.population])

    def selection(self, fitness):
        idx = np.argsort(-fitness)
        return self.population[idx[:self.pop_size // 2]]

    def crossover(self, parents):
        children = []
        while len(children) < self.pop_size:
            p1, p2 = parents[np.random.randint(len(parents), size=2)]
            point = np.random.randint(1, self.num_items - 1)
            child = np.concatenate([p1[:point], p2[point:]])
            children.append(child)
        return np.array(children)

    def mutation(self, population):
        for i in range(len(population)):
            for j in range(self.num_items):
                if np.random.rand() < self.mutation_rate:
                    population[i][j] = 1 - population[i][j]
        return population

    def run(self):
        self.best_history = []

        for _ in range(self.generations):
            fitness = self.evaluate()
            self.best_history.append(np.max(fitness))

            parents = self.selection(fitness)
            children = self.crossover(parents)
            self.population = self.mutation(children)

        fitness = self.evaluate()
        best_idx = np.argmax(fitness)
        best_solution = self.population[best_idx]
        best_value = fitness[best_idx]
        return best_solution, best_value, self.best_history

if __name__ == "__main__":
    weights, values, capacity = make_knapsack_instance(num_items=20, capacity=47, seed=47)

    ga = GA_Knapsack(weights, values, capacity, pop_size=100, generations=1000, mutation_rate=0.01, seed=42)
    best_solution, best_value, history = ga.run()

    print("Best Value:", best_value)
    print("Total Weight:", np.sum(best_solution * weights))
    print("Selected Items:", best_solution)

    plt.plot(history)
    plt.title("Knapsack GA Convergence")
    plt.xlabel("Generation")
    plt.ylabel("Best Value")
    plt.show()