import numpy as np
import random
import matplotlib.pyplot as plt
from itertools import permutations

# ==============================
# TLBO CORE
# ==============================

class TLBO:
    def __init__(self, obj_func, dim, bounds=None,
                 pop_size=30, iterations=100,
                 problem_type="continuous"):
        
        self.obj_func = obj_func
        self.dim = dim
        self.bounds = bounds
        self.pop_size = pop_size
        self.iterations = iterations
        self.problem_type = problem_type
        
        self.history = []

    # --------------------------
    # Initialization
    # --------------------------
    def initialize(self):
        if self.problem_type == "continuous":
            low, high = self.bounds
            return np.random.uniform(low, high,
                                     (self.pop_size, self.dim))
        
        elif self.problem_type == "tsp":
            pop = []
            for _ in range(self.pop_size):
                perm = np.random.permutation(self.dim)
                pop.append(perm)
            return np.array(pop)
        
        elif self.problem_type == "knapsack":
            return np.random.randint(0, 2,
                                     (self.pop_size, self.dim))

    # --------------------------
    # Continuous TLBO
    # --------------------------
    def optimize_continuous(self):
        pop = self.initialize()
        fitness = np.array([self.obj_func(x) for x in pop])

        for it in range(self.iterations):

            # Teacher Phase
            mean = np.mean(pop, axis=0)
            teacher = pop[np.argmin(fitness)]
            TF = random.randint(1, 2)

            for i in range(self.pop_size):
                new_sol = pop[i] + \
                          np.random.rand(self.dim) * \
                          (teacher - TF * mean)

                new_sol = np.clip(new_sol,
                                  self.bounds[0],
                                  self.bounds[1])

                new_fit = self.obj_func(new_sol)
                if new_fit < fitness[i]:
                    pop[i] = new_sol
                    fitness[i] = new_fit

            # Learner Phase
            for i in range(self.pop_size):
                j = random.randint(0, self.pop_size - 1)
                while j == i:
                    j = random.randint(0, self.pop_size - 1)

                if fitness[i] < fitness[j]:
                    diff = pop[i] - pop[j]
                else:
                    diff = pop[j] - pop[i]

                new_sol = pop[i] + np.random.rand(self.dim) * diff
                new_sol = np.clip(new_sol,
                                  self.bounds[0],
                                  self.bounds[1])

                new_fit = self.obj_func(new_sol)
                if new_fit < fitness[i]:
                    pop[i] = new_sol
                    fitness[i] = new_fit

            self.history.append(np.min(fitness))

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]

    # --------------------------
    # Combinatorial TLBO
    # --------------------------
    def optimize_combinatorial(self):
        pop = self.initialize()
        fitness = np.array([self.obj_func(x) for x in pop])

        for it in range(self.iterations):

            teacher = pop[np.argmin(fitness)]

            for i in range(self.pop_size):

                # Crossover style update
                if self.problem_type == "tsp":
                    new = self.tsp_crossover(pop[i], teacher)
                else:
                    new = self.binary_crossover(pop[i], teacher)

                new_fit = self.obj_func(new)
                if new_fit < fitness[i]:
                    pop[i] = new
                    fitness[i] = new_fit

            self.history.append(np.min(fitness))

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]

    # --------------------------
    # Operators
    # --------------------------
    def tsp_crossover(self, a, b):
        cut = random.randint(1, self.dim - 2)
        child = list(a[:cut])
        for city in b:
            if city not in child:
                child.append(city)
        return np.array(child)

    def binary_crossover(self, a, b):
        mask = np.random.randint(0, 2, self.dim)
        child = np.where(mask, a, b)
        return child

    # --------------------------
    # Run
    # --------------------------
    def run(self):
        if self.problem_type == "continuous":
            return self.optimize_continuous()
        else:
            return self.optimize_combinatorial()


# ==============================
# BENCHMARK FUNCTIONS
# ==============================

def sphere(x):
    return np.sum(x**2)

def rastrigin(x):
    A = 10
    return A*len(x) + np.sum(x**2 - A*np.cos(2*np.pi*x))


# ==============================
# TSP
# ==============================

def generate_tsp(n):
    coords = np.random.rand(n,2)
    dist = np.sqrt(((coords[:,None,:]-coords[None,:,:])**2).sum(axis=2))
    return coords, dist

def tsp_objective_factory(dist):
    def tsp(route):
        return sum(dist[route[i], route[(i+1)%len(route)]]
                   for i in range(len(route)))
    return tsp


# ==============================
# KNAPSACK
# ==============================

def knapsack_objective_factory(weights, values, capacity):
    def knapsack(x):
        total_w = np.sum(x * weights)
        total_v = np.sum(x * values)
        if total_w > capacity:
            return 1e6
        return -total_v
    return knapsack


# ==============================
# VISUALIZATION
# ==============================

def plot_convergence(history, title):
    plt.figure()
    plt.plot(history)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.show()


def plot_tsp_route(coords, route):
    ordered = coords[route]
    plt.figure()
    plt.plot(ordered[:,0], ordered[:,1], 'o-')
    plt.title("Best TSP Route")
    plt.show()


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    # -------- Sphere --------
    tlbo1 = TLBO(sphere, dim=10,
                 bounds=(-5.12,5.12),
                 iterations=100)
    best, fit = tlbo1.run()
    print("Sphere:", fit)
    plot_convergence(tlbo1.history, "Sphere")

    # -------- Rastrigin --------
    tlbo2 = TLBO(rastrigin, dim=10,
                 bounds=(-5.12,5.12),
                 iterations=100)
    best, fit = tlbo2.run()
    print("Rastrigin:", fit)
    plot_convergence(tlbo2.history, "Rastrigin")

    # -------- TSP --------
    coords, dist = generate_tsp(15)
    tsp_obj = tsp_objective_factory(dist)

    tlbo3 = TLBO(tsp_obj, dim=15,
                 iterations=200,
                 problem_type="tsp")
    best, fit = tlbo3.run()
    print("TSP:", fit)
    plot_convergence(tlbo3.history, "TSP")
    plot_tsp_route(coords, best)

    # -------- Knapsack --------
    n_items = 20
    weights = np.random.randint(1,20,n_items)
    values = np.random.randint(10,100,n_items)
    capacity = 50

    knapsack_obj = knapsack_objective_factory(
        weights, values, capacity)

    tlbo4 = TLBO(knapsack_obj,
                 dim=n_items,
                 iterations=200,
                 problem_type="knapsack")

    best, fit = tlbo4.run()
    print("Knapsack best value:",
          -fit)
    plot_convergence(tlbo4.history, "Knapsack")