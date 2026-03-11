import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def sphere(x):
    return np.sum(x**2)

def rastrigin(x):
    n = len(x)
    return 10*n + np.sum(x**2 - 10*np.cos(2*np.pi*x))

def rosenbrock(x):
    return np.sum(100*(x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

def ackley(x):
    n = len(x)
    return (-20 * np.exp(-0.2 * np.sqrt(np.sum(x**2)/n))
            - np.exp(np.sum(np.cos(2*np.pi*x))/n)
            + 20 + np.e)

def griewank(x):
    sum_part = np.sum(x**2)/4000
    prod_part = np.prod(np.cos(x/np.sqrt(np.arange(1, len(x)+1))))
    return sum_part - prod_part + 1

class GA:
    def __init__(self, func, dim=2, pop_size=100,
                 generations=100, lb=-5.12, ub=5.12, seed=None):
        self.func = func
        self.dim = dim
        self.pop_size = pop_size
        self.generations = generations
        self.lb = lb
        self.ub = ub
        self.tournament_k = 5

        if seed is not None:
            np.random.seed(seed)

        self.population = np.random.uniform(lb, ub, (pop_size, dim))
        self.best_history = []
        self.avg_history = []

    def evaluate(self):
        return np.array([self.func(ind) for ind in self.population])

    def selection(self, fitness):
        k = self.tournament_k
        parents = []
        for _ in range(self.pop_size // 2):
            idx = np.random.choice(self.pop_size, k, replace=False)
            best = idx[np.argmin(fitness[idx])]
            parents.append(self.population[best])
        return np.array(parents)

    def crossover(self, parents):
        children = []
        while len(children) < self.pop_size:
            p1, p2 = parents[np.random.randint(len(parents), size=2)]
            alpha = np.random.rand()
            child = alpha*p1 + (1-alpha)*p2
            children.append(child)
        return np.array(children)

    def mutation(self, population, rate=0.15):
        for i in range(len(population)):
            if np.random.rand() < rate:
                population[i] += np.random.normal(0, 0.1, self.dim)
        return np.clip(population, self.lb, self.ub)

    def run(self):
        self.best_history = []
        self.avg_history = []

        for _ in range(self.generations):
            fitness = self.evaluate()
            self.best_history.append(np.min(fitness))
            self.avg_history.append(np.mean(fitness))

            elite_k = min(2, self.pop_size - 1)
            elite_idx = np.argsort(fitness)[:elite_k]
            elite = self.population[elite_idx]

            parents = self.selection(fitness)
            children = self.crossover(parents)
            children = self.mutation(children)

            self.population = np.vstack((elite, children[:self.pop_size - elite_k]))

        fitness = self.evaluate()
        best_idx = np.argmin(fitness)
        best_solution = self.population[best_idx]
        best_value = fitness[best_idx]

        return best_solution, best_value, self.best_history, self.avg_history

def visualize(func, best_point, best_history, avg_history=None, lb=-5.12, ub=5.12):
    x = np.linspace(lb, ub, 200)
    y = np.linspace(lb, ub, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[func(np.array([i, j])) for i in x] for j in y])

    fig = plt.figure(figsize=(14, 5))

    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(X, Y, Z)
    ax.scatter(best_point[0], best_point[1], func(best_point))
    ax.set_title("Function Landscape")

    ax2 = fig.add_subplot(122)
    ax2.plot(best_history, label="best")
    if avg_history is not None:
        ax2.plot(avg_history, label="avg")
    ax2.set_title("Convergence Curve")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Fitness")
    ax2.legend()

    plt.show()

if __name__ == "__main__":
    ga = GA(func=griewank, dim=2, pop_size=100, generations=100, lb=-5.12, ub=5.12, seed=42)
    best_solution, best_value, best_hist, avg_hist = ga.run()

    print("Best Solution:", best_solution)
    print("Best Value:", best_value)
    print("History length:", len(best_hist), len(avg_hist))

    visualize(ga.func, best_solution, best_hist, avg_history=avg_hist, lb=ga.lb, ub=ga.ub)