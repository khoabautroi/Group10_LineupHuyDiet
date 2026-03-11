import numpy as np
import matplotlib.pyplot as plt



# DIFFERENTIAL EVOLUTION

def differential_evolution(func,
                           bounds,
                           pop_size=50,
                           F=0.8,
                           CR=0.9,
                           generations=500,
                           seed=None):

    if seed is not None:
        np.random.seed(seed)

    dim = len(bounds)

    # ----- Initialize population -----
    pop = np.array([
        np.random.uniform(bounds[i][0], bounds[i][1], pop_size)
        for i in range(dim)
    ]).T

    fitness = np.array([func(ind) for ind in pop])

    best_history = []

    # ----- Evolution loop -----
    for gen in range(generations):
        new_pop = np.copy(pop)

        for i in range(pop_size):

            # Select r1,r2,r3 different from each other and i
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1, r2, r3 = np.random.choice(idxs, 3, replace=False)

            # Mutation (DE/rand/1)
            mutant = pop[r1] + F * (pop[r2] - pop[r3])

            # Boundary handling
            for d in range(dim):
                mutant[d] = np.clip(mutant[d],
                                    bounds[d][0],
                                    bounds[d][1])

            # Crossover (binomial)
            trial = np.copy(pop[i])
            j_rand = np.random.randint(dim)

            for j in range(dim):
                if np.random.rand() < CR or j == j_rand:
                    trial[j] = mutant[j]

            # Selection
            trial_fitness = func(trial)
            if trial_fitness < fitness[i]:
                new_pop[i] = trial
                fitness[i] = trial_fitness

        pop = new_pop

        best_history.append(np.min(fitness))

    best_idx = np.argmin(fitness)

    return pop[best_idx], fitness[best_idx], best_history



# BENCHMARK FUNCTIONS


def sphere(x):
    return np.sum(x**2)


def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))


def rosenbrock(x):
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 +
                  (x[:-1] - 1)**2)


def ackley(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)

    term1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / d))
    term2 = -np.exp(np.sum(np.cos(c * x)) / d)

    return term1 + term2 + a + np.e



# MAIN TEST

if __name__ == "__main__":

    dim = 30
    bounds = [(-5, 5)] * dim

    best_sol, best_val, history = differential_evolution(
        ackley,
        bounds,
        pop_size=80,
        F=0.8,
        CR=0.9,
        generations=2000,
        seed=42
    )

    print("Best solution:", best_sol)
    print("Best fitness:", best_val)

    # Plot convergence
    plt.plot(history)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("DE Convergence")
    plt.grid()
    plt.show()
