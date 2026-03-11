import numpy as np
from DE import differential_evolution, sphere, rastrigin, rosenbrock, ackley

def run_benchmark(func, dim=30, runs=30,
                  pop_size=80, F=0.8, CR=0.9, generations=2000,
                  lb=-5, ub=5):
    bounds = [(lb, ub)] * dim

    final_bests = []

    for seed in range(runs):
        best_sol, best_val, history = differential_evolution(
            func,
            bounds,
            pop_size=pop_size,
            F=F,
            CR=CR,
            generations=generations,
            seed=seed
        )
        final_bests.append(best_val)

    final_bests = np.array(final_bests)

    print("Function:", func.__name__)
    print("Dimension:", dim)
    print("Runs:", runs)
    print("pop_size:", pop_size, "generations:", generations, "F:", F, "CR:", CR)
    print("Best (min):", np.min(final_bests))
    print("Mean:", np.mean(final_bests))
    print("Std:", np.std(final_bests))

if __name__ == "__main__":
    for dim in [2, 10, 30]:
        run_benchmark(ackley, dim=dim, runs=30, pop_size=80, generations=200, F=0.8, CR=0.9, lb=-5, ub=5)
        print("-" * 40)