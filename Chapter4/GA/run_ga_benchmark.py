import numpy as np
from GA_Continuous_Optimization import GA, sphere, rastrigin, rosenbrock, ackley, griewank

def run_benchmark(func, dim=2, runs=30):
    final_bests = []

    for seed in range(runs):
        ga = GA(
            func=func,
            dim=dim,
            pop_size=100,
            generations=100,
            lb=-5.12,
            ub=5.12,
            seed=seed
        )

        best_solution, best_value, best_hist, avg_hist = ga.run()
        final_bests.append(best_value)

    final_bests = np.array(final_bests)

    print("Function:", func.__name__)
    print("Dimension:", dim)
    print("Runs:", runs)
    print("Best (min):", np.min(final_bests))
    print("Mean:", np.mean(final_bests))
    print("Std:", np.std(final_bests))


if __name__ == "__main__":
    for dim in [2, 10, 30]:
        run_benchmark(rastrigin, dim=dim, runs=30)
        print("-" * 40)