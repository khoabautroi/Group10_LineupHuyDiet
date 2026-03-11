import numpy as np
from GA_Knapsack import GA_Knapsack, make_knapsack_instance

def run_benchmark_knapsack(runs=30, pop_size=100, generations=1000, mutation_rate=0.01):
    weights, values, capacity = make_knapsack_instance(num_items=20, capacity=47, seed=47)

    final_bests = []
    final_weights = []

    for seed in range(runs):
        ga = GA_Knapsack(
            weights, values, capacity,
            pop_size=pop_size,
            generations=generations,
            mutation_rate=mutation_rate,
            seed=seed
        )
        best_solution, best_value, history = ga.run()
        total_weight = np.sum(best_solution * weights)

        final_bests.append(best_value)
        final_weights.append(total_weight)

    final_bests = np.array(final_bests)
    final_weights = np.array(final_weights)

    print("Knapsack Benchmark")
    print("Runs:", runs)
    print("Items:", len(weights), "Capacity:", capacity)
    print("pop_size:", pop_size, "generations:", generations, "mutation_rate:", mutation_rate)
    print("Best (max):", np.max(final_bests))
    print("Mean:", np.mean(final_bests))
    print("Std:", np.std(final_bests))
    print("Feasible rate:", np.mean(final_weights <= capacity))

if __name__ == "__main__":
    run_benchmark_knapsack(runs=30, pop_size=100, generations=1000, mutation_rate=0.01)