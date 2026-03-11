import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import math

# HÀM MỤC TIÊU
def cost_function(x, y):
    return x**2 + y**2


# SIMULATED ANNEALING
class SimulatedAnnealing:
    def __init__(self, cost_func, bounds, n_iterations):
        self.cost_func = cost_func
        self.bounds = bounds
        self.n_iterations = n_iterations

        # Khởi tạo điểm ban đầu
        self.current_position = np.array([
            random.uniform(bounds[0], bounds[1]),
            random.uniform(bounds[0], bounds[1])
        ])

        self.best_position = self.current_position.copy()
        self.best_value = float('inf')

        self.temperature = 10
        self.cooling_rate = 0.95
        self.step_size = 1.0

        self.path = [self.current_position.copy()]

    def run(self):
        fig, ax = plt.subplots(figsize=(8, 6))

        # Vẽ contour landscape
        x_lin = np.linspace(self.bounds[0], self.bounds[1], 100)
        y_lin = np.linspace(self.bounds[0], self.bounds[1], 100)
        X, Y = np.meshgrid(x_lin, y_lin)
        Z = self.cost_func(X, Y)

        ax.contour(X, Y, Z, levels=20, cmap='viridis')

        current_point = ax.scatter([], [], c='red', s=80, label='Current')
        best_point = ax.scatter([], [], c='blue', marker='*', s=200, label='Best')
        path_line, = ax.plot([], [], 'g-', alpha=0.6)

        ax.set_xlim(self.bounds[0], self.bounds[1])
        ax.set_ylim(self.bounds[0], self.bounds[1])
        ax.legend()

        def update(frame):

            current_value = self.cost_func(
                self.current_position[0],
                self.current_position[1]
            )

            if current_value < self.best_value:
                self.best_value = current_value
                self.best_position = self.current_position.copy()

            # Sinh neighbor ngẫu nhiên
            neighbor = self.current_position + np.random.uniform(
                -self.step_size,
                self.step_size,
                2
            )

            neighbor = np.clip(neighbor, self.bounds[0], self.bounds[1])

            neighbor_value = self.cost_func(neighbor[0], neighbor[1])
            delta = neighbor_value - current_value

            # Luật chấp nhận SA
            if delta < 0:
                self.current_position = neighbor
            else:
                probability = math.exp(-delta / self.temperature)
                if random.random() < probability:
                    self.current_position = neighbor

            # Giảm nhiệt độ
            self.temperature *= self.cooling_rate

            self.path.append(self.current_position.copy())

            # Update hình vẽ
            current_point.set_offsets(self.current_position.reshape(1, -1))
            best_point.set_offsets(self.best_position.reshape(1, -1))

            path_array = np.array(self.path)
            path_line.set_data(path_array[:, 0], path_array[:, 1])

            ax.set_title(
                f"Simulated Annealing - Iter {frame+1} | "
                f"T: {self.temperature:.3f} | "
                f"Best: {self.best_value:.6f}"
            )

            return current_point, best_point, path_line

        anim = FuncAnimation(
            fig,
            update,
            frames=self.n_iterations,
            interval=300,
            blit=False,
            repeat=False
        )

        plt.show()

        print("--- KẾT QUẢ SIMULATED ANNEALING ---")
        print(f"Tọa độ tìm được (x, y): ({self.best_position[0]:.10f}, {self.best_position[1]:.10f})")
        print(f"Giá trị nhỏ nhất (Cost): {self.best_value:.10f}")


# CHẠY CHƯƠNG TRÌNH
if __name__ == "__main__":
    sa = SimulatedAnnealing(cost_function, bounds=[-10, 10], n_iterations=60)
    sa.run()
