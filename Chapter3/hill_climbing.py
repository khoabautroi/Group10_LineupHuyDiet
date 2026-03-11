import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# =============================
# HÀM MỤC TIÊU
# =============================
def cost_function(x, y):
    return x**2 + y**2


# =============================
# HILL CLIMBING
# =============================
class HillClimbing:
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
        
        self.step_size = 0.5
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
        path_line, = ax.plot([], [], 'r-', alpha=0.6)

        ax.set_xlim(self.bounds[0], self.bounds[1])
        ax.set_ylim(self.bounds[0], self.bounds[1])
        ax.legend()

        def get_neighbors(position):
            neighbors = []
            for dx in [-self.step_size, 0, self.step_size]:
                for dy in [-self.step_size, 0, self.step_size]:
                    if dx != 0 or dy != 0:
                        neighbor = position + np.array([dx, dy])
                        neighbor = np.clip(neighbor, self.bounds[0], self.bounds[1])
                        neighbors.append(neighbor)
            return neighbors

        def update(frame):
            current_value = self.cost_func(
                self.current_position[0],
                self.current_position[1]
            )

            if current_value < self.best_value:
                self.best_value = current_value
                self.best_position = self.current_position.copy()

            neighbors = get_neighbors(self.current_position)

            best_neighbor = self.current_position
            best_neighbor_value = current_value

            for n in neighbors:
                value = self.cost_func(n[0], n[1])
                if value < best_neighbor_value:
                    best_neighbor = n
                    best_neighbor_value = value

            # Nếu không cải thiện → random jump nhỏ
            if np.array_equal(best_neighbor, self.current_position):
                random_jump = np.random.uniform(-1, 1, 2)
                self.current_position += random_jump
                self.current_position = np.clip(
                    self.current_position,
                    self.bounds[0],
                    self.bounds[1]
                )
            else:
                self.current_position = best_neighbor

            self.path.append(self.current_position.copy())

            # Update hình vẽ
            current_point.set_offsets(self.current_position.reshape(1, -1))
            best_point.set_offsets(self.best_position.reshape(1, -1))

            path_array = np.array(self.path)
            path_line.set_data(path_array[:, 0], path_array[:, 1])

            ax.set_title(
                f"Hill Climbing - Iter {frame+1} | Best: {self.best_value:.6f}"
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

        print("--- KẾT QUẢ HILL CLIMBING ---")
        print(f"Tọa độ tìm được (x, y): ({self.best_position[0]:.10f}, {self.best_position[1]:.10f})")
        print(f"Giá trị nhỏ nhất (Cost): {self.best_value:.10f}")


# =============================
# CHẠY CHƯƠNG TRÌNH
# =============================
if __name__ == "__main__":
    hc = HillClimbing(cost_function, bounds=[-10, 10], n_iterations=50)
    hc.run()
