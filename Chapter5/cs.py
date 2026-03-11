import numpy as np
import math
from scipy.special import gamma
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
def cost_function(position):
    x, y = position
    # Công thức: (4 - 2.1x^2 + x^4/3)x^2 + xy + (-4 + 4y^2)y^2
    term1 = (4 - 2.1 * x**2 + (x**4) / 3) * x**2
    term2 = x * y
    term3 = (-4 + 4 * y**2) * y**2
    return term1 + term2 + term3

class CuckooSearch:
    def __init__(self, cost_func, bounds, n_nests, n_iterations, pa=0.25):
        self.cost_func = cost_func
        self.bounds = bounds
        self.n = n_nests
        self.n_iterations = n_iterations
        self.pa = pa 
        self.beta = 1.5 

        # Khởi tạo tổ
        self.nests = np.random.uniform(bounds[0], bounds[1], (self.n, 2))
        self.fitness = np.zeros(self.n)
        self.best_nest = None
        self.best_fitness = float('inf')
        
        self.evaluate_nests()

    def evaluate_nests(self):
        for i in range(self.n):
            self.fitness[i] = self.cost_func(self.nests[i])
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_nest = self.nests[i].copy()

    def get_cuckoos(self):
        # Tạo bước nhảy Lévy (Lévy Flights)
        sigma_u = (gamma(1 + self.beta) * np.sin(np.pi * self.beta / 2) / 
                   (gamma((1 + self.beta) / 2) * self.beta * 2 ** ((self.beta - 1) / 2))) ** (1 / self.beta)
        sigma_v = 1

        for i in range(self.n):
            s = self.nests[i].copy()
            u = np.random.normal(0, sigma_u, 2)
            v = np.random.normal(0, sigma_v, 2)
            step = u / (np.abs(v) ** (1 / self.beta))
            
            # Cập nhật vị trí
            step_size = 0.01 * step * (s - self.best_nest)
            s_new = s + step_size * np.random.randn(2)
            s_new = np.clip(s_new, self.bounds[0], self.bounds[1])
            
            f_new = self.cost_func(s_new)
            if f_new < self.fitness[i]:
                self.nests[i] = s_new
                self.fitness[i] = f_new

    def empty_nests(self):
        # Chim chủ phát hiện trứng lạ (Discovery)
        new_nests = self.nests.copy()
        discovery_mat = np.random.random((self.n, 2)) < self.pa
        
        rand_idx1 = np.random.permutation(self.n)
        rand_idx2 = np.random.permutation(self.n)
        
        step_size = np.random.random() * (self.nests[rand_idx1] - self.nests[rand_idx2])
        new_nests = self.nests + step_size * discovery_mat
        new_nests = np.clip(new_nests, self.bounds[0], self.bounds[1])
        
        for i in range(self.n):
            f_new = self.cost_func(new_nests[i])
            if f_new < self.fitness[i]:
                self.nests[i] = new_nests[i]
                self.fitness[i] = f_new

    def run(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Vẽ địa hình Camel
        x_lin = np.linspace(self.bounds[0], self.bounds[1], 100)
        y_lin = np.linspace(self.bounds[0], self.bounds[1], 100)
        X, Y = np.meshgrid(x_lin, y_lin)
        
        # Tính Z theo công thức Camel
        term1 = (4 - 2.1 * X**2 + (X**4) / 3) * X**2
        term2 = X * Y
        term3 = (-4 + 4 * Y**2) * Y**2
        Z = term1 + term2 + term3
        
        ax.contour(X, Y, Z, levels=40, cmap='plasma') 
        
        scatter = ax.scatter([], [], c='lime', edgecolors='black', s=50, label='Cúc cu')
        best_marker = ax.scatter([], [], c='red', marker='+', s=150, linewidth=2, label='Global Best')
        
        ax.set_xlim(self.bounds[0], self.bounds[1])
        ax.set_ylim(self.bounds[0], self.bounds[1])
        ax.legend()
        ax.set_title("Cuckoo Search")

        def update(frame):
            self.get_cuckoos()
            self.empty_nests()
            
            # Cập nhật Best
            current_best_idx = np.argmin(self.fitness)
            if self.fitness[current_best_idx] < self.best_fitness:
                self.best_fitness = self.fitness[current_best_idx]
                self.best_nest = self.nests[current_best_idx].copy()

            if frame == self.n_iterations - 1:
                print("\n" + "="*40)
                print("KẾT QUẢ CUỐI CÙNG (Six-Hump Camel)")
                print(f"Mục tiêu: -1.0316")
                print(f"Tìm được: ({self.best_nest[0]:.5f}, {self.best_nest[1]:.5f})")
                print(f"Cost:     {self.best_fitness:.5f}")
                print("="*40)

            scatter.set_offsets(self.nests)
            best_marker.set_offsets(self.best_nest.reshape(1, -1))
            ax.set_title(f"Vòng {frame+1} - Best Cost: {self.best_fitness:.5f}")
            return scatter, best_marker

        anim = FuncAnimation(fig, update, frames=self.n_iterations, interval=100, blit=False, repeat=False)
        plt.show()

if __name__ == "__main__":
    # Dùng bounds [-2, 2] để tập trung vào khu vực có cực trị
    cs = CuckooSearch(
        cost_func=cost_function,
        bounds=[-2, 2], 
        n_nests=25,
        n_iterations=50,
        pa=0.25
    )
    cs.run()