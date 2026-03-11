import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

def cost_function(position):
    x, y = position
    # Công thức Six-Hump Camel
    # Chỉ dùng phép nhân và lũy thừa, không có sin/cos
    term1 = (4 - 2.1 * x**2 + (x**4) / 3) * x**2
    term2 = x * y
    term3 = (-4 + 4 * y**2) * y**2
    return term1 + term2 + term3

class FireflyAlgorithm:
    def __init__(self, cost_func, bounds, n_fireflies, n_iterations):
        self.cost_func = cost_func
        self.bounds = bounds
        self.n = n_fireflies
        self.n_iterations = n_iterations
        
        self.alpha = 0.5      
        self.beta0 = 1.0      
        self.gamma = 1.0      
        self.alpha_damp = 0.97 

        # Khởi tạo đom đóm ngẫu nhiên
        # Lưu ý: Hàm này có biên x và y khác nhau, nhưng để đơn giản ta khởi tạo trong hình vuông
        self.fireflies = np.random.uniform(bounds[0], bounds[1], (self.n, 2))
        self.intensity = np.zeros(self.n)
        
        self.update_intensity()
        self.global_best_pos = np.zeros(2)
        self.global_best_val = float('inf')
        self.check_best()

    def update_intensity(self):
        for i in range(self.n):
            self.intensity[i] = self.cost_func(self.fireflies[i])

    def check_best(self):
        min_idx = np.argmin(self.intensity)
        if self.intensity[min_idx] < self.global_best_val:
            self.global_best_val = self.intensity[min_idx]
            self.global_best_pos = self.fireflies[min_idx].copy()

    def move_fireflies(self):
        for i in range(self.n):
            for j in range(self.n):
                if self.intensity[j] < self.intensity[i]:
                    r = np.linalg.norm(self.fireflies[i] - self.fireflies[j])
                    beta = self.beta0 * np.exp(-self.gamma * r**2)
                    noise = self.alpha * (np.random.rand(2) - 0.5)
                    move_vector = beta * (self.fireflies[j] - self.fireflies[i])
                    self.fireflies[i] += move_vector + noise
            
            # Giữ đom đóm trong biên
            self.fireflies[i] = np.clip(self.fireflies[i], self.bounds[0], self.bounds[1])

        self.alpha *= self.alpha_damp
        self.update_intensity()
        self.check_best()

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
        
        # Vẽ đường đồng mức
        ax.contour(X, Y, Z, levels=40, cmap='plasma') # Dùng màu plasma cho lạ mắt
        scatter = ax.scatter([], [], c='yellow', edgecolors='black', s=60, label='Đom đóm')
        best_marker = ax.scatter([], [], c='cyan', marker='*', s=200, label='Best Found')
        
        ax.set_xlim(self.bounds[0], self.bounds[1])
        ax.set_ylim(self.bounds[0], self.bounds[1])
        ax.legend()
        ax.set_title("FA ")

        def update(frame):
            self.move_fireflies()
            
            if frame == self.n_iterations - 1:
                print("\n" + "="*40)
                print("KẾT QUẢ CUỐI CÙNG")
                print(f"Tọa độ: ({self.global_best_pos[0]:.5f}, {self.global_best_pos[1]:.5f})")
                print(f"Giá trị Cost: {self.global_best_val:.5f}")
                print("="*40)

            scatter.set_offsets(self.fireflies)
            best_marker.set_offsets(self.global_best_pos.reshape(1, -1))
            ax.set_title(f"Vòng {frame+1} - Cost: {self.global_best_val:.5f}")
            return scatter, best_marker

        anim = FuncAnimation(fig, update, frames=self.n_iterations, interval=100, blit=False, repeat=False)
        plt.show()

if __name__ == "__main__":
    # Hàm Camel nằm gọn trong khoảng [-2, 2] nên ta thu hẹp biên lại để nhìn rõ hơn
    fa = FireflyAlgorithm(
        cost_func=cost_function,
        bounds=[-2, 2], 
        n_fireflies=30,
        n_iterations=50
    )
    fa.run()