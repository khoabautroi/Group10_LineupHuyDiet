import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# --- CẤU HÌNH BÀI TOÁN ---
def cost_function(x, y):
    """Hàm mục tiêu: f(x,y) = x^2 + y^2"""
    return x**2 + y**2

class Particle:
    def __init__(self, bounds):
        # Khởi tạo vị trí ngẫu nhiên trong vùng giới hạn
        self.position = np.array([random.uniform(bounds[0], bounds[1]), 
                                  random.uniform(bounds[0], bounds[1])])
        
        # Khởi tạo vận tốc ngẫu nhiên
        self.velocity = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
        
        # P_best: Vị trí tốt nhất của riêng hạt này
        self.p_best_position = self.position.copy()
        self.p_best_value = float('inf') # Ban đầu coi như vô cực
        
    def update_velocity(self, g_best_position, w, c1, c2):
        """
        Cập nhật vận tốc dựa trên 3 yếu tố:
        1. Quán tính (w)
        2. Kinh nghiệm cá nhân (c1)
        3. Kinh nghiệm bầy đàn (c2)
        """
        r1 = random.random()
        r2 = random.random()
        
        inertia = w * self.velocity
        cognitive = c1 * r1 * (self.p_best_position - self.position)
        social = c2 * r2 * (g_best_position - self.position)
        
        self.velocity = inertia + cognitive + social

    def update_position(self, bounds):
        """Cập nhật vị trí mới dựa trên vận tốc"""
        self.position += self.velocity
        
        # Giữ hạt không bay ra khỏi vùng tìm kiếm (Clamping)
        self.position = np.clip(self.position, bounds[0], bounds[1])

# --- THUẬT TOÁN PSO ---
class PSO:
    def __init__(self, cost_func, bounds, n_particles, n_iterations):
        self.cost_func = cost_func
        self.bounds = bounds
        self.n_iterations = n_iterations
        self.global_best_position = np.array([0, 0])
        self.global_best_value = float('inf')
        
        # Tạo đàn hạt
        self.particles = []
        for _ in range(n_particles):
            self.particles.append(Particle(bounds))
            
    def run(self):
        # Các tham số điều khiển (Hyperparameters)
        W = 0.5   # Trọng số quán tính (Inertia)
        C1 = 1.5  # Hệ số cá nhân (Cognitive)
        C2 = 1.5  # Hệ số xã hội (Social)
        
        # Chuẩn bị biểu đồ
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Tạo lưới nền để vẽ đường đồng mức (Contour plot)
        x_lin = np.linspace(self.bounds[0], self.bounds[1], 100)
        y_lin = np.linspace(self.bounds[0], self.bounds[1], 100)
        X, Y = np.meshgrid(x_lin, y_lin)
        Z = self.cost_func(X, Y)
        
        ax.contour(X, Y, Z, levels=20, cmap='viridis') # Vẽ nền địa hình
        scatter = ax.scatter([], [], c='red', s=50, label='Hạt (Particles)') # Các điểm đỏ
        best_marker = ax.scatter([], [], c='blue', marker='*', s=200, label='Global Best') # Điểm tốt nhất
        
        ax.set_xlim(self.bounds[0], self.bounds[1])
        ax.set_ylim(self.bounds[0], self.bounds[1])
        ax.legend()
        ax.set_title("Mô phỏng PSO - Tìm đáy của cái bát")

        # Hàm cập nhật cho Animation
        def update(frame):
            # Trong mỗi khung hình, cập nhật toàn bộ đàn hạt
            positions = []
            
            for particle in self.particles:
                # 1. Đánh giá vị trí hiện tại
                fitness = self.cost_func(particle.position[0], particle.position[1])
                
                # 2. Cập nhật P_best (Cá nhân)
                if fitness < particle.p_best_value:
                    particle.p_best_value = fitness
                    particle.p_best_position = particle.position.copy()
                
                # 3. Cập nhật G_best (Toàn cục)
                if fitness < self.global_best_value:
                    self.global_best_value = fitness
                    self.global_best_position = particle.position.copy()
            
            # 4. Di chuyển hạt
            for particle in self.particles:
                particle.update_velocity(self.global_best_position, W, C1, C2)
                particle.update_position(self.bounds)
                positions.append(particle.position)
            
            # Cập nhật hình ảnh
            positions = np.array(positions)
            scatter.set_offsets(positions)
            best_marker.set_offsets(self.global_best_position.reshape(1, -1))
            ax.set_title(f"Vòng lặp {frame+1} - Best Value: {self.global_best_value:.5f}")
            return scatter, best_marker

        # Chạy Animation
        anim = FuncAnimation(fig, update, frames=self.n_iterations, interval=200, blit=False, repeat=False)
        plt.show()
        
        print("--- KẾT QUẢ TÌM KIẾM PSO ---")
        
        # Lấy tọa độ x và y từ vị trí tốt nhất
        best_x = self.global_best_position[0]
        best_y = self.global_best_position[1]
        
        # In ra với định dạng 10 chữ số thập phân (.10f)
        print(f"Tọa độ tìm được (x, y): ({best_x:.10f}, {best_y:.10f})")
        print(f"Độ sâu tại đáy (Cost):  {self.global_best_value:.10f}")

# --- CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    # Tìm cực tiểu trong vùng từ -10 đến 10
    pso = PSO(cost_function, bounds=[-10, 10], n_particles=30, n_iterations=50)
    pso.run()