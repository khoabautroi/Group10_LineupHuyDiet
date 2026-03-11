import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# --- HÀM MỤC TIÊU ---
def cost_function(position):
    x, y = position
    return x**2 + y**2

# --- LỚP NGUỒN THỨC ĂN (ĐẠI DIỆN CHO GIẢI PHÁP) ---
class FoodSource:
    def __init__(self, bounds):
        self.bounds = bounds
        # Vị trí ngẫu nhiên
        self.position = np.array([random.uniform(bounds[0], bounds[1]), 
                                  random.uniform(bounds[0], bounds[1])])
        self.cost = cost_function(self.position)
        self.trial = 0  # Đếm số lần không cải thiện (để kích hoạt Ong trinh sát)

# --- THUẬT TOÁN ABC ---
class ArtificialBeeColony:
    def __init__(self, cost_func, bounds, n_bees, n_iterations, limit):
        self.cost_func = cost_func
        self.bounds = bounds
        self.n_bees = n_bees
        self.n_employed = n_bees // 2     # 50% là ong thợ
        self.n_onlookers = n_bees // 2    # 50% là ong chờ
        self.n_iterations = n_iterations
        self.limit = limit                # Giới hạn thử thách trước khi bỏ cuộc
        
        # Khởi tạo đàn ong (nguồn thức ăn)
        self.foods = [FoodSource(bounds) for _ in range(self.n_employed)]
        
        # Lưu nguồn tốt nhất toàn cục
        self.global_best_position = np.array([0, 0])
        self.global_best_cost = float('inf')
        self.update_global_best()

    def update_global_best(self):
        # Tìm nguồn tốt nhất trong đàn hiện tại
        best_food = min(self.foods, key=lambda f: f.cost)
        if best_food.cost < self.global_best_cost:
            self.global_best_cost = best_food.cost
            self.global_best_position = best_food.position.copy()

    def calculate_fitness(self):
        # Chuyển đổi Cost (càng nhỏ càng tốt) thành Fitness (càng lớn càng tốt)
        # Để dùng cho vòng quay may mắn của Ong chờ
        fitness_list = []
        for food in self.foods:
            if food.cost >= 0:
                fitness_list.append(1.0 / (1.0 + food.cost))
            else:
                fitness_list.append(1.0 + abs(food.cost))
        return fitness_list

    def explore_neighborhood(self, current_idx):
        """Hàm tìm kiếm xung quanh (dùng chung cho Ong thợ và Ong chờ)"""
        current_food = self.foods[current_idx]
        
        # 1. Chọn ngẫu nhiên một nguồn khác để tham khảo
        neighbor_idx = current_idx
        while neighbor_idx == current_idx:
            neighbor_idx = random.randint(0, self.n_employed - 1)
        neighbor_food = self.foods[neighbor_idx]
        
        # 2. Tạo vị trí mới dựa trên công thức lai ghép
        # v_i = x_i + phi * (x_i - x_k)
        phi = random.uniform(-1, 1)
        new_position = current_food.position.copy()
        
        # Chỉ thay đổi 1 chiều ngẫu nhiên (x hoặc y) để tăng sự đa dạng
        param_idx = random.randint(0, 1) 
        new_position[param_idx] = current_food.position[param_idx] + \
                                  phi * (current_food.position[param_idx] - neighbor_food.position[param_idx])
        
        # Giữ trong biên
        new_position = np.clip(new_position, self.bounds[0], self.bounds[1])
        new_cost = self.cost_func(new_position)
        
        # 3. Đánh giá tham lam (Greedy Selection)
        if new_cost < current_food.cost:
            current_food.position = new_position
            current_food.cost = new_cost
            current_food.trial = 0  # Reset bộ đếm nếu tìm được cái mới tốt hơn
        else:
            current_food.trial += 1 # Tăng bộ đếm thất bại

    def run(self):
        # Chuẩn bị biểu đồ
        fig, ax = plt.subplots(figsize=(8, 6))
        x_lin = np.linspace(self.bounds[0], self.bounds[1], 100)
        y_lin = np.linspace(self.bounds[0], self.bounds[1], 100)
        X, Y = np.meshgrid(x_lin, y_lin)
        Z = X**2 + Y**2
        
        ax.contour(X, Y, Z, levels=20, cmap='viridis')
        scatter = ax.scatter([], [], c='orange', s=50, label='Ong (Food Sources)') # Màu cam cho ong
        best_marker = ax.scatter([], [], c='blue', marker='*', s=200, label='Global Best')
        
        ax.set_xlim(self.bounds[0], self.bounds[1])
        ax.set_ylim(self.bounds[0], self.bounds[1])
        ax.legend()
        ax.set_title("Mô phỏng ABC (Ong nhân tạo)")

        def update(frame):
            # --- PHA 1: ONG THỢ (Employed Bees) ---
            for i in range(self.n_employed):
                self.explore_neighborhood(i)

            # --- PHA 2: ONG CHỜ (Onlooker Bees) ---
            fitness_list = self.calculate_fitness()
            max_fit = max(fitness_list)
            
            # Ong chờ chọn nguồn thức ăn dựa trên xác suất (Roulette Wheel)
            for i in range(self.n_onlookers):
                # Chọn nguồn thức ăn ngẫu nhiên nhưng ưu tiên fitness cao
                selected_idx = -1
                while selected_idx == -1:
                    idx = random.randint(0, self.n_employed - 1)
                    prob = fitness_list[idx] / max_fit # Xác suất
                    if random.random() < prob:
                        selected_idx = idx
                
                # Ong chờ đi tìm kiếm xung quanh nguồn đã chọn
                self.explore_neighborhood(selected_idx)

            # --- PHA 3: ONG TRINH SÁT (Scout Bees) ---
            # Tìm nguồn đã bị khai thác cạn kiệt (trial > limit)
            for i in range(self.n_employed):
                if self.foods[i].trial > self.limit:
                    # Bỏ nguồn cũ, tạo nguồn mới ngẫu nhiên hoàn toàn
                    self.foods[i] = FoodSource(self.bounds)

            # Cập nhật Global Best
            self.update_global_best()

            # Vẽ hình
            positions = np.array([f.position for f in self.foods])
            scatter.set_offsets(positions)
            best_marker.set_offsets(self.global_best_position.reshape(1, -1))
            ax.set_title(f"Vòng lặp {frame+1} - Best Cost: {self.global_best_cost:.10f}")
            return scatter, best_marker

        anim = FuncAnimation(fig, update, frames=self.n_iterations, interval=200, blit=False, repeat=False)
        plt.show()
        
        # In kết quả cuối cùng
        print("\n" + "="*40)
        print("KẾT QUẢ TÌM KIẾM ABC")
        print("="*40)
        print(f"Tọa độ tìm được: ({self.global_best_position[0]:.10f}, {self.global_best_position[1]:.10f})")
        print(f"Giá trị tối ưu:  {self.global_best_cost:.10f}")
        print("="*40)

# --- CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    # n_bees=30 (15 thợ, 15 chờ), chạy 50 vòng, limit=10 lần thử
    abc = ArtificialBeeColony(cost_function, bounds=[-10, 10], n_bees=30, n_iterations=50, limit=5)
    abc.run()