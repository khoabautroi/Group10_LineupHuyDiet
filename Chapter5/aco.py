import numpy as np
import random

class AntColony:
    def __init__(self, distances, n_ants, n_best, n_iterations, decay, alpha=1, beta=1):
        """
        Args:
            distances (2D numpy array): Ma trận khoảng cách giữa các thành phố
            n_ants (int): Số lượng kiến mỗi lần thả
            n_best (int): Số lượng kiến tốt nhất được chọn để cập nhật pheromone (ưu tú)
            n_iterations (int): Số vòng lặp chạy thuật toán
            decay (float): Hệ số bay hơi pheromone (0 < decay < 1)
            alpha (int): Hệ số quan trọng của pheromone (mùi hương)
            beta (int): Hệ số quan trọng của heuristic (khoảng cách)
        """
        self.distances  = distances
        self.pheromone  = np.ones(self.distances.shape) / len(distances)
        self.all_inds   = range(len(distances))
        self.n_ants     = n_ants
        self.n_best     = n_best
        self.n_iterations = n_iterations
        self.decay      = decay
        self.alpha      = alpha
        self.beta       = beta

    def run(self):
        shortest_path = None
        all_time_shortest_path = ("placeholder", np.inf)
        
        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths()
            self.spread_pheronome(all_paths, self.n_best, shortest_path=shortest_path)
            
            # Tìm đường ngắn nhất trong thế hệ này
            shortest_path = min(all_paths, key=lambda x: x[1])
            
            # Cập nhật đường ngắn nhất toàn cục
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path
            
            # Cập nhật bay hơi pheromone sau mỗi vòng lặp
            self.pheromone * (1 - self.decay)
            
            print(f"Vòng lặp {i+1}: Chi phí thấp nhất = {all_time_shortest_path[1]:.2f}")

        return all_time_shortest_path

    def spread_pheronome(self, all_paths, n_best, shortest_path):
        # Sắp xếp các đường đi từ tốt nhất đến xấu nhất
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        
        # Chỉ lấy n_best đường đi tốt nhất để rải pheromone (chiến lược Elite)
        for path, dist in sorted_paths[:n_best]:
            for move in path:
                # Cộng thêm pheromone: càng ngắn thì cộng càng nhiều (1/dist)
                self.pheromone[move] += 1.0 / self.distances[move]

    def gen_path_dist(self, path):
        total_dist = 0
        for ele in path:
            total_dist += self.distances[ele]
        return total_dist

    def gen_all_paths(self):
        all_paths = []
        for i in range(self.n_ants):
            path = self.gen_path(0) # Giả sử luôn bắt đầu từ thành phố 0
            all_paths.append((path, self.gen_path_dist(path)))
        return all_paths

    def gen_path(self, start_node):
        path = []
        visited = set()
        visited.add(start_node)
        prev = start_node
        
        for i in range(len(self.distances) - 1):
            move = self.pick_move(self.pheromone[prev], self.distances[prev], visited)
            path.append((prev, move))
            prev = move
            visited.add(move)
        
        # Quay về điểm xuất phát để khép kín vòng
        path.append((prev, start_node)) 
        return path

    def pick_move(self, pheromone, dist, visited):
        # Sao chép pheromone để tránh sửa đổi bản gốc
        pheromone = np.copy(pheromone)
        
        # Gán xác suất = 0 cho các thành phố đã đi qua
        pheromone[list(visited)] = 0

        # Tính toán độ hấp dẫn: Pheromone^alpha * (1/Distance)^beta
        # Lưu ý: Tránh chia cho 0 bằng cách thêm giá trị nhỏ hoặc xử lý inf
        with np.errstate(divide='ignore'):
            heuristic = 1.0 / dist
            heuristic[dist == 0] = 0 # Khoảng cách đến chính nó là 0
        
        row = pheromone ** self.alpha * (heuristic ** self.beta)

        # Chuẩn hóa thành xác suất (tổng = 1)
        norm_row = row / row.sum()
        
        # Chọn ngẫu nhiên dựa trên xác suất (Roulette Wheel Selection)
        move = np.random.choice(self.all_inds, 1, p=norm_row)[0]
        return move
if __name__ == '__main__':
    # Tạo dữ liệu giả lập: 5 thành phố với tọa độ ngẫu nhiên
    num_cities = 10
    cities = np.random.rand(num_cities, 2) * 100 # Tọa độ x, y từ 0 đến 100
    
    # Tính ma trận khoảng cách (Euclidean distance)
    distances = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                distances[i][j] = np.linalg.norm(cities[i] - cities[j])
            else:
                distances[i][j] = np.inf # Khoảng cách đến chính nó là vô cực để tránh chọn lại

    print("Đang khởi tạo bầy kiến...")
    
    # Cấu hình thuật toán
    ant_colony = AntColony(
        distances, 
        n_ants=5,           # Số lượng kiến mỗi vòng
        n_best=2,           # Số lượng kiến ưu tú được rải pheromone
        n_iterations=20,    # Số vòng lặp
        decay=0.1,          # Tốc độ bay hơi (0.1 = 10% mỗi vòng)
        alpha=1,            # Tầm quan trọng của Pheromone
        beta=2              # Tầm quan trọng của Khoảng cách (Heuristic)
    )

    shortest_path = ant_colony.run()
    
    print("\n--- KẾT QUẢ ---")
    print(f"Đường đi ngắn nhất tìm được có độ dài: {shortest_path[1]:.2f}")
    
    # Hiển thị thứ tự các thành phố đi qua
    path_indices = [int(edge[0]) for edge in shortest_path[0]] + [shortest_path[0][-1][1]]
    print(f"Lộ trình: {path_indices}")