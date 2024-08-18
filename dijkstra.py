import numpy as np
import open3d as o3d
import time
import heapq

# 텍스트 파일에서 점들의 좌표를 읽기
def load_points(file_path):
    return np.loadtxt(file_path, delimiter=' ')

# Open3D PointCloud 객체 생성
def create_point_cloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

# Voxel 그리드 생성
def create_voxel_grid(points, voxel_size, buffer=3):
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    dimensions = np.ceil((max_bound - min_bound) / voxel_size).astype(int)
    voxel_grid = np.zeros(dimensions, dtype=bool)

    for point in points:
        voxel_index = np.floor((point - min_bound) / voxel_size).astype(int)
        voxel_grid[tuple(voxel_index)] = True

    # 주어진 좌표 주변의 버퍼 비워주는 함수
    def clear_buffer_around(voxel_grid, point, buffer):
        idx = np.floor((point - min_bound) / voxel_size).astype(int)
        for x in range(max(0, idx[0] - buffer), min(voxel_grid.shape[0], idx[0] + buffer + 1)):
            for y in range(max(0, idx[1] - buffer), min(voxel_grid.shape[1], idx[1] + buffer + 1)):
                for z in range(max(0, idx[2] - buffer), min(voxel_grid.shape[2], idx[2] + buffer + 1)):
                    voxel_grid[x, y, z] = False

    return voxel_grid, min_bound, voxel_size, clear_buffer_around

# Dijkstra 알고리즘
def dijkstra(voxel_grid, start, goal):

    # 6방향 이웃 노드들 반환 함수
    def get_neighbors(node):
        neighbors = [
            (node[0] + 1, node[1], node[2]),
            (node[0] - 1, node[1], node[2]),
            (node[0], node[1] + 1, node[2]),
            (node[0], node[1] - 1, node[2]),
            (node[0], node[1], node[2] + 1),
            (node[0], node[1], node[2] - 1),
        ]
        # 그리드 내에 있는 유효한 이웃만 반환
        return [neighbor for neighbor in neighbors if 0 <= neighbor[0] < voxel_grid.shape[0]
                and 0 <= neighbor[1] < voxel_grid.shape[1] and 0 <= neighbor[2] < voxel_grid.shape[2]]
   
    # 경로 재구성 함수
    def reconstruct_path(came_from, current):
        path = []
        while current is not None:
            path.append(current)
            current = came_from[current]
        return path[::-1]

    start_time = time.time()
    start = tuple(start)
    goal = tuple(goal)
    frontier = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}

    visited_nodes = set()

    while frontier:
        current_cost, current = heapq.heappop(frontier)

        if current == goal:
            end_time = time.time()
            print(f"Dijkstra algorithm executed in {end_time - start_time:.2f} seconds.")
            return reconstruct_path(came_from, current), visited_nodes

        for neighbor in get_neighbors(current):
            if voxel_grid[neighbor] == 0:  # Check if the neighbor is navigable
                new_cost = current_cost + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current
                    visited_nodes.add(neighbor)

    return None, visited_nodes

# 경로 시각화
def visualize_path(pcd, path, min_bound, voxel_size, visited_nodes):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    # 탐색된 노드 샘플링 (너무 많은 노드가 시각화되는 것 방지)
    sample_rate = max(1, len(visited_nodes) // 1000)  # 샘플링 비율 조정
    sampled_visited_nodes = list(visited_nodes)[::sample_rate]

    for node in sampled_visited_nodes:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sphere.translate(np.array(node) * voxel_size + min_bound)
        sphere.paint_uniform_color([0, 1, 0])  # 탐색된 노드는 초록색
        vis.add_geometry(sphere)

    for i in range(len(path) - 1):
        line_set = o3d.geometry.LineSet()
        points = [
            np.array(path[i]) * voxel_size + min_bound,
            np.array(path[i + 1]) * voxel_size + min_bound,
        ]
        lines = [[0, 1]]
        colors = [[1, 0, 0]]  # 경로는 빨간색
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(line_set)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])  # 배경색 설정 (검정색)
    opt.point_size = 1.0  # 점 크기 설정
    vis.run()
    vis.destroy_window()

def main():
    points = load_points('GlobalMap.txt')
    pcd = create_point_cloud(points)

    voxel_size = 0.3
    voxel_grid, min_bound, voxel_size, clear_buffer_around = create_voxel_grid(points, voxel_size)

    start = np.array([5, 9, 1])  # 시작점 (예시)
    goal = np.array([-2, -30, 0])  # 목표점 (예시)

    # 시작점과 목표점 주변의 포인트 비워줌
    buffer = 3
    clear_buffer_around(voxel_grid, start, buffer)
    clear_buffer_around(voxel_grid, goal, buffer)

    # 시작점과 목표점이 그리드 내에 있는지 확인
    start_index = np.floor((start - min_bound) / voxel_size).astype(int)
    goal_index = np.floor((goal - min_bound) / voxel_size).astype(int)

    if (0 <= start_index[0] < voxel_grid.shape[0] and
        0 <= start_index[1] < voxel_grid.shape[1] and
        0 <= start_index[2] < voxel_grid.shape[2] and
        0 <= goal_index[0] < voxel_grid.shape[0] and
        0 <= goal_index[1] < voxel_grid.shape[1] and
        0 <= goal_index[2] < voxel_grid.shape[2]):
        
        path, visited_nodes = dijkstra(voxel_grid, start_index, goal_index)
        if path:
            print("Path found.")
            visualize_path(pcd, path, min_bound, voxel_size, visited_nodes)
        else:
            print("No path found.")
    else:
        print("Start or goal is out of grid bounds.")


if __name__ == "__main__":
    main()
