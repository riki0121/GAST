import numpy as np
from common.camera import world_to_camera

rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)

file_path = 'output/self_baseball.npz'

data = np.load(file_path)

reconstruction_data = data['reconstruction'][0]
print(f'reconstruction_data: {reconstruction_data}')

# 手首から肘までのユークリッド距離を算出する関数
def calculate_euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

# 各フレームでの手首から肘までのユークリッド距離を算出
distances_wrist_to_elbow = []
for frame in reconstruction_data:
    wrist = frame[12]  # 手首の座標
    elbow = frame[13]  # 肘の座標
    distance = calculate_euclidean_distance(wrist, elbow)
    distances_wrist_to_elbow.append(distance)

distances_knee_to_toe= []
for frame in reconstruction_data:
    knee = frame[5]  # 膝の座標
    toe = frame[6]  # 足の指の座標
    distance = calculate_euclidean_distance(knee, toe)
    distances_knee_to_toe.append(distance)


# average_distance_wrist_to_elbow = np.mean(distances_wrist_to_elbow)
# # 各フレームでの手首から肘までのユークリッド距離を表示
# for i, distance in enumerate(distances_wrist_to_elbow):
#     print(f'Frame {i}: Distance = {distance}')

# print(f'Average distance: {average_distance_wrist_to_elbow}')

average_distance_knee_to_toe = np.mean(distances_knee_to_toe)
# 各フレームでの膝から足首までのユークリッド距離を表示
for i, distance in enumerate(distances_knee_to_toe):
    print(f'Frame {i}: Distance = {distance}')

print(f'Average distance: {average_distance_knee_to_toe}')


# values_to_subtract = reconstruction_data[0][0][0]

# # 各値から指定された値を引く
# reconstruction_data -= values_to_subtract

def inverse_image_coordinates_3D(X_prime, w, h):
    assert X_prime.shape[-1] == 3

    # Reverse normalization for 3D coordinates based on the 2D normalization function
    X_prime[:, :2] = (X_prime[:, :2] + [1, h/w]) * w / 2
    
    X_prime[:, 2] = X_prime[:, 2] * 2 / w  # Z軸にもスケーリングを適用

    return X_prime






