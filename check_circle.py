import numpy as np
import matplotlib.pyplot as plt# npzファイルのパス
file_path = 'output0_rot_camera.npz'

# npzファイルからデータを読み込む
data = np.load(file_path)
reconstruction_data = data['reconstruction_camera']
print(f'reconstruction_data: {reconstruction_data[0][0]}')

# 各フレームのreconstruction_data[0][i][16]を収集
points = []
for i in range(len(reconstruction_data[0])):
    point = reconstruction_data[0][i][16]  # reconstruction_data[0][i][16]の座標
    points.append(point)

print(f'points: {points}')
# pointsをnumpy配列に変換
points = np.array(points)
print(f'points: {points}')

# プロット
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 各フレームの点をプロット
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o')

ax.set_zlim(-40, 140)
ax.set_zticks(np.arange(-40, 140, 20))
ax.set_xlim(-80, 100)
ax.set_xticks(np.arange(-80, 100, 20))
ax.set_ylim(-180, 0)
ax.set_yticks(np.arange(-180, 0, 20))

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()