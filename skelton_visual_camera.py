import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from common.camera import normalize_screen_coordinates, world_to_camera
rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)
ratio = 101.72144

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# npzファイルのパス
file_path = 'output/baseball.npz'

# npzファイルからデータを読み込む
data = np.load(file_path)
reconstruction_data = data['reconstruction'][0]


# npzファイルからデータを読み込む
data = np.load(file_path) 
reconstruction_data = data['reconstruction']
list_from_reconstruction_world= [reconstruction_data[i] for i in range(len(reconstruction_data))]
list_from_reconstruction_camera = []
for i in range(len(list_from_reconstruction_world)):
    sub_prediction = list_from_reconstruction_world[i]

    sub_prediction= world_to_camera(sub_prediction, R=rot, t=0)

    sub_prediction = sub_prediction * ratio

    list_from_reconstruction_camera.append(sub_prediction)

reconstruction_data = data['reconstruction'][0]
relative = list_from_reconstruction_camera[0][16]
print(f'list_from_reconstruction_camera: {(list_from_reconstruction_camera[0][0])}')
reconstruction_data_camera = list_from_reconstruction_camera[0]

output_file_path= 'output0_rot_camera.npz'
np.savez(output_file_path, reconstruction_camera=list_from_reconstruction_camera)

print(f'Data saved to {output_file_path}')

# 骨格の関節の親子関係（例: Human3.6Mの骨格）
skeleton_connections = [
    (8, 9), (9, 10), # 頭部
    (0, 1), (0, 4), (0, 7), (7, 8), (1, 14), (4, 11),  # 胴体
    (8, 11), (11, 12), (12, 13),  # 左腕
    (8, 14), (14, 15), (15, 16),  # 右腕
    (4, 5), (5, 6),  # 左脚
    (1, 2), (2, 3),  # 右脚
]

# 三次元骨格座標データをアニメーションとして可視化する関数
def update(frame, scat, lines, texts):
    x = frame[:, 0]
    y = frame[:, 1]
    z = frame[:, 2]
    scat._offsets3d = (x, y, z)
    
    for line, (i, j) in zip(lines, skeleton_connections):
        line.set_data([x[i], x[j]], [y[i], y[j]])
        line.set_3d_properties([z[i], z[j]])
    
    for i, text in enumerate(texts):
        text.set_position((x[i], y[i]))
        text.set_3d_properties(z[i], 'z')

    return scat, lines, texts

def plot_3d_skeleton_animation(skeleton_data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # ウィンドウタイトルを設定
    fig.canvas.manager.set_window_title('Camera Coordinate')

    # 初期フレームの座標をプロット
    x = skeleton_data[0][:, 0]
    y = skeleton_data[0][:, 1]
    z = skeleton_data[0][:, 2]
    scat = ax.scatter(x, y, z, c='r', marker='o')

    # 関節を線で結ぶ（例: 親子関係に基づく）
    lines = [ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], 'b')[0] for i, j in skeleton_connections]

    texts = [ax.text(x[i], y[i], z[i], str(i), color='black', fontsize=8) for i in range(len(x))]

    # 座標の範囲を設定
    # ax.set_xlim([-1, 1])
    # ax.set_ylim([-2, 0])
    # ax.set_zlim([-1, 1])
    ax.set_zlim(-40, 140)
    ax.set_zticks(np.arange(-40, 140, 20))
    ax.set_xlim(-80, 100)
    ax.set_xticks(np.arange(-80, 100, 20))
    ax.set_ylim(-180, 0)
    ax.set_yticks(np.arange(-180, 0, 20))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ani = FuncAnimation(fig, update, frames=skeleton_data, fargs=(scat, lines, texts), interval=100, blit=False)
    plt.show()

# reconstruction_dataをアニメーションとして可視化
plot_3d_skeleton_animation(reconstruction_data_camera)