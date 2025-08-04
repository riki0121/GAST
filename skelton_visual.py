import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# npzファイルのパス
file_path = 'output/self_baseball.npz'

# npzファイルからデータを読み込む
data = np.load(file_path)
reconstruction_data = data['reconstruction'][0]
print(reconstruction_data[0])
print(f'data_type: {type(reconstruction_data)}')

# 骨格の関節の親子関係（例: Human3.6Mの骨格）
skeleton_connections = [
    (8, 9), (9, 10), # 頭部
    (0, 1), (0, 4), (0, 7), (7, 8),  # 胴体
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
    fig.canvas.manager.set_window_title('World Coordinate')

    # 初期フレームの座標をプロット
    x = skeleton_data[0][:, 0]
    y = skeleton_data[0][:, 1]
    z = skeleton_data[0][:, 2]
    scat = ax.scatter(x, y, z, c='r', marker='o')

    # 関節を線で結ぶ（例: 親子関係に基づく）
    lines = [ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], 'b')[0] for i, j in skeleton_connections]

    texts = [ax.text(x[i], y[i], z[i], str(i), color='black', fontsize=8) for i in range(len(x))]

    # 座標の範囲を設定
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 2])


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ani = FuncAnimation(fig, update, frames=skeleton_data, fargs=(scat, lines, texts), interval=100, blit=False)
    plt.show()

# reconstruction_dataをアニメーションとして可視化
plot_3d_skeleton_animation(reconstruction_data)