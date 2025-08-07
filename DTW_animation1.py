import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, TextBox
from common.camera import normalize_screen_coordinates, world_to_camera
from tensorflow.keras.models import load_model
from common.camera import world_to_camera
from fastdtw import fastdtw



# 回転行列とスケーリング比率
rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)
ratio = 101.72144

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# --- キーポイント別距離計算関数 ---
def pose_distance(pose1, pose2):
    """2つの3D姿勢間の距離を計算（各キーポイントの3D距離の平均）"""
    total_distance = 0.0
    for i in range(17):
        keypoint_distance = np.linalg.norm(pose1[i] - pose2[i])#２つのフォームの座標間の距離を計算
        total_distance += keypoint_distance
    return total_distance / 17.0


# npzファイルのパス
file_path = 'output/my_pitching1.npz'
data = np.load(file_path)

# 追加: 2つ目のフォーム
file_path2 = 'output/my_pitching_slow.npz'
data2 = np.load(file_path2)
reconstruction_data2 = data2['reconstruction']



# npzファイルからデータを読み込む
data = np.load(file_path) 
model = load_model('lstm_model.h5')
reconstruction_data = data['reconstruction']

key = data.files[0]
X = data[key]

if X.shape[0] == 1:
    X = X[0]


# reshape → (フレーム数, 51)
X = X.reshape(X.shape[0], 17 * 3)

# フレーム数を130に揃える
max_frames = 130
if X.shape[0] < max_frames:
    padding = np.zeros((max_frames - X.shape[0], X.shape[1]))
    X = np.concatenate([X, padding], axis=0)

elif X.shape[0] > max_frames:
    X = X[:max_frames]

# 最後にバッチ次元を追加してモデルへ
X_val = np.expand_dims(X, axis=0).astype(np.float32)

# 推論
prediction = model.predict(X_val)
predicted_class = np.argmax(prediction, axis=1)[0]

if predicted_class == 0:
    print("左投げと識別。")
    first_joint_index = 6
else:
    print("右投げと識別。")
    first_joint_index = 3

#1フレームごとにワールド座標からカメラ座標に変換。
list_from_reconstruction_world= [reconstruction_data[i] for i in range(len(reconstruction_data))]
list_from_reconstruction_camera = []

# 2つ目もカメラ座標に変換
list_from_reconstruction_world2 = [reconstruction_data2[i] for i in range(len(reconstruction_data2))]
list_from_reconstruction_camera2 = []


prediction = list_from_reconstruction_world[0]  # 1フレーム分 (17, 3)
prediction2 = list_from_reconstruction_world2[0]  # 1フレーム分 (17, 3)

list_from_reconstruction_camera = []

joint_index = first_joint_index  # 初期値は識別結果に基づく

frame1 = prediction.shape[0] #フレームの総数
frame2 = prediction2.shape[0]


# --- DTWでフレーム対応を取得（キーポイント別距離計算で）---
distance, path = fastdtw(prediction, prediction2, dist=pose_distance, radius=1)
print(f"DTW距離: {distance:.2f}")
print(f"対応フレーム数: {len(path)}")

min_frame = min(frame1,frame2)

seen = set()
filtered_path = []
for i, j in path:
    if i not in seen:
        filtered_path.append((i, j))
        seen.add(i)
    if len(filtered_path) >= min_frame:
        break

print(f"filtered_pathの長さ: {len(filtered_path)}")  # 小さい方のフレーム数と一致する


y1_3 = prediction[0, 3, 1]
y1_6 = prediction[0, 6, 1]

y2_3 = prediction2[0, 3, 1]
y2_6 = prediction2[0, 6, 1]


# 全フレームに対して処理（prediction.shape[0]はフレームの総数
for i in range(frame1):

    #y配列を定義（参照しやすくするため）
    y = {3:y1_3, 6:y1_6}

    # 30フレーム目以降はから3か6を判定、それ以前は固定のインデックスを使う
    if i > 0 :
        if first_joint_index == 3:  # 右投げならば
            if y1_3 <= y1_6:
                joint_index = 6
            else:
                joint_index = 3 
        
        elif first_joint_index == 6: #左投げならば
            if y1_3 >= y1_6:
                joint_index = 3
            else:
                joint_index = 6 
    else:
        joint_index = first_joint_index  
    

    # 基準となるキーポイントの座標を取得
    joint = prediction[i, joint_index, :]                          # i番目のフレーム、joint_index番目のキーポイントの(x,y,z)の座標である
    keypoints_world = list_from_reconstruction_world[0][i]         # shape: (17, 3),i番目のフレームの17個のキーポイント座標を取得
    sub_prediction = world_to_camera(keypoints_world, R=rot, t=0)  # ワールド座標 → カメラ座標へ変換
    sub_prediction = sub_prediction * ratio                        # スケーリング
    y_offset = sub_prediction[joint_index, 1]                      # この関節のy位置
    sub_prediction[:, 1] = sub_prediction[:, 1] - y_offset         # y=0にする
    list_from_reconstruction_camera.append(sub_prediction)         # 変換後データをリストに追加

    # 更新
    y1_3 = sub_prediction[3, 1]
    y1_6 = sub_prediction[6, 1]

# 2つ目のフォームも同様に処理
for i in range(frame2):

    #y配列を定義（参照しやすくするため）
    y = {3:y2_3, 6:y2_6}

    # 30フレーム目以降はから3か6を判定、それ以前は固定のインデックスを使う
    if i > 0 :
        if first_joint_index == 3:  # 右投げならば
            if y2_3 <= y2_6:
                joint_index = 6
            else:
                joint_index = 3 
        
        elif first_joint_index == 6: #左投げならば
            if y2_3 >= y2_6:
                joint_index = 3
            else:
                joint_index = 6 
    else:
        joint_index = first_joint_index  

    keypoints_world2 = list_from_reconstruction_world2[0][i]
    sub_prediction2 = world_to_camera(keypoints_world2, R=rot, t=0)
    sub_prediction2 = sub_prediction2 * ratio
    
    y_offset2 = sub_prediction2[joint_index, 1]  # この関節のy位置
    sub_prediction2[:, 1] = sub_prediction2[:, 1]  - y_offset2
    list_from_reconstruction_camera2.append(sub_prediction2)

            # 更新
    y2_3 = sub_prediction2[3, 1]
    y2_6 = sub_prediction2[6, 1]

# 最後に numpy 配列へ変換
list_from_reconstruction_camera = np.array(list_from_reconstruction_camera)
list_from_reconstruction_camera2 = np.array(list_from_reconstruction_camera2)

#relativeは16番目の関節の座標
relative = list_from_reconstruction_camera[0][16]

#最初のフレームの関節座標データを表示
print(f'list_from_reconstruction_camera: {(list_from_reconstruction_camera[0][0])}')

reconstruction_data_camera = list_from_reconstruction_camera
reconstruction_data_camera2 = list_from_reconstruction_camera2

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


def update_dual(frame_idx, scat1, scat2, lines1, lines2, texts1, texts2, frame_text, data1, data2):
    frame1 = data1[frame_idx]
    frame2 = data2[frame_idx % len(data2)]  # フレーム数が違う場合にも対応

    x1, y1, z1 = frame1[:, 0], frame1[:, 1], frame1[:, 2]
    x2, y2, z2 = frame2[:, 0], frame2[:, 1], frame2[:, 2]

    scat1._offsets3d = (x1, y1, z1)
    scat2._offsets3d = (x2, y2, z2)

    for line, (i, j) in zip(lines1, skeleton_connections):
        line.set_data([x1[i], x1[j]], [y1[i], y1[j]])
        line.set_3d_properties([z1[i], z1[j]])

    for line, (i, j) in zip(lines2, skeleton_connections):
        line.set_data([x2[i], x2[j]], [y2[i], y2[j]])
        line.set_3d_properties([z2[i], z2[j]])

    for i, text in enumerate(texts1):
        text.set_position((x1[i], y1[i]))
        text.set_3d_properties(z1[i], 'z')

    for i, text in enumerate(texts2):
        text.set_position((x2[i], y2[i]))
        text.set_3d_properties(z2[i], 'z')

    frame_text.set_text(f'Frame: {frame_idx}')
    return scat1, scat2, lines1, lines2, texts1, texts2, frame_text



def plot_dual_skeleton_animation(data1, data2):
    global ani
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    fig.canvas.manager.set_window_title('Dual Camera Coordinate')

    x1, y1, z1 = data1[0][:, 0], data1[0][:, 1], data1[0][:, 2]
    x2, y2, z2 = data2[0][:, 0], data2[0][:, 1], data2[0][:, 2]

    scat1 = ax.scatter(x1, y1, z1, c='blue', marker='o', label='Form 1')
    scat2 = ax.scatter(x2, y2, z2, c='red', marker='o', label='Form 2')

    lines1 = [ax.plot([x1[i], x1[j]], [y1[i], y1[j]], [z1[i], z1[j]], 'b')[0] for i, j in skeleton_connections]
    lines2 = [ax.plot([x2[i], x2[j]], [y2[i], y2[j]], [z2[i], z2[j]], 'r')[0] for i, j in skeleton_connections]

    texts1 = [ax.text(x1[i], y1[i], z1[i], str(i), color='black', fontsize=1) for i in range(len(x1))]
    texts2 = [ax.text(x2[i], y2[i], z2[i], str(i), color='darkred', fontsize=1) for i in range(len(x2))]

    all_points = np.concatenate(data1)
    x_mean = np.mean(all_points[:, 0])
    y_mean = np.mean(all_points[:, 1])
    z_mean = np.mean(all_points[:, 2])
    ax.set_xlim([x_mean - 100, x_mean + 100])
    ax.set_ylim([-200, 0])
    ax.set_zlim([z_mean - 100, z_mean + 100])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=-90, azim=-90)

    X_floor, Z_floor = np.meshgrid(np.linspace(x_mean - 100, x_mean + 100, 10),
                                   np.linspace(z_mean - 100, z_mean + 100, 10))
    Y_floor = np.zeros_like(X_floor)
    ax.plot_surface(X_floor, Y_floor, Z_floor, alpha=0.5, color='gray', edgecolor='none')

    frame_text = ax.text2D(0.85, 0.95, "", transform=ax.transAxes, fontsize=10)
    frame_idx = [0]

    ani = FuncAnimation(fig, update_dual,
                    fargs=(scat1, scat2, lines1, lines2, texts1, texts2, frame_text,
                           data1, data2),  # ← data1, data2はNumPy配列
                    frames=len(data1), interval=100, blit=False)
    plt.legend()
    plt.show()
    return ani

# reconstruction_dataをアニメーションとして可視化
plot_dual_skeleton_animation(reconstruction_data_camera, reconstruction_data_camera2)