import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, TextBox
from common.camera import normalize_screen_coordinates, world_to_camera
from tensorflow.keras.models import load_model
from common.camera import world_to_camera
from matplotlib.widgets import Button

# 回転行列とスケーリング比率
rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)
ratio = 101.72144

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# npzファイルのパス
file_path = 'output/sample2.npz'
file_data = np.load(file_path)

# npzファイルからデータを読み込む
data = np.load(file_path) 
model = load_model('lstm_model.h5')
reconstruction_data = data['reconstruction']

key = file_data.files[0]
X = file_data[key]

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
print("list_from_reconstruction_world:", list_from_reconstruction_world[0].shape)

prediction = list_from_reconstruction_world[0]  # 1フレーム分 (17, 3)
print("prediction.shape:",prediction.shape)


list_from_reconstruction_camera = []

joint_index = first_joint_index  # 初期値は識別結果に基づく

Total_frames = prediction.shape[0] #フレームの総数


y_3 = prediction[0,3,1]
y_6 = prediction[0,6,1]

prev_joint_index = first_joint_index
now_positions = None  # 各関節の現在位置
now_anchor = first_joint_index

for i in range(Total_frames):
    # アンカー関節の判定
    if i > 0:
        if first_joint_index == 3:
            joint_index = 6 if y_3 < y_6 else 3
        else:  # first_joint_index == 6
            joint_index = 3 if y_3 > y_6 else 6
    else:
        joint_index = first_joint_index
    
    # 現在の座標（1フレーム分）を取得
    keypoints_world = list_from_reconstruction_world[0][i]  # shape: (17, 3)

    #ループ初回のみ
    if now_positions is None:
        #最初の固定場所は（0,0,0）
        now_positions = keypoints_world - keypoints_world[joint_index] 
        now_anchor = joint_index
  
    # ループ2回目以降
    else:

        now_anchor = joint_index

        #now_positions[joint_index,1] = -10 
        print(f"frame:{i}, anchor: {now_anchor}")
        print(f"frame:{i}, new_positions[3]: {now_positions[3]}, new_positions[6]: {now_positions[6]}")


        # 1つ前のフレームの座標を取得
        prev_frame = list_from_reconstruction_world[0][i-1]
        # フレーム間の変化を計算
        diff = keypoints_world - prev_frame #shape: (17, 3)
        #現在のアンカー（固定座標）の変化量
        anchor_diff = diff[now_anchor] 
        print(f"frame:{i}, anchor_diff: {anchor_diff}")

        #現在の位置にアンカーの変化量を引く
        now_positions = (now_positions + diff) - anchor_diff

        #

    # カメラ座標変換
    sub_prediction= world_to_camera(now_positions, R=rot, t=0) * ratio
    list_from_reconstruction_camera.append(sub_prediction)
    
    # 次回判定用
    y_3 = sub_prediction[3, 1]
    y_6 = sub_prediction[6, 1]

# 最後に numpy 配列へ変換
list_from_reconstruction_camera = np.array(list_from_reconstruction_camera)


#relativeは16番目の関節の座標
relative = list_from_reconstruction_camera[0][16]

#最初のフレームの関節座標データを表示
print(f'list_from_reconstruction_camera: {(list_from_reconstruction_camera[0][0])}')

reconstruction_data_camera = list_from_reconstruction_camera

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
def update(frame, scat, lines, texts, frame_text, frame_idx):
    x = frame[:, 0] #全関節の座標
    y = frame[:, 1]
    z = frame[:, 2]
    # scatは関節の位置を表す散布図
    scat._offsets3d = (x, y, z)
    
    for line, (i, j) in zip(lines, skeleton_connections):
        line.set_data([x[i], x[j]], [y[i], y[j]])
        line.set_3d_properties([z[i], z[j]])
    
    for i, text in enumerate(texts):
        text.set_position((x[i], y[i]))
        text.set_3d_properties(z[i], 'z')


    frame_text.set_text(f'Frame: {frame_idx[0]}')
    frame_idx[0] += 1
    if frame_idx[0] >= len(reconstruction_data_camera):  # 最後のフレームまで行ったら
        frame_idx[0] = 0  # フレーム番号をリセット

    return scat, lines, texts, frame_text

fig = plt.figure(figsize=(10, 10))  # figsizeを指定してウィンドウサイズを調整
ax = fig.add_subplot(111, projection='3d')



def plot_3d_skeleton_animation(skeleton_data):

    global ani
    # ウィンドウタイトルを設定
    fig.canvas.manager.set_window_title('Camera Coordinate')

    # 最初のフレームの座標をプロット
    #skelton_data[0]は1フレームの関節座標データ

    x = skeleton_data[0][:, 0] #[:,0]全関節のx座標
    y = skeleton_data[0][:, 1]
    z = skeleton_data[0][:, 2]

    #各関節の色をそれぞれプロット
    colors = "red"

    scat = ax.scatter(x, y, z, c=colors, marker='o')

    # 関節を線で結ぶ（例: 親子関係に基づく）
    lines = [ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], 'b')[0] for i, j in skeleton_connections]

    texts = [ax.text(x[i], y[i], z[i], str(i), color='black', fontsize=6) for i in range(len(x))]

   # 全フレームの全キーポイントの座標から平均を計算
    all_points = np.concatenate(skeleton_data)  # 全フレーム・全関節を結合
    
    # 各軸の平均を計算
    x_mean = np.mean(all_points[:, 0])
    y_mean = np.mean(all_points[:, 1])
    z_mean = np.mean(all_points[:, 2])
    
    # 表示範囲を平均±100に設定
    ax.set_xlim([x_mean - 100, x_mean + 100])
    ax.set_ylim([y_mean - 100, y_mean + 100])
    ax.set_zlim([z_mean - 100, z_mean + 100])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.view_init(elev=-60, azim=-90)


   # フレーム番号表示のテキスト追加（右上）
    frame_text = ax.text2D(0.85, 0.95, "", transform=ax.transAxes, fontsize=10)
    frame_idx = [0]  # 参照渡し用のミュータブルなリスト

    ani = FuncAnimation(fig, update, frames=skeleton_data, fargs=(scat, lines, texts, frame_text, frame_idx), interval=100, blit=False)

    plt.show()
    return ani

# reconstruction_dataをアニメーションとして可視化
plot_3d_skeleton_animation(reconstruction_data_camera)