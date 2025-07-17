import numpy as np
import japanize_matplotlib
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# --- データ読み込み ---
file_path = 'output/my_pitching1.npz'
file_path2 = 'output/my_pitching_slow.npz'
data = np.load(file_path)
data2 = np.load(file_path2)

reconstruction_data = data['reconstruction']
reconstruction_data2 = data2['reconstruction']

prediction = reconstruction_data[0]   # (フレーム数1, 17, 3)
prediction2 = reconstruction_data2[0] # (フレーム数2, 17, 3)

# --- DTWでフレーム対応を取得（17関節→51次元で）
form1 = prediction.reshape(prediction.shape[0], -1)  # (T1, 51)
form2 = prediction2.reshape(prediction2.shape[0], -1)

distance, path = fastdtw(form1, form2, dist=euclidean, radius=1)
print(f"DTW距離: {distance:.2f}")
print(f"対応フレーム数: {len(path)}")

# --- 各関節ごとのユークリッド距離差（DTW後）を格納 ---
joint_diffs = [[] for _ in range(17)]  # 各関節の差分を格納するリスト

for i, j in path:
    for k in range(17):
        joint1 = prediction[i, k]
        joint2 = prediction2[j, k]
        diff = np.linalg.norm(joint1 - joint2)
        joint_diffs[k].append(diff)

# --- グラフ描画 ---
plt.figure(figsize=(14, 7))
x = range(len(path))

# GASTのキーポイントに対応するラベル
keypoint_labels = [
    "腰",          # 0
    "右腰",        # 1
    "右膝",        # 2
    "右足",        # 3
    "左腰",        # 4
    "左膝",        # 5
    "左足",        # 6
    "背骨",        # 7
    "胸郭",        # 8
    "首",          # 9
    "頭",          # 10
    "左肩",        # 11
    "左肘",        # 12
    "左手首",      # 13
    "右肩",        # 14
    "右肘",        # 15
    "右手首",      # 16
]

# グラフ描画（例）
for joint_idx in range(17):
    plt.plot(x, joint_diffs[joint_idx], label=f'{keypoint_labels[joint_idx]}')

    
plt.title("各キーポイントのユークリッド距離差（DTW後）")
plt.xlabel("対応フレーム番号（DTW）")
plt.ylabel("距離差（3D空間）")
plt.legend(loc='upper right', fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.show()