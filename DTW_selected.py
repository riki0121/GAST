import numpy as np
import japanize_matplotlib
import matplotlib.pyplot as plt
from fastdtw import fastdtw

# --- キーポイント別距離計算関数 ---
def pose_distance(pose1, pose2):
    """2つの3D姿勢間の距離を計算（各キーポイントの3D距離の平均）"""
    total_distance = 0.0
    for i in range(17):
        keypoint_distance = np.linalg.norm(pose1[i] - pose2[i])#２つのフォームの座標間の距離を計算
        total_distance += keypoint_distance
    return total_distance / 17.0

# --- データ読み込み ---
file_path = 'output/my_pitching1.npz'
file_path2 = 'output/my_pitching_slow.npz'
data = np.load(file_path)
data2 = np.load(file_path2)

reconstruction_data = data['reconstruction']
reconstruction_data2 = data2['reconstruction']

prediction = reconstruction_data[0]   # (フレーム数1, 17, 3)
prediction2 = reconstruction_data2[0] # (フレーム数2, 17, 3)

print(f"prediction1のフレーム数: {prediction.shape[0]}")
print(f"prediction2のフレーム数: {prediction2.shape[0]}")

# --- DTW前の比較（単純なフレーム対応）---
min_frames = min(prediction.shape[0], prediction2.shape[0])
joint_diffs_before = [[] for _ in range(17)]  # DTW前の各関節の差分

for frame_idx in range(min_frames):
    for k in range(17):
        joint1 = prediction[frame_idx, k]
        joint2 = prediction2[frame_idx, k]
        diff = np.linalg.norm(joint1 - joint2)
        joint_diffs_before[k].append(diff)

# --- DTWでフレーム対応を取得（キーポイント別距離計算で）---
distance, path = fastdtw(prediction, prediction2, dist=pose_distance, radius=1)
print(f"DTW全体距離: {distance:.2f}")


frame1 = len(prediction[:,0,0])  #dataのフレーム数
frame2 = len(prediction2[:,0,0])#data2のフレーム数
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

path = filtered_path 

# --- 各関節ごとのユークリッド距離差（DTW後）を格納 ---
joint_diffs_after = [[] for _ in range(17)]  # DTW後の各関節の差分

#DTWで決まった対応フレームを使って距離を分析
for i, j in path:
    for k in range(17):
        joint1 = prediction[i, k]
        joint2 = prediction2[j, k]
        diff = np.linalg.norm(joint1 - joint2)
        joint_diffs_after[k].append(diff)
# --- グラフ描画（1つのグラフに統合） ---
fig, ax = plt.subplots(figsize=(14, 7))

# GASTのキーポイントに対応するラベル
keypoint_labels = [
    "腰", "右腰", "右膝", "右足", "左腰", "左膝", "左足",
    "背骨", "胸郭", "首", "頭", "左肩", "左肘", "左手首",
    "右肩", "右肘", "右手首",
]

# 背骨、左肩、右肩、左肘、右肘(7,11,14,12,15)
selected_joints = [12,15]  
selected_labels = ["左肘", "右肘"]
colors = ['blue', 'red']

# DTW前のグラフ（点線）
x_before = range(min_frames)
for i, joint_idx in enumerate(selected_joints):
    ax.plot(x_before, joint_diffs_before[joint_idx],
            linestyle='--', linewidth=2,
            label=f'{selected_labels[i]} (DTW前)', color=colors[i])

# DTW後のグラフ（実線）
x_after = range(len(path))
for i, joint_idx in enumerate(selected_joints):
    ax.plot(x_after, joint_diffs_after[joint_idx],
            linestyle='-', linewidth=2,
            label=f'{selected_labels[i]} (DTW後)', color=colors[i])

ax.set_xlabel("フレーム番号")
ax.set_ylabel("キーポイント間距離差")
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()