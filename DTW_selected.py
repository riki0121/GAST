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
print(f"DTW対応フレーム数: {len(path)}")
print(f"フレーム数の差: prediction1={prediction.shape[0]}, prediction2={prediction2.shape[0]}")

# --- 各関節ごとのユークリッド距離差（DTW後）を格納 ---
joint_diffs_after = [[] for _ in range(17)]  # DTW後の各関節の差分

#DTWで決まった対応フレームを使って距離を分析
for i, j in path:
    for k in range(17):
        joint1 = prediction[i, k]
        joint2 = prediction2[j, k]
        diff = np.linalg.norm(joint1 - joint2)
        joint_diffs_after[k].append(diff)

# --- グラフ描画 ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

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

# 抽出したいキーポイントのインデックス
selected_joints = [7, 11, 14, 12, 15]  #背骨、左肩、右肩、左肘、右肘
selected_labels = ["背骨", "左肩", "右肩", "左肘", "右肘"]

# DTW前のグラフ（左）
x_before = range(min_frames)
for i, joint_idx in enumerate(selected_joints):
    ax1.plot(x_before, joint_diffs_before[joint_idx], label=f'{selected_labels[i]}', linewidth=2)

ax1.set_title(f"主要キーポイントの3Dユークリッド距離差（DTW前）\nフレーム数: {min_frames}", fontsize=14)
ax1.set_xlabel("フレーム番号")
ax1.set_ylabel("キーポイント間距離差")
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)

# DTW後のグラフ（右）
x_after = range(len(path))
for i, joint_idx in enumerate(selected_joints):
    ax2.plot(x_after, joint_diffs_after[joint_idx], label=f'{selected_labels[i]}', linewidth=2)

ax2.set_title(f"主要キーポイントの3Dユークリッド距離差（DTW後）\n対応フレーム数: {len(path)}", fontsize=14)
ax2.set_xlabel("DTW対応フレーム番号")
ax2.set_ylabel("キーポイント間距離差")
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- DTWのフレーム対応を可視化 ---
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# パスを描画
path_array = np.array(path)
ax.plot(path_array[:, 0], path_array[:, 1], 'b-', linewidth=1, alpha=0.7)
ax.scatter(path_array[:, 0], path_array[:, 1], c='red', s=1, alpha=0.5)

ax.set_xlabel('Prediction1のフレーム番号')
ax.set_ylabel('Prediction2のフレーム番号')
ax.set_title('DTWによるフレーム対応関係')
ax.grid(True, alpha=0.3)

# 対角線（完全に同期している場合）を参考線として描画
max_frame = max(prediction.shape[0], prediction2.shape[0])
ax.plot([0, max_frame], [0, max_frame], 'k--', alpha=0.3, label='完全同期線')
ax.legend()

plt.tight_layout()
plt.show()

# --- 各キーポイントの統計情報を表示（DTW前後） ---
print("\n=== DTW前の各キーポイントの統計情報 ===")
for i, joint_idx in enumerate(selected_joints):
    avg_diff = np.mean(joint_diffs_before[joint_idx])
    max_diff = np.max(joint_diffs_before[joint_idx])
    min_diff = np.min(joint_diffs_before[joint_idx])
    std_diff = np.std(joint_diffs_before[joint_idx])
    
    print(f"{selected_labels[i]:>4}: 平均={avg_diff:.2f}, 最大={max_diff:.2f}, 最小={min_diff:.2f}, 標準偏差={std_diff:.2f}")

print("\n=== DTW後の各キーポイントの統計情報 ===")
for i, joint_idx in enumerate(selected_joints):
    avg_diff = np.mean(joint_diffs_after[joint_idx])
    max_diff = np.max(joint_diffs_after[joint_idx])
    min_diff = np.min(joint_diffs_after[joint_idx])
    std_diff = np.std(joint_diffs_after[joint_idx])
    
    print(f"{selected_labels[i]:>4}: 平均={avg_diff:.2f}, 最大={max_diff:.2f}, 最小={min_diff:.2f}, 標準偏差={std_diff:.2f}")

# --- 統計情報の比較 ---
print("\n=== DTW前後の平均差分の変化 ===")
for i, joint_idx in enumerate(selected_joints):
    avg_before = np.mean(joint_diffs_before[joint_idx])
    avg_after = np.mean(joint_diffs_after[joint_idx])
    improvement = avg_before - avg_after
    improvement_percent = (improvement / avg_before) * 100 if avg_before != 0 else 0
    
    print(f"{selected_labels[i]:>4}: {avg_before:.2f} → {avg_after:.2f} (差分: {improvement:+.2f}, {improvement_percent:+.1f}%)")

# --- 全体的な改善度合いの評価 ---
print(f"\n=== 全体評価 ===")
print(f"DTW全体距離: {distance:.2f}")

# 全キーポイントの平均改善率
total_improvement = 0
for i, joint_idx in enumerate(selected_joints):
    avg_before = np.mean(joint_diffs_before[joint_idx])
    avg_after = np.mean(joint_diffs_after[joint_idx])
    if avg_before != 0:
        improvement_percent = ((avg_before - avg_after) / avg_before) * 100
        total_improvement += improvement_percent

average_improvement = total_improvement / len(selected_joints)
print(f"選択キーポイントの平均改善率: {average_improvement:.1f}%")