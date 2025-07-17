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

# --- DTWでフレーム対応を取得（キーポイント別距離計算で）---
distance, path = fastdtw(prediction, prediction2, dist=pose_distance, radius=1)
print(f"DTW距離: {distance:.2f}")
print(f"対応フレーム数: {len(path)}")

# --- 各関節ごとのユークリッド距離差（DTW後）を格納 ---
joint_diffs = [[] for _ in range(17)]  # 各関節の差分を格納するリスト

#DTWで決まった対応フレームを使って距離を分析
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

# 抽出したいキーポイントのインデックス
selected_joints = [7, 11, 14, 12, 15]  #背骨、左肩、右肩、左肘、右肘
selected_labels = ["背骨", "左肩", "右肩", "左肘", "右肘"]

# グラフ描画（選択したキーポイントのみ）
for i, joint_idx in enumerate(selected_joints):
    plt.plot(x, joint_diffs[joint_idx], label=f'{selected_labels[i]}', linewidth=2)

plt.title("主要キーポイントのユークリッド距離差（DTW後)")
plt.xlabel("対応フレーム番号（DTW）")
plt.ylabel("距離差（3D空間）")
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- 各キーポイントの統計情報を表示 ---
print("\n=== 各キーポイントの統計情報 ===")
for i, joint_idx in enumerate(selected_joints):
    avg_diff = np.mean(joint_diffs[joint_idx])
    max_diff = np.max(joint_diffs[joint_idx])
    min_diff = np.min(joint_diffs[joint_idx])
    
    print(f"{selected_labels[i]:>4}: 平均={avg_diff:.2f}, 最大={max_diff:.2f}, 最小={min_diff:.2f}")