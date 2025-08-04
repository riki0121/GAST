import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib  # 日本語対応（任意）

# --- キーポイントごとの距離を計算する関数 ---
def calc_joint_diff(pose1, pose2):

    joint_diffs = np.zeros((17, min_length))  # shape = (17, min_length)

    for frame in range(min_length): #１フレームごとにループ
        for i in range(17): # 17キーポイント分ループ
            joint_diffs[i, frame] = np.linalg.norm(pose1[frame, i] - pose2[frame, i]) # 各キーポイントの距離を計算

    return joint_diffs

# --- ファイル読み込み ---
data1 = np.load("output/my_pitching1.npz")  # 通常フォーム
data2 = np.load("output/my_pitching_slow.npz")  # 比較対象フォーム

pose1 = data1["reconstruction"][0]  # (フレーム数, 17, 3)
pose2 = data2["reconstruction"][0]

# --- 距離をフレームごとに計算 ---
min_length = min(len(pose1), len(pose2))  # 共通のフレーム数で揃える
    

# 距離を計算
joint_diffs = calc_joint_diff(pose1, pose2)

# グラフプロット
plt.figure(figsize=(12, 6))
x = np.arange(joint_diffs.shape[1])

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

for i, joint_idx in enumerate(selected_joints):
    plt.plot(x, joint_diffs[joint_idx], label=f'{selected_labels[i]}')

plt.xlabel("対応フレーム番号")
plt.ylabel("距離差（3D空間）")
plt.title("各キーポイントのユークリッド距離差（DTW前)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True)
plt.show()