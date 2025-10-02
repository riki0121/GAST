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


def distance_to_score(dist, max_dist):
    """
    距離をスコア(0~100)に変換する関数
    max_dist以上ならスコア0
    """
    point = (1 - dist / max_dist) * 100
    score = max(0, min(100,point))

    return score



def cal_scores(form1, form2, max_dist, size=20):
    """
    2つのフォームの類似度スコアを算出する関数
    size: 何フレームごとにスコアを出すか
    max_dist: この距離以上ならスコア0
    
    スコアをまとめたリスト(results)を返す
    """

    frames = min(len(form1), len(form2))
    num = frames // size          # スコアを出す回数

    results = []

    keypoint_labels = ["左肩","左肘","右肩","右肘"]
    keypoint_nums   = [11,12,14,15]


    for i in range(num):
        start = i * size        # スコアを出すフレームの開始位置
        end = start + size      # スコアを出すフレームの終了位置
        seg1 = form1[start:end] # (size,17,3)
        seg2 = form2[start:end]

        joint_scores = []

        #---個別スコア(肩,肘)を求める---
        for j in keypoint_nums:

            dists = [np.linalg.norm(seg1[t, j] - seg2[t, j]) for t in range(len(seg1))]#フレーム間の距離を計算,リストに保存
            
            #平均距離を計算 -> スコアに変換
            mean_dists = np.mean(dists) 
            score = distance_to_score(mean_dists, max_dist) #スコアに変換
      
            joint_scores.append(score) 


        #---全体スコア（全関節を計算）を求める---
        for s in range(17):
   
            all_dists = [np.linalg.norm(seg1[k,s]-seg2[k,s])for k in range(len(seg1))]#フレーム間の距離を計算,リストに保存

            #平均距離を計算 -> スコアに変換
            all_mean_dists = np.mean(all_dists) 
            score = distance_to_score(all_mean_dists, max_dist) #スコアに変換
            joint_scores.append(score) 
        #全体スコア
        overall_score = np.mean(joint_scores)


            # --- 結果をまとめる ---
        results.append({
            "segment": f"{start}-{end}",
            "overall": overall_score,
            "per_joint": dict(zip(keypoint_labels, joint_scores))
        })

    return results


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

# --- DTW path に基づいて、data2 を整列させる ---
aligned_data1 = []
aligned_data2 = []
counter = 0

max_frames = 130
print(f"max_frames: {max_frames}")
print(f"len(reconstruction_data_camera): {len(reconstruction_data_camera)}")
print(f"len(reconstruction_data_camera2): {len(reconstruction_data_camera2)}")

for i, j in filtered_path:
    if i < len(reconstruction_data_camera) and j < len(reconstruction_data_camera2):
        aligned_data1.append(reconstruction_data_camera[i])
        aligned_data2.append(reconstruction_data_camera2[j])
        counter += 1

print(f'counter:{counter}, i:{i}, j:{j}')
aligned_data1 = np.array(aligned_data1)
aligned_data2 = np.array(aligned_data2)


for i in range(len(aligned_data1)):
    hip1 = aligned_data1[i, 0]  # form1 の腰
    hip2 = aligned_data2[i, 0]  # form2 の腰
    diff = hip1 - hip2        # 座標の差分を計算
    aligned_data2[i] += diff  # form2 を平行移動

scores = cal_scores(aligned_data1, aligned_data2, 1,size=20)

    # 出力確認
for s in scores:
    print(f"-----{s['segment']}フレーム: 全体={s['overall']:.1f}点-----")
    print(f"左肩={s['per_joint']['左肩']:.1f}点 "f"右肩={s['per_joint']['右肩']:.1f}点")
    print(f"左肘={s['per_joint']['左肘']:.1f}点 "f"右肘={s['per_joint']['右肘']:.1f}点")