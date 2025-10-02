import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from common.camera import normalize_screen_coordinates, world_to_camera
from tensorflow.keras.models import load_model
from common.camera import world_to_camera

# 1つの npz ファイルから読み込む（1つの投球フォーム）
model = load_model('lstm_model.h5')

# データ読み込み
data = np.load('P1-30_1.npz')
key = data.files[0]
X = data[key]

print("読み込み直後の X shape:", X.shape)

# (1, 133, 17, 3) → (133, 17, 3) に変換（バッチ次元を削除）
if X.shape[0] == 1:
    X = X[0]

print("バッチ次元削除後 X shape:", X.shape)  # → (133, 17, 3)

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

print("最終的な X_val shape:", X_val.shape)  # → (1, 130, 51)

# 推論
prediction = model.predict(X_val)
predicted_class = np.argmax(prediction, axis=1)[0]

print("予測されたクラス:", predicted_class)