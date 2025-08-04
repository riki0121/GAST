# Graduation-research
単眼カメラで撮影された動画を入力とし、3次元姿勢推定モデルGASTを用いて、3次元骨格座標を推定する。

## プログラムの仕様

### gen_skes.py
単眼カメラで撮影された動画を入力とし、3次元骨格座標を推定するプログラムです。
実行時オプション引数を指定することで、アニメーションを出力したり、骨格座標をnpzファイルに保存することができます。

### skelton_visual_camera.py
gen_skes.pyで出力されたnpzファイルを読み込み、カメラ座標系に変換し、cm単位に変換し、それぞれのフレームでの骨格座標を結ぶ線を描画し、アニメーションを出力するプログラムです。


## 仮想環境の作成
```
conda create -n GAST python=3.12
conda activate GAST
```

## 必要なパッケージのインストール法
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## プログラムの使い方

### gen_skes.pyの使い方

* 必要なモデルファイルのダウンロードをするフォルダの作成
```
mkdir checkpoint
cd checkpoint
mkdir gastnet
mkdir yolov3
mkdir hrnet
```
* モデルファイルのダウンロード
    * gastnetモデルファイル
    学習済みGAST-Netモデル[27_frame_model.bin](https://drive.google.com/file/d/1vh29QoxIfNT4Roqw1SuHDxxKex53xlOB/)をダウンロードして、checkpoint/gastnetフォルダに保存する
    * yolov3モデルファイル
    学習済みYOLOv3モデルをダウンロードして、checkpoint/yolov3フォルダに保存する
    ```
    cd checkpoint/yolov3
    wget https://pjreddie.com/media/files/yolov3.weights
    ```
    * hrnetモデルファイル
    checkpoint/hrnetフォルダ内にpose_cocoフォルダを作成し、学習済みHRNetモデル[pose_hrnet_w48_384x288.pth](https://drive.google.com/file/d/1UoJhTtjHNByZSm96W3yFTfU5upJnsKiS/view)をダウンロードして、pose_cocoフォルダに保存する
    

最終的なフォルダ構成は以下のようになります。
```
    ${root_path}
    -- checkpoint
        |-- yolov3
            |-- yolov3.weights
        |-- hrnet
            |-- pose_coco
                |-- pose_hrnet_w48_384x288.pth
        |-- gastnet
            |-- 27_frame_model.bin
```



* アニメーションを出力する場合
data/video内に保存された動画ファイルを引数にとり(以下の例では、data/video内のbaseball.mp4)とし、骨格座標を推定し、アニメーションを出力する(outputフォルダに出力される)

    * 1人の骨格座標を推定する場合
    ```
    python gen_skes.py -v baseball.mp4 -np 1 --animation
    ```

    * 2人の骨格座標を推定する場合
    ```
    python gen_skes.py -v baseball.mp4 -np 2 --animation
    ```


* 骨格座標をnpzファイルに保存する場合

data/video内に保存された動画ファイルを引数にとり(以下の例では、data/video内のbaseball.mp4)とし、骨格座標を推定し、npzファイルを出力する(outputフォルダに出力される)

```
    python gen_skes.py -v baseball.mp4 -np 1
```

### skelton_visual_camera.pyの使い方

骨格座標npzファイルのパスを指定し、骨格を描画する

```
python skelton_visual_camera.py
```
実行すると、カメラ座標系の3次元骨格座標(cm単位)を結ぶ線を描画し、アニメーションを表示し、骨格座標はnpzファイルで出力する


## 開発者

* 作成者：上野瑛博
* 所属：関西学院大学工学部情報工学課程
* E-mail：guv80359@kwansei.ac.jp

