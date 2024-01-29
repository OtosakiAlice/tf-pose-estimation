import cv2
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

import tensorflow as tf

# パラメータの設定
video_file = '/home/pana8912/research/tf-pose-estimation/video.mp4'  # 処理する動画ファイルのパス
model = 'cmu'  # 使用するモデル。'mobilenet_thin', 'mobilenet_v2_large', 'mobilenet_v2_small'も選べます
resize_out_ratio = 10.0  # 出力画像のサイズ比。大きいほど処理が遅くなりますが、精度は上がります

# モデルのロード
w, h = model_wh('432x368')  # モデルの入力サイズ。'0x0'に設定するとデフォルトサイズが使用されます
if w == 0 or h == 0:
    e = TfPoseEstimator(get_graph_path(model), target_size=(432, 368))
else:
    e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))

# 動画ファイルを読み込む
cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    raise IOError("動画ファイルが開けません")

# 動画書き出し用の設定
fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter('output.mp4', fourcc, 30, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))), True)

while True:
    ret, image = cap.read()
    if not ret:
        break
    
    # 姿勢推定の実行
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)

    print(humans)

    # 姿勢の描画
    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

    # 結果の表示
    cv2.imshow('tf-pose-estimation result', image)

    # 結果の保存
    out.write(image)  # 結果を動画ファイルに書き込む

    # qを押して終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 終了処理
cap.release()
out.release()
cv2.destroyAllWindows()