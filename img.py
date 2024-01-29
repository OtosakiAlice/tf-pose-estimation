import cv2
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

# モデルの入力サイズを取得
w, h = model_wh('432x368')

# モデルをロード
model = 'cmu'  # 使用するモデル。'mobilenet_thin', 'mobilenet_v2_large', 'mobilenet_v2_small'も選べます
resize_out_ratio = 10.0  # 出力画像のサイズ比。大きいほど処理が遅くなりますが、精度は上がります

# モデルのロード
w, h = model_wh('432x368')  # モデルの入力サイズ。'0x0'に設定するとデフォルトサイズが使用されます
if w == 0 or h == 0:
    e = TfPoseEstimator(get_graph_path(model), target_size=(432, 368))
else:
    e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))

# 画像を読み込む
image = cv2.imread('tf-pose-estimation/images/milkboy.jpg')

# 姿勢推定を実行
humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)

for item in humans:
    print(humans)

# 推定された姿勢を描画
image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

# 推定結果をコンソールに出力

# 推定結果を表示
cv2.imshow('tf-pose-estimation result', image)

# 'q'を押すまで画像を表示し続ける
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break