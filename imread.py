import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import time

# TensorFlow HubからFaster R-CNNモデルをロード
module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
detector = hub.load(module_handle).signatures['default']

# 画像をNumPy配列に読み込む
image_path = 'tf-pose-estimation/multi_person.jpg'  # 実際の画像のパスに置き換えてください
image_np = np.array(Image.open(image_path))

# 推論を実行する前に画像をバッチに変換
converted_img = tf.image.convert_image_dtype(image_np, tf.float32)[tf.newaxis, ...]
start_time = time.time()
result = detector(converted_img)
end_time = time.time()

# 結果をNumPy配列に変換
result = {key: value.numpy() for key, value in result.items()}

print(result)

# 検出結果をフィルタリングして人のみを取得
person_indices = np.where(result["detection_class_entities"] == b'Person')[0]
person_boxes = result["detection_boxes"][person_indices]
person_scores = result["detection_scores"][person_indices]

# 検出結果に基づいて画像にボックスを描画する
image_pil = Image.fromarray(np.uint8(image_np)).convert('RGB')
draw = ImageDraw.Draw(image_pil)
font = ImageFont.load_default()

max_boxes = 20  # 描画する最大のボックス数
min_score = 0.1  # スコアの閾値

for i in range(min(len(person_boxes), max_boxes)):
    if person_scores[i] >= min_score:
        ymin, xmin, ymax, xmax = tuple(person_boxes[i])
        display_str = "Person: {}%".format(int(100 * person_scores[i]))
        color = 'red'
        draw.rectangle(
            [(xmin * image_pil.width, ymin * image_pil.height), 
             (xmax * image_pil.width, ymax * image_pil.height)], 
            outline=color, width=2)
        draw.text(
            (xmin * image_pil.width, ymin * image_pil.height), 
            display_str, 
            fill=color, 
            font=font)

# 画像を表示する
plt.figure(figsize=(12, 8))
plt.imshow(image_pil)
plt.axis('off')  # 軸を非表示にする
plt.show()