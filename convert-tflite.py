import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def convert_frozen_graph_to_tflite(pb_file_path, saved_model_dir, tflite_file_path):
    # フリーズされたグラフをロード
    with tf.io.gfile.GFile(pb_file_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # グラフをTensorFlow 2.0互換グラフに変換
    wrapped_func = tf.compat.v1.wrap_function(
        lambda: tf.import_graph_def(graph_def, name=""), [])
    frozen_func = convert_variables_to_constants_v2(wrapped_func.graph)

    # SavedModelを作成
    tf.saved_model.save(wrapped_func, saved_model_dir)

    # SavedModelをTensorFlow Liteモデルに変換
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()

    # TFLiteモデルをファイルに保存
    with open(tflite_file_path, 'wb') as f:
        f.write(tflite_model)

    print(f'モデルが {tflite_file_path} に保存されました。')

# 使用例
convert_frozen_graph_to_tflite(
    pb_file_path='tf-pose-estimation/models/graph/cmu/graph_opt.pb',  # フリーズされたグラフのパス
    saved_model_dir='tf-pose-estimation/models/graph/cmu/',        # 保存するSavedModelのディレクトリ
    tflite_file_path='tf-pose-estimation/models/graph/cmu/graph.tflite' # 生成するTFLiteモデルのパス
)
