import json  # JSONファイルの読み込みと操作のためのライブラリをインポート
import os  # ファイル操作のためのライブラリをインポート
import numpy as np  # 数値演算のためのライブラリをインポート

def calculate_distance(p1, p2):
    # 二点間のユークリッド距離を計算する関数を定義
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def process_json_files(json_folder):
    # 指定されたフォルダ内のJSONファイルを処理する関数を定義
    json_files = sorted([os.path.join(json_folder, file) for file in os.listdir(json_folder) if file.endswith('.json')])
    # 指定フォルダ内のJSONファイルのパスをリストに格納し、ソート

    previous_frame_data = None  # 前のフレームのキーポイントデータを保持する変数を初期化
    personwise_distances = {}  # 人物ごとの移動距離を記録する辞書を初期化

    for json_file in json_files:
        # JSONファイルごとに処理を繰り返す
        with open(json_file, 'r') as f:
            current_frame_data = json.load(f)
        # JSONファイルを読み込み、その内容をcurrent_frame_dataに格納

        if previous_frame_data is not None:
            # 前のフレームと現在のフレームのキーポイントの移動距離を計算
            for person_index, person_data in enumerate(current_frame_data['people']):
                # 現在のフレーム内の各人物ごとに処理を繰り返す
                current_keypoints = person_data['pose_keypoints_2d']
                if person_index not in personwise_distances:
                    personwise_distances[person_index] = 0
                # personwise_distancesに人物のキーが存在しない場合、0で初期化

                if person_index < len(previous_frame_data['people']):
                    previous_keypoints = previous_frame_data['people'][person_index]['pose_keypoints_2d']
                    # 前のフレームの対応する人物のキーポイントを取得
                    for i in range(0, len(current_keypoints), 3):
                        if current_keypoints[i] != 0 and current_keypoints[i+1] != 0 and previous_keypoints[i] != 0 and previous_keypoints[i+1] != 0:
                            # 有効なキーポイントの場合のみ距離を計算
                            distance = calculate_distance((current_keypoints[i], current_keypoints[i+1]), (previous_keypoints[i], previous_keypoints[i+1]))
                            # 二点間のユークリッド距離を計算
                            personwise_distances[person_index] += distance
                            # 人物ごとの距離を加算

        previous_frame_data = current_frame_data
        # 現在のフレームデータを次の比較のために保存

    # 各人物の平均移動距離を計算
    average_distances = {person: total_distance / len(json_files) for person, total_distance in personwise_distances.items()}
    # 人物ごとの距離をフレーム数で割って平均を計算
    return average_distances

# JSONファイルが格納されているフォルダを指定
json_folder = 'tf-pose-estimation/work/162353.crf.avi3' # 実際のパスに置き換えてください
average_distances = process_json_files(json_folder)
# 指定フォルダ内のJSONファイルを処理し、平均移動距離を計算

# 各人物の平均移動距離を表示
print("Average distances per person:", average_distances)