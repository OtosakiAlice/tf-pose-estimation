import json
import numpy as np
import os
import glob

# JSONファイルの読み込み関数
def load_keypoints(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    # JSONファイルに人物データが存在するかを確認
    if 'people' in data and len(data['people']) > 0:
        keypoints = data['people'][0]['pose_keypoints_2d']
        return [(keypoints[i], keypoints[i + 1]) for i in range(0, len(keypoints), 3)]
    else:
        # キーポイントのデータが存在しない場合、空のリストを返す
        return []

# ディレクトリから全JSONファイルを取得する関数
def get_json_files(directory):
    return glob.glob(os.path.join(directory, '*.json'))

# フォルダ内のJSONファイルから距離平均を計算する関数
def calculate_average_distance(directory):
    files = get_json_files(directory)
    all_distances = []
    for file1 in files:
        for file2 in files:
            if file1 != file2:
                keypoints1 = load_keypoints(file1)
                keypoints2 = load_keypoints(file2)

                # キーポイントが両方のファイルに存在する場合のみ距離計算
                if keypoints1 and keypoints2:
                    distances = [np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) for p1, p2 in zip(keypoints1, keypoints2)]
                    all_distances.extend(distances)

    # 平均距離の計算
    if all_distances:
        average_distance = np.mean(all_distances)
        return average_distance
    else:
        return None

# ディレクトリパス設定（ユーザーに合わせて変更）
directory1 = 'tf-pose-estimation/work/hikaku/162353.crf3'
directory2_parent = 'tf-pose-estimation/work'

# directory2内のすべてのフォルダに対して距離計算と出力を行う
for subdirectory in os.listdir(directory2_parent):
    directory2 = os.path.join(directory2_parent, subdirectory)
    average_distance = calculate_average_distance(directory2)
    if average_distance is not None:
        print(f"{subdirectory}: {average_distance}")
    else:
        print(f"{subdirectory}: No keypoints data available for distance calculation.")
