import cv2
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import os
import time
import json
import glob

# モデルのロード
model = 'cmu'
resize_out_ratio = 15.0
w, h = model_wh('432x368')
e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))

# 処理する動画ファイルがあるフォルダ
video_folder = 'tf-pose-estimation/work'

# フォルダ内のすべてのmp4ファイルを取得
video_files = glob.glob(os.path.join(video_folder, '*.avi'))  # AVIファイルを検索

for video_file in video_files:
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise IOError(f"動画ファイル {video_file} が開けませんでした。")

    fps = cap.get(cv2.CAP_PROP_FPS)  # FPSを取得

    video_basename = os.path.basename(video_file)
    output_dir_name = os.path.splitext(video_basename)[0]
    output_dir = os.path.join(video_folder, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    # 動画保存用の設定
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # AVI形式で保存するためにコーデックを変更
    out_video_path = 'output.avi'  # 出力ファイル名を 'output.avi' に変更
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    image_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    previous_keypoints_list = []  # 前のフレームのキーポイントを保存するリスト
    t = 0.5  # 補間係数（0.5は中間点）

    # 動画の処理を開始
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        humans = e.inference(frame, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)

        current_keypoints_list = []
        for human_idx, human in enumerate(humans):
            keypoints = []
            for i in range(18):
                body_part = human.body_parts.get(i)
                if body_part:
                    x_current, y_current = int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5)
                    keypoints.append([x_current, y_current, body_part.score])
                else:
                    if len(previous_keypoints_list) > human_idx and len(previous_keypoints_list[human_idx]) > i:
                        x_prev, y_prev, _ = previous_keypoints_list[human_idx][i]
                        # 線形補間
                        x_interpolated = x_prev + t * (x_current - x_prev)
                        y_interpolated = y_prev + t * (y_current - y_prev)
                        keypoints.append([int(x_interpolated), int(y_interpolated), 0])
                    else:
                        keypoints.append([0, 0, 0])
            current_keypoints_list.append(keypoints)

        previous_keypoints_list = current_keypoints_list

        # JSONデータの作成
        people_keypoints = []
        for keypoints in current_keypoints_list:
            keypoints_flat = [coord for point in keypoints for coord in point]
            people_keypoints.append({"pose_keypoints_2d": keypoints_flat})

        frame_data = {"version": 1.2, "people": people_keypoints}

        # JSONファイルに保存
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        output_json_path = os.path.join(output_dir, f'output_{frame_number}.json')
        with open(output_json_path, 'w') as f:
            json.dump(frame_data, f, ensure_ascii=False, indent=4)

        # 描画処理
        for keypoints in current_keypoints_list:
            for x, y, _ in keypoints:
                if x != 0 and y != 0:
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)

        out.write(frame)

        cv2.imshow('Pose Estimation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()

print(f"姿勢推定の結果を '{output_dir}' ディレクトリに保存しました。")