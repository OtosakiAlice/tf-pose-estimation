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
video_files = glob.glob(os.path.join(video_folder, 'shinjuku_st1.mp4'))
for video_file in video_files:
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise IOError(f"動画ファイル {video_file} が開けませんでした。")

    video_basename = os.path.basename(video_file)
    output_dir_name = os.path.splitext(video_basename)[0]
    output_dir = os.path.join(video_folder, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    # 動画保存用の設定
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    fps = cap.get(cv2.CAP_PROP_FPS)  # 元の動画のフレームレートを使用
    out = cv2.VideoWriter(f'output_{output_dir_name}.mp4', fourcc, fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))), True)
    image_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 動画の処理を開始
    while True:
        # フレーム処理の開始時刻を記録
        start_time = time.time()
    
        ret, frame = cap.read()
        if not ret:
            break
        
        # 現在のフレーム番号を取得
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        # 出力するJSONファイルの設定
        output_json_path = os.path.join(output_dir, f'output_{frame_number}.json')
        frame_data = {
            "version": 1.2,
            "people": []
        }

        # 姿勢推定を実行
        humans = e.inference(frame, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)

        for human in humans:
            keypoints = []  # List to store keypoints data
            # Assuming there are 18 keypoints. If the model has a different number of keypoints, adjust accordingly.
            for i in range(18):
                body_part = human.body_parts.get(i)
                if body_part:  # If the keypoint is detected
                    keypoints.append([int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5), body_part.score])
                else:  # If the keypoint is not detected
                    keypoints.append([0, 0, 0])  # Append zeros for x, y, and score

            frame_data["people"].append({"pose_keypoints_2d": keypoints})
        
        # JSONファイルに保存
        with open(output_json_path, 'w') as f:
            json.dump(frame_data, f, ensure_ascii=False, indent=4)

        # 姿勢推定の結果を描画して表示
        frame = TfPoseEstimator.draw_humans(frame, humans, imgcopy=True)

        # フレーム処理の終了時刻を記録し、FPSを計算
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(frame)

        cv2.imshow('Pose Estimation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    out.release()
    cap.release()
    cv2.destroyAllWindows()

    print(f"姿勢推定の結果を '{output_dir}' ディレクトリに保存しました。")
