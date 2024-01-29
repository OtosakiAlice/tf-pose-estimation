import cv2
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import os
import time
import json
import glob
import multiprocessing

def process_video(video_file, gpu_id):
    # GPUを指定
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # モデルのロード
    model = 'cmu'
    resize_out_ratio = 4.0
    w, h = model_wh('432x368')
    e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise IOError(f"動画ファイル {video_file} が開けませんでした。")

    video_basename = os.path.basename(video_file)
    output_dir_name = os.path.splitext(video_basename)[0]
    output_dir = os.path.join('output/', output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    # 動画保存用の設定
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(os.path.join(output_dir, f'output_{output_dir_name}.mp4'), fourcc, fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))), True)

    while True:
        start_time = time.time()
    
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        output_json_path = os.path.join(output_dir, f'output_{frame_number}.json')
        frame_data = {
            "version": 1.2,
            "people": []
        }

        humans = e.inference(frame, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)

        for human in humans:
            keypoints = []
            for i in range(18):
                body_part = human.body_parts.get(i)
                if body_part:
                    keypoints.append([body_part.x, body_part.y, body_part.score])
                else:
                    keypoints.append([0, 0, 0])

            frame_data["people"].append({"pose_keypoints_2d": keypoints})
        
        with open(output_json_path, 'w') as f:
            json.dump(frame_data, f, ensure_ascii=False, indent=4)

        frame = TfPoseEstimator.draw_humans(frame, humans, imgcopy=True)

        end_time = time.time()
        fps = 1 / (end_time - start_time)
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(frame)

    out.release()
    cap.release()

def main():
    video_folder = '20170104/'
    video_files = glob.glob(os.path.join(video_folder, '*.mp4'))
    
    # 使用するGPUのIDリスト
    gpu_ids = [0, 1]  # 2つのGPUがあると仮定

    # プロセスプールを作成し、各動画を異なるGPUで処理する
    processes = []
    for video_file, gpu_id in zip(video_files, gpu_ids):
        p = multiprocessing.Process(target=process_video, args=(video_file, gpu_id))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == '__main__':
    main()
