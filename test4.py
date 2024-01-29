import cv2
import os

def save_binary_frames(video_path, output_folder, img_format='jpg', threshold=127):
    # 動画ファイルを読み込む
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"動画ファイル {video_path} を開けませんでした。")
        return

    # 出力フォルダの作成
    os.makedirs(output_folder, exist_ok=True)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # グレースケールに変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 二値化処理
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # 二値化されたフレームの書き出し
        frame_filename = f"frame_{frame_count:04d}.{img_format}"
        frame_path = os.path.join(output_folder, frame_filename)
        cv2.imwrite(frame_path, binary)

        frame_count += 1

    cap.release()
    print(f"{frame_count} フレームが二値化されて {output_folder} に保存されました。")

# 使用例
video_path = 'tf-pose-estimation/work/162353.crf3.avi'  # 動画ファイルのパス
output_folder = 'tf-pose-estimation/work/output_crf3_f'  # 画像を保存するフォルダのパス

save_binary_frames(video_path, output_folder)