import cv2
import os
# accept video file, read using opencv,extract frames every second, save them as images
def extract_frames(video_path, output_dir, interval=1):
    os.makedirs(output_dir, exist_ok=True)
    # open the video
    cap = cv2.VideoCapture(video_path)
    # get frames per second
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)
    count = 0
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read() # read one frame
        if not ret:
            break
        if count % frame_interval == 0:
            filename = f"{output_dir}/frame_{frame_id}.jpg"
            cv2.imwrite(filename, frame)  #save as image
            frame_id += 1
        count += 1
    cap.release()
