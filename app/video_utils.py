import cv2
import os
import shutil
import numpy as np



GLOBAL_PROGRESS = {"percent": 0, "status": "idle"}
def update_progress(percent, status):
    """Updates the global progress dictionary."""
    GLOBAL_PROGRESS["percent"] = percent
    GLOBAL_PROGRESS["status"] = status


def clear_folder(folder_path):
    """Deletes all contents of the specified folder."""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)


def extract_frames(video_path, output_folder, max_width=1024):
    """Extracts frames from a video and saves them as images in the output folder."""
    clear_folder(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Video açılamadı: {video_path}")
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    except:
        total_frames = 1
    
    frame_count = 0
    success = True

    while success:
        success, image = cap.read()
        if success:
            h,w= image.shape[:2]
            if w > max_width:
                scale = max_width / w
                new_h = int(h * scale)
                image = cv2.resize(image, (int(w * scale), new_h))

            frame_name = f"{frame_count:05d}.jpg"
            frame_filename = os.path.join(output_folder, frame_name)
            cv2.imwrite(frame_filename, image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            frame_count += 1

            if frame_count % 20 == 0:
                ratio = frame_count / total_frames
                current_percent = 10 + int(ratio * 90)
                current_percent = min(99, current_percent)
                update_progress(current_percent, f"Frames Extraction: {frame_count}/{total_frames}")
    cap.release()
    update_progress(100, "Frame extraction completed.")
    return frame_count



def create_video_from_frames(frames_folder, output_video_path, fps=30):
    """Creates a video from a sequence of image frames stored in a folder."""
    update_progress(90, "Creating video from processed frames...")
    images = [img for img in os.listdir(frames_folder) if img.endswith(".jpg")]
    images.sort(key=lambda x: int(os.path.splitext(x)[0]))

    if not images:
        raise ValueError("No image frames found in the specified folder.")


    #take shape of first frame
    first_frame = cv2.imread(os.path.join(frames_folder, images[0]))
    height, width, layers = first_frame.shape


    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print("H.264 (avc1) codec çalışmadı, mp4v deneniyor...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    total = len(images)
    for i, image in enumerate(images):
        frame_path = os.path.join(frames_folder, image)
        frame = cv2.imread(frame_path)

        if frame is not None:
            video_writer.write(frame)

        if i % 50 == 0:
            percent = 90 + int((i / total) * 10)
            update_progress(percent, f"Creating video: {i}/{total} frames added.")
    
    cv2.destroyAllWindows()
    video_writer.release()
    update_progress(100, "Video creation completed.")
    return output_video_path


