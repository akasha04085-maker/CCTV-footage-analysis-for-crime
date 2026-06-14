import cv2
import os

def extract_frames(video_path, output_folder):
    
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = total_frames / fps

    print("FPS : ", fps)

    print("Total Frames : ", total_frames)

    print("Duration : ", duration, "Seconds")

    frame_count = 0
    
    
    
    while True:
        success, frame = cap.read()
        
        if not success:
            break
        
        filename = os.path.join(
            output_folder,
            f"frame_{frame_count:06d}.jpg"
        )
        
        cv2.imwrite(filename, frame)
        
        frame_count += 1 
        
    cap.release()
    
    print(f"{frame_count} frame extracted successfully.")