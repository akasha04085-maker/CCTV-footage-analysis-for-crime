from src.frame_extractor import extract_frames
from src.image_preprocessing import resize_frames   
from src.normalizeation import normalize_frames
from src.image_loader import load_image

video_path = "videos/fight_video1.mp4"

extract_frames(
    video_path,
    "frames"
)

resize_frames(
    "frames",
    "resized_frames"
)

normalize_frames(
    "resized_frames",
    "normalized_frames"
)