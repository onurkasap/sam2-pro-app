import torch
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
import os


#Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS = os.path.join(BASE_DIR, 'checkpoints', "sam2.1_hiera_base_plus.pt")
CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"

#Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
    dtype = torch.bfloat16
    print("Using bfloat16 for better performance on compatible GPUs.")
else:
    dtype = torch.float32
    print("Stanard float32 is used. For better performance, use a GPU with compute capability 8.0 or higher.")

print(f"Using device: {device}")
print(f"Loading SAM2 model from {CHECKPOINTS}...")

#1.Image Predictor
try:
    sam2_model = build_sam2(CONFIG, CHECKPOINTS, device=device)
    image_predictor = SAM2ImagePredictor(sam2_model)
    print("SAM2 Image Predictor loaded successfully.")
except Exception as e:
    print(f"Error loading SAM2 Image Predictor: {e}")
    image_predictor = None


#2.Video Predictor
try:
    video_predictor = build_sam2_video_predictor(CONFIG, CHECKPOINTS, device=device)
    print("SAM2 Video Predictor loaded successfully.")
except Exception as e:
    print(f"Error loading SAM2 Video Predictor: {e}")
    video_predictor = None

def get_image_predictor():
    return image_predictor

def get_video_predictor():
    return video_predictor

def get_dtype():
    return dtype

