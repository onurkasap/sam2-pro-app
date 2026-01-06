import os
import cv2
import numpy as np
import base64
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from PIL import Image
from io import BytesIO
import asyncio

#moduls
from app.core import get_image_predictor, get_video_predictor, get_dtype
from app.video_utils import extract_frames, create_video_from_frames, clear_folder, update_progress, GLOBAL_PROGRESS


app = FastAPI(title="SAM2 Video Segmentation API-Pro Server")


#Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
TEMP_FRAMES_DIR = os.path.join(BASE_DIR, 'temp_frames')
PROCESS_DIR = os.path.join(BASE_DIR, 'processed')
PROCESSED_FRAMES_DIR = os.path.join(PROCESS_DIR, "processed_frames")


#Create Folders
for d in [UPLOAD_DIR, TEMP_FRAMES_DIR, PROCESS_DIR, PROCESSED_FRAMES_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)


#-----------0. HomePage ----------
@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return "<h1>Hata: index.html dosyası bulunamadı!</h1>"
    


# --- İLERLEME DURUMU ---
@app.get("/progress")
async def get_progress():
    return GLOBAL_PROGRESS



#---------- 1. Image Predictor ----------
@app.post("/predict-image")
async def predict_image(
    file: UploadFile = File(...), 
    point_x: int = Form(...), 
    point_y: int = Form(...)):

    image_predictor = get_image_predictor()
    if not image_predictor:
        raise HTTPException(status_code=500, detail="Image predictor model is not loaded.")
    
    #Read Image
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)

    #SAM2 Inference
    image_predictor.set_image(image_np)
    masks, _, _ = image_predictor.predict(
        point_coords=np.array([[point_x, point_y]]), 
        point_labels=np.array([1]), 
        multimask_output=False)
    
    #Black Background with Mask
    mask = masks[0]
    mask_rgb = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    processed_image = image_np * mask_rgb

    #Base64 Encode
    res_pil = Image.fromarray(processed_image.astype(np.uint8))
    buffered = BytesIO()
    res_pil.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {"image": f"data:image/jpeg;base64,{img_str}"}




#---------- 2. Video Upload and Take First Frame ----------
@app.post("/upload-video")
def upload_video(file: UploadFile = File(...)):
    print("Received video upload request.")
    #Max 20MB
    try:
        update_progress(0, "Starting video upload...")
        clear_folder(TEMP_FRAMES_DIR)
        
        # Save video by splitting it into chunks
        video_path = os.path.join(UPLOAD_DIR, "input_video.mp4")
        
        with open(video_path, "wb") as buffer:
            while True:
                chunk = file.file.read(1024 * 1024) # 1MB oku
                if not chunk:
                    break
                buffer.write(chunk)

        print("Starting video upload to disk...")        
        update_progress(5, "Video do analyze .")
        print("Writing video to disk completed.")

        # Frame seperation
        print("Starting frame extraction...")
        frame_count = extract_frames(video_path, TEMP_FRAMES_DIR)
        print(f"Finised Processed: Total Frame: {frame_count}")

        if frame_count == 0:
             raise HTTPException(status_code=500, detail="Video'dan kare çıkarılamadı.")

        # Convert first frame to base64 and return
        all_frames = sorted(os.listdir(TEMP_FRAMES_DIR))
        jpg_frames = [f for f in all_frames if f.endswith(".jpg")]

        if not jpg_frames: raise HTTPException(500, "No JPG frames found in the temporary frames directory.")

        first_frame_name = jpg_frames[0]
        first_frame_path = os.path.join(TEMP_FRAMES_DIR, first_frame_name)
        if os.path.exists(first_frame_path):
            with open(first_frame_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            update_progress(10, "Video upload and frame extraction completed.")
            return {"first_frame": f"data:image/jpeg;base64,{encoded}"}
        else:
            raise HTTPException(status_code=500, detail="First frame file not found.")
            
    except Exception as e:
        print(f"VIDEO UPLOAD ERROR: {e}")
        import traceback
        traceback.print_exc() # Detaylı hata basar
        raise HTTPException(status_code=500, detail=str(e))
    

#---------- 3. Video Predictor ----------
@app.post("/process-video")
def predict_video(point_x: int = Form(...), point_y: int = Form(...)):
    try:
        video_predictor = get_video_predictor()
        dtype = get_dtype()
        if not video_predictor:
            raise HTTPException(status_code=500, detail="Video predictor model is not loaded.")
        
        update_progress(5, "AI Engine is being loaded (Loading to GPU, waiting)...")
        
        #State Starting
        with torch.autocast(device_type="cuda", dtype=dtype):
            inference_state = video_predictor.init_state(video_path=TEMP_FRAMES_DIR)

        #Coordinates for first frame
        video_predictor.add_new_points_or_box(
            inference_state = inference_state,
            frame_idx=0,
            obj_id = 1,
            points = np.array([[point_x, point_y]], dtype=np.float32),
            labels = np.array([1], dtype=np.int32)
        )
        print(f"Added point ({point_x}, {point_y}) for object ID 1 on frame 0.")
        
        
        #Process Video
        clear_folder(PROCESSED_FRAMES_DIR)

        #total frames
        total_frames = len(os.listdir(TEMP_FRAMES_DIR))
        processed_count = 0
        with torch.autocast(device_type="cuda", dtype=dtype):
            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):

                safe_frame_idx = int(out_frame_idx)
                frame_name = f"{safe_frame_idx:05d}.jpg"

                #read original frame
                frame_path = os.path.join(TEMP_FRAMES_DIR, frame_name)

                if not os.path.exists(frame_path):
                    print(f"⚠️ Dosya bulunamadı atlanıyor: {frame_name}")
                    continue
                
                original_frame = cv2.imread(frame_path)
                if original_frame is None:
                    print(f"⚠️ Dosya okunamadı atlanıyor: {frame_name}")
                    continue


                #take mask (logits -> mask)
                mask = (out_mask_logits[0] > 0).cpu().numpy().squeeze()

                #black background with mask
                mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

                processed_frame = original_frame * mask_3ch
                
                cv2.imwrite(os.path.join(PROCESSED_FRAMES_DIR, frame_name), processed_frame)

                # Update Progress Bar 
                processed_count += 1
                if processed_count % 10 == 0:
                    ai_progress = int((processed_count / total_frames) * 60)
                    update_progress(30 + ai_progress, f"AI Proccessing: {processed_count}/{total_frames}")

        #Create Video from Processed Frames
        output_video_path = os.path.join(PROCESS_DIR, "final_video.mp4")
        create_video_from_frames(PROCESSED_FRAMES_DIR, output_video_path, fps=30)
        return FileResponse(output_video_path, media_type='video/mp4', filename="sam2_results.mp4")

    except Exception as e:
            print(f"VIDEO PROCESSING ERROR: {e}")
            import traceback
            traceback.print_exc() # Detaylı hata basar
            raise HTTPException(status_code=500, detail=str(e))