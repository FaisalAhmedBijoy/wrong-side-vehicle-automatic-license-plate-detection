from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
from uuid import uuid4
from pathlib import Path

# Import the functions for video processing
from license_plate_detection import process_video, load_yolo_model  # Assuming the core logic is in license_plate_detection.py

app = FastAPI()

# Directory setup
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load models
vehicle_model = load_yolo_model(model_path='models/yolo11l.pt')
plate_model = load_yolo_model(model_path='models/numer_plates_detection_model/license_plate_detector.pt')

@app.post("/process-video")
async def process_video_endpoint(
    video: UploadFile = File(...),
    yellow_line: int = Form(...),
    blue_line: int = Form(...)
):
    """Endpoint to process video and return output files."""
    # Save uploaded file
    input_video_path = f"{UPLOAD_DIR}/{uuid4()}_{video.filename}"
    with open(input_video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # Define output paths
    output_video_path = f"{OUTPUT_DIR}/{Path(input_video_path).stem}_output.mp4"
    csv_file_path = f"{OUTPUT_DIR}/{Path(input_video_path).stem}_output.csv"

    # Process the video
    try:
        process_video(
            video_path=input_video_path,
            vehicle_model=vehicle_model,
            plate_model=plate_model,
            ocr_reader=None,  # Replace with your EasyOCR object if needed
            line_y_blue=blue_line,
            line_y_yellow=yellow_line,
            csv_file_path=csv_file_path,
            output_video_path=output_video_path,
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

    return {
        "video_url": f"/{output_video_path}",
        "csv_url": f"/{csv_file_path}"
    }

# Serve static files
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# Serve index.html
@app.get("/")
async def serve_index():
    return FileResponse("index.html")
