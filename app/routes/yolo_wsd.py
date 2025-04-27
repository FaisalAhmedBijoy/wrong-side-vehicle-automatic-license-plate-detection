import os
import shutil
import uuid
import logging
from app.yolo_run_pipeline import run_pipeline  
from fastapi.staticfiles import StaticFiles
from fastapi import UploadFile, File
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, FileResponse

router = APIRouter()

@router.get("/")
async def get_index():
    return {"message": "Hello route triggered!"}

@router.get("/webpage/")
async def get_uploads_html():
    return FileResponse("app/static/uploads.html")

@router.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    temp_video_path = f"app/temp_videos/{uuid.uuid4()}_{file.filename}"
    os.makedirs("app/temp_videos", exist_ok=True)

    try:
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file.file.close()  # ✅ Close the uploaded file properly

        output = run_pipeline(temp_video_path)
        return JSONResponse(content=output)
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)