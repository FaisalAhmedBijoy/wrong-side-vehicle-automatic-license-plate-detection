from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid
import logging
import uvicorn

from app.license_plate_pipeline import run_pipeline  # your import

app = FastAPI()

# Serve static files at /static
app.mount("/app/static", StaticFiles(directory="app/static"), name="static")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the uploads.html manually at root
@app.get("/")
async def root():
    return FileResponse("static/uploads.html")

# Upload endpoint
@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    temp_video_path = f"temp_videos/{uuid.uuid4()}_{file.filename}"
    os.makedirs("temp_videos", exist_ok=True)

    with open(temp_video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        output = run_pipeline(temp_video_path)
        return JSONResponse(content=output)
    finally:
        os.remove(temp_video_path)



@app.get("/")
async def get_index():
    return {"message": "Hello route triggered!"}

# Entry point for running the application
def run():
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")


if __name__ == "__main__":
    run()