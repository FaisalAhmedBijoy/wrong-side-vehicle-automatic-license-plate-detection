import uuid
import easyocr
from app.configurations import Config
from app.license_plate_detection import load_yolo_model, process_video

def run_pipeline(video_path: str):
    config = Config()
    vehicle_model = load_yolo_model(config.VEHICLE_DETECTION_MODEL)
    plate_model = load_yolo_model(config.LICENSE_PLATE_DETECTION_MODEL)
    ocr_reader = easyocr.Reader(['en'], gpu=False)

    results_csv = f"app/results/output_{uuid.uuid4()}.csv"
    output_video_path = f"app/results/processed_{uuid.uuid4()}.mp4"

    line_y_blue = int(config.LINE_Y_BLUE)
    line_y_yellow = int(config.LINE_Y_YELLOW)
    window_width = int(config.WINDOW_WIDTH)
    window_height = int(config.WINDOW_HEIGHT)
    fps_reduction = int(config.FPS_REDUCTION)

    process_video(
        video_path,
        vehicle_model,
        plate_model,
        ocr_reader,
        line_y_blue,
        line_y_yellow,
        window_width,
        window_height,
        output_video_path,
        results_csv,
        fps_reduction=fps_reduction
    )

    return {
        "message": "Processing complete",
        "output_video": output_video_path,
        "results_csv": results_csv
    }
