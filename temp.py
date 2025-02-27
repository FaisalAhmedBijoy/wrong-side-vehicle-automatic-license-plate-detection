import os
import cv2
import easyocr
import pandas as pd
from dotenv import load_dotenv
from collections import defaultdict
from ultralytics import YOLO

# Load environment variables
load_dotenv()

class Config:
    """Configuration class to manage environment variables."""
    REQUIRED_VARS = [
        "VEHICLE_DETECTION_MODEL", "LICENSE_PLATE_DETECTION_MODEL", "INPUT_SAMPLE_VIDEO_PATH",
        "YOLO_OUTPUT_VIDEO_PATH", "OUTPUT_RESULTS_CSV_PATH", "FPS_REDUCTION", "LINE_Y_BLUE",
        "LINE_Y_YELLOW", "WINDOW_WIDTH", "WINDOW_HEIGHT"
    ]
    
    def __init__(self):
        for var in self.REQUIRED_VARS:
            setattr(self, var, self.get_required_env(var))

    def get_required_env(self, var_name):
        value = os.getenv(var_name)
        if value is None:
            raise ValueError(f"Missing required environment variable: {var_name}")
        return int(value) if value.isdigit() else value

config = Config()


def load_yolo_model(model_path):
    """Load YOLO model."""
    return YOLO(model_path)


def detect_and_recognize_number_plate(vehicle_crop, plate_model, ocr_reader):
    """Detect and recognize a number plate in the given vehicle crop."""
    try:
        plate_results = plate_model.predict(vehicle_crop, conf=0.5)
        if plate_results[0].boxes.data is not None and len(plate_results[0].boxes.xyxy) > 0:
            x1, y1, x2, y2 = map(int, plate_results[0].boxes.xyxy[0].cpu().numpy())
            plate_crop = vehicle_crop[y1:y2, x1:x2]
            ocr_results = ocr_reader.readtext(plate_crop, detail=0)
            return " ".join(ocr_results) if ocr_results else "No Plate Detected"
    except Exception as e:
        print(f"Error detecting/recognizing plate: {e}")
    return "Error"


def process_frame(frame, vehicle_model, plate_model, ocr_reader, line_y_blue, line_y_yellow,
                  object_status, direction_counts, results_df, class_counts):
    """Process a single frame to detect vehicles and recognize plates."""
    vehicle_results = vehicle_model.track(frame, persist=True, classes=[2, 3, 5, 7], conf=0.6, imgsz=640)
    if vehicle_results[0].boxes.data is not None:
        for box, track_id, class_idx, conf in zip(
            vehicle_results[0].boxes.xyxy.cpu().numpy(),
            vehicle_results[0].boxes.id.int().cpu().numpy().tolist(),
            vehicle_results[0].boxes.cls.int().cpu().numpy().tolist(),
            vehicle_results[0].boxes.conf.cpu().numpy()
        ):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            class_name = vehicle_model.names[class_idx]

            if track_id is None:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"{class_name} ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            object_status.setdefault(track_id, {"yellow": False, "blue": False})

            if line_y_yellow - 10 <= cy <= line_y_yellow + 10:
                object_status[track_id]["yellow"] = True
            if line_y_blue - 10 <= cy <= line_y_blue + 10:
                object_status[track_id]["blue"] = True

            direction = None
            if object_status[track_id]["yellow"] and not object_status[track_id]["blue"]:
                direction = "Right"
                direction_counts["right_direction"] += 1
                class_counts[class_name]["right"] += 1
            elif object_status[track_id]["blue"] and not object_status[track_id]["yellow"]:
                direction = "Wrong"
                direction_counts["wrong_direction"] += 1
                class_counts[class_name]["wrong"] += 1

            if direction:
                vehicle_crop = frame[y1:y2, x1:x2]
                number_plate = detect_and_recognize_number_plate(vehicle_crop, plate_model, ocr_reader)
                results_df.append({'Track ID': track_id, 'Vehicle Class': class_name, 'Direction': direction, 'Confidence': conf, 'Number Plate': number_plate})

    cv2.putText(frame, f"Right: {direction_counts['right_direction']}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Wrong: {direction_counts['wrong_direction']}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.line(frame, (50, line_y_yellow), (frame.shape[1] - 50, line_y_yellow), (0, 255, 255), 3)
    cv2.line(frame, (50, line_y_blue), (frame.shape[1] - 50, line_y_blue), (255, 0, 0), 3)

def save_video(output_path, frame, frame_width, frame_height, fps, writer=None):
    """Save the processed video frames to an output file."""
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    writer.write(frame)
    return writer

def process_video():
    """Process video for vehicle detection, direction analysis, and plate recognition."""
    cap = cv2.VideoCapture(config.INPUT_SAMPLE_VIDEO_PATH)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_skip = fps // int(config.FPS_REDUCTION)

    object_status = defaultdict(lambda: {"yellow": False, "blue": False})
    direction_counts = {"right_direction": 0, "wrong_direction": 0}
    class_counts = defaultdict(lambda: {"right": 0, "wrong": 0})
    results_df, frame_count = [], 0

    writer = cv2.VideoWriter(config.YOLO_OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        frame = cv2.resize(frame, (int(config.WINDOW_WIDTH), int(config.WINDOW_HEIGHT)))
        process_frame(frame, vehicle_model, plate_model, ocr_reader, int(config.LINE_Y_BLUE), int(config.LINE_Y_YELLOW), object_status, direction_counts, results_df, class_counts)
        # writer.write(frame)
        writer = save_video(config.YOLO_OUTPUT_VIDEO_PATH, frame, 1000, 900, fps, writer)
        cv2.imshow("Processed Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    pd.DataFrame(results_df).to_csv(config.OUTPUT_RESULTS_CSV_PATH, index=False)

if __name__ == "__main__":
    vehicle_model = load_yolo_model(config.VEHICLE_DETECTION_MODEL)
    plate_model = load_yolo_model(config.LICENSE_PLATE_DETECTION_MODEL)
    ocr_reader = easyocr.Reader(['en'], gpu=False)
    process_video()
