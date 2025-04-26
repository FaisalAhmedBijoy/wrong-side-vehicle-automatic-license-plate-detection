import cv2
import easyocr
import pandas as pd
import numpy as np

from ultralytics import YOLO
from collections import defaultdict

from app.configurations import Config
from app.road_line_draw_algo import detect_road_lines

config = Config()
line_y_blue_list = []
line_y_yellow_list = []

def load_yolo_model(model_path):
    """Load the YOLO model from the specified path."""
    return YOLO(model_path)


def preprocess_plate(plate_crop):
    """Apply preprocessing to enhance the license plate image for OCR."""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        return gray

    except Exception as e:
        print(f"Error in plate preprocessing: {e}")
        return plate_crop


def detect_and_recognize_number_plate(frame,vehicle_crop, plate_model, ocr_reader):
    """Detect and recognize the number plate in the given vehicle crop."""
    try:
        plate_results = plate_model.predict(vehicle_crop, conf=0.5)
        if plate_results[0].boxes.data is not None and len(plate_results[0].boxes.xyxy) > 0:
            plate_box = plate_results[0].boxes.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = plate_box

            print('plate_box: ',plate_box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            # cv2.putText(frame, f"Plate Box: ", (x1-10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

            plate_crop = vehicle_crop[y1:y2, x1:x2]
            processed_plate = preprocess_plate(plate_crop)  # Apply image processing
            
            ocr_results = ocr_reader.readtext(processed_plate, detail=0)
            cv2.imshow("License Plate Box", vehicle_crop)
            cv2.imshow(f"Results", processed_plate)

            return " ".join(ocr_results)
        return "No Plate Detected"
    except Exception as e:
        print(f"Error detecting/recognizing plate: {e}")
        return "Error"

def process_frame(frame, 
                  vehicle_model, 
                  plate_model, 
                  ocr_reader, 
                  line_y_blue, 
                  line_y_yellow, 
                  object_status, 
                  direction_counts, 
                  results_df, 
                  class_counts):
    """Process a single frame to detect vehicles, count directions, and recognize plates."""
    vehicle_results = vehicle_model.track(frame, 
                                          persist=True,                    
                                          conf=0.6, 
                                          imgsz=640)
    if vehicle_results[0].boxes.data is not None:
        boxes = vehicle_results[0].boxes.xyxy.cpu().numpy()
        class_indices = vehicle_results[0].boxes.cls.int().cpu().numpy().tolist()
        confidences = vehicle_results[0].boxes.conf.cpu().numpy()
        track_ids = vehicle_results[0].boxes.id.int().cpu().numpy().tolist()

        for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            class_name = vehicle_model.names[class_idx]

            if track_id is None:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"{class_name} ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if track_id not in object_status:
                object_status[track_id] = {"yellow": False, "blue": False}

            if line_y_yellow - 10 <= cy <= line_y_yellow + 10 and not object_status[track_id]["yellow"]:
                object_status[track_id]["yellow"] = True

            if line_y_blue - 10 <= cy <= line_y_blue + 10 and not object_status[track_id]["blue"]:
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
                
                number_plate = detect_and_recognize_number_plate(frame, vehicle_crop, plate_model, ocr_reader)

                # Append results to the DataFrame
                results_df.append({
                    'Track ID': track_id,
                    'Vehicle Class': class_name,
                    'Direction': direction,
                    'Confidence': conf,
                    'Number Plate': number_plate
                })

        # cv2.putText(frame, f"Right: {direction_counts['right_direction']}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # cv2.putText(frame, f"Wrong: {direction_counts['wrong_direction']}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.line(frame, (50, line_y_yellow), (frame.shape[1] - 50, line_y_yellow), (0, 255, 255), 3)
    cv2.line(frame, (50, line_y_blue), (frame.shape[1] - 50, line_y_blue), (255, 0, 0), 3)

def save_video(output_path, frame, frame_width, frame_height, fps, writer=None):
    """Save the processed video frames to an output file."""
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    writer.write(frame)
    return writer

def process_video(video_path, 
                  vehicle_model, 
                  plate_model, 
                  ocr_reader, 
                  line_y_blue, 
                  line_y_yellow,
                  window_width,
                  window_height, 
                  output_video_path,
                  output_results_csv_path, 
                  fps_reduction=1):
    """Process the input video for vehicle detection, direction analysis, and plate recognition."""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("frame_width: ",frame_width,"frame_height: ", frame_height)

    object_status = defaultdict(lambda: {"yellow": False, "blue": False})
    direction_counts = {"right_direction": 0, "wrong_direction": 0}
    class_counts = defaultdict(lambda: {"right": 0, "wrong": 0})

    results_df = []

    writer = None
    frame_count = 0

    print('fps: ',fps)
    print('fps_reduction: ',fps_reduction)
    frame_skip = fps // fps_reduction

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue
        
        frame = cv2.resize(frame, (window_width, window_height))

        # Process the line
        line_y_blue, line_y_yellow = detect_road_lines(frame)
        line_y_blue_list.append(line_y_blue)
        line_y_yellow_list.append(line_y_yellow)
        # make average of line_y_blue and line_y_yellow
        line_y_blue = int(np.mean(line_y_blue_list))
        line_y_yellow = int(np.mean(line_y_yellow_list))
     
        print('by algo line_y_blue: ',line_y_blue, 'by algo line_y_yellow: ',line_y_yellow)

        process_frame(frame, vehicle_model, plate_model, ocr_reader, line_y_blue, line_y_yellow, object_status, direction_counts, results_df, class_counts)
        writer = save_video(output_video_path, frame, window_width, window_height, fps, writer)

        cv2.imshow("Processed Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer is not None:
        writer.release()  
    cv2.destroyAllWindows()

    # Convert results to a Pandas DataFrame and save
    results_df = pd.DataFrame(results_df)
    results_df.to_csv(output_results_csv_path, index=False)

if __name__ == "__main__":
    vehicle_detection_model = load_yolo_model(model_path=config.VEHICLE_DETECTION_MODEL)
    license_plate_model = load_yolo_model(model_path=config.LICENSE_PLATE_DETECTION_MODEL)
    ocr_reader = easyocr.Reader(['en'], gpu=False)

    input_sample_video_path = config.INPUT_SAMPLE_VIDEO_PATH
    YOLO_output_video_path = config.YOLO_OUTPUT_VIDEO_PATH

    line_y_blue = int(config.LINE_Y_BLUE)
    line_y_yellow = int(config.LINE_Y_YELLOW)
    fps_reduction = int(config.FPS_REDUCTION)

    output_results_csv_path = config.OUTPUT_RESULTS_CSV_PATH

    window_width = int(config.WINDOW_WIDTH)
    window_height = int(config.WINDOW_HEIGHT)

    process_video(input_sample_video_path, 
                  vehicle_detection_model, 
                  license_plate_model, 
                  ocr_reader, 
                  line_y_blue, 
                  line_y_yellow, 
                  window_width,
                  window_height,
                  YOLO_output_video_path, 
                  output_results_csv_path, 
                  fps_reduction=fps_reduction)
