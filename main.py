import gradio as gr
import cv2
from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict
import easyocr


# Load models
vehicle_model = YOLO("models/yolo11l.pt")  # Vehicle detection model
plate_model = YOLO("models/numer_plates_detection_model/license_plate_detector.pt")  # License plate detection model
ocr_reader = easyocr.Reader(['en'], gpu=False)  # OCR reader for license plate recognition

# Paths
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)  # Create output directory if it doesn't exist


def process_video_realtime(input_path: str, fps_reduction: int = 4):
    """
    Processes a video in real-time with YOLO, displaying live output in an OpenCV window.

    Args:
        input_path (str): Path to the input video file.
        fps_reduction (int): Reduction factor for processing video frames. Higher values process fewer frames.

    Returns:
        str: Path to the processed video saved to disk.
    """
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = output_dir / "output_video.mp4"

    # Define lines for vehicle direction detection
    line_y_blue = 240
    line_y_yellow = 200

    # Data structures for tracking
    object_status = defaultdict(lambda: {"yellow": False, "blue": False})
    direction_counts = {"right_direction": 0, "wrong_direction": 0}
    class_counts = defaultdict(lambda: {"right": 0, "wrong": 0})

    writer = None
    frame_count = 0
    frame_skip = max(1, fps // fps_reduction)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Process the frame for vehicle and license plate detection
        process_frame(
            frame,
            vehicle_model,
            plate_model,
            ocr_reader,
            line_y_blue,
            line_y_yellow,
            object_status,
            direction_counts,
            class_counts,
        )

        # Display the frame in real-time
        cv2.imshow("YOLO Real-Time Output", frame)

        # Initialize the video writer when processing the first frame
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

        writer.write(frame)

        # Press 'q' to quit real-time visualization
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    return str(output_path)


def process_frame(frame, vehicle_model, plate_model, ocr_reader, line_y_blue, line_y_yellow, object_status, direction_counts, class_counts):
    """
    Processes a single frame for vehicle and license plate detection.

    Args:
        frame: The current video frame to process.
        vehicle_model: YOLO model for vehicle detection.
        plate_model: YOLO model for license plate detection.
        ocr_reader: EasyOCR instance for license plate text recognition.
        line_y_blue (int): Y-coordinate of the blue line for vehicle direction detection.
        line_y_yellow (int): Y-coordinate of the yellow line for vehicle direction detection.
        object_status (dict): Dictionary tracking object crossings over lines.
        direction_counts (dict): Dictionary tracking counts of vehicles in the right or wrong direction.
        class_counts (dict): Dictionary tracking vehicle classes in each direction.

    Returns:
        None
    """
    results = vehicle_model.track(frame, persist=True, classes=[2, 3, 5, 7], conf=0.6, imgsz=640)

    if results[0].boxes.data is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        class_indices = results[0].boxes.cls.int().cpu().numpy().tolist()
        confidences = results[0].boxes.conf.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy().tolist()

        for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Detect crossings over yellow and blue lines
            if line_y_yellow - 10 <= cy <= line_y_yellow + 10:
                object_status[track_id]["yellow"] = True

            if line_y_blue - 10 <= cy <= line_y_blue + 10:
                object_status[track_id]["blue"] = True

            # Draw bounding box and label
            label = f"ID: {track_id}, Class: {vehicle_model.names[class_idx]}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


def gradio_interface(video_path: str, fps_reduction: int):
    """
    Gradio interface function to process the video.

    Args:
        video_path (str): Path to the uploaded video file.
        fps_reduction (int): FPS reduction factor chosen by the user.

    Returns:
        str: Path to the processed video saved to disk.
    """
    try:
        output_path = process_video_realtime(video_path, fps_reduction)
        return video_path, output_path
    except Exception as e:
        return f"Error: {e}", None


# Create Gradio interface
app = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Video(label="Upload Video"),
        gr.Slider(1, 10, step=1, value=4, label="FPS Reduction"),
    ],
    outputs=[
        gr.Video(label="Input Video"),
        gr.Video(label="Processed YOLO Output Video"),
    ],
    title="Real-Time YOLO License Plate Detection",
    description="Upload a video to detect vehicles and license plates in real-time. "
                "Live output will also be shown in a separate window.",
)

if __name__ == "__main__":
    app.launch(debug=True)
