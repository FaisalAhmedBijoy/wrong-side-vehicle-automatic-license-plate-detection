import csv
import cv2
from ultralytics import YOLO
from collections import defaultdict

# Load YOLO model
def load_yolo_model(model_path='models/yolo11n.pt'):
    """Load the YOLO model from the specified path."""
    return YOLO(model_path)

# Resize the frame to fixed dimensions
def resize_frame(frame, width, height):
    """Resize the frame to the specified width and height."""
    return cv2.resize(frame, (width, height))

# Process function to handle frame-by-frame processing
def process_frame(frame, model, line_y_blue, line_y_yellow, object_status, direction_counts, csv_writer, class_counts):
    """
    Process a single frame to detect vehicles, track their directions, and update the counts.
    """
    results = model.track(frame, persist=True, classes=[2, 3, 5, 7])
    
    if results[0].boxes.data is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy().tolist()
        class_indices = results[0].boxes.cls.int().cpu().numpy().tolist()
        confidences = results[0].boxes.conf.cpu().numpy()

        for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
            x1, y1, x2, y2 = map(int, box)  # Extract bounding box coordinates
            cx = (x1 + x2) // 2  # Calculate center x-coordinate
            cy = (y1 + y2) // 2  # Calculate center y-coordinate
            class_name = model.names[class_idx]  # Get vehicle class name

            # Draw bounding box and track ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"{class_name} ID:{track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Track crossing status and determine direction
            if track_id not in object_status:
                object_status[track_id] = {"yellow": False, "blue": False}

            if cy >= line_y_yellow-10 and cy <= line_y_yellow+10 and not object_status[track_id]["yellow"]:
                object_status[track_id]["yellow"] = True  # Mark yellow line touched

            if cy >= line_y_blue-10 and cy<= line_y_blue+10 and not object_status[track_id]["blue"]:
                object_status[track_id]["blue"] = True  # Mark blue line touched

            # Determine direction based on line touch sequence
            if object_status[track_id]["yellow"] and not object_status[track_id]["blue"]:
                # If yellow line is touched first
                direction_counts["right_direction"] += 1
                class_counts[class_name]["right"] += 1
                csv_writer.writerow([track_id, class_name, "Right", conf])
                object_status[track_id]["blue"] = True  # Prevent re-counting

            elif object_status[track_id]["blue"] and not object_status[track_id]["yellow"]:
                # If blue line is touched first
                direction_counts["wrong_direction"] += 1
                class_counts[class_name]["wrong"] += 1
                csv_writer.writerow([track_id, class_name, "Wrong", conf])
                object_status[track_id]["yellow"] = True  # Prevent re-counting

    # Draw blue and yellow lines
    cv2.line(frame, (50, line_y_yellow), (frame.shape[1] - 50, line_y_yellow), (0, 255, 255), 3)
    cv2.line(frame, (50, line_y_blue), (frame.shape[1] - 50, line_y_blue), (255, 0, 0), 3)

    # Display the updated vehicle counts
    display_counts(frame, class_counts, direction_counts)


# Display counts on the left side of the frame
def display_counts(frame, class_counts, direction_counts):
    """
    Display the vehicle count statistics on the left side of the frame.
    """
    y_offset = 30
    for class_name, counts in class_counts.items():
        right_count = counts["right"]
        wrong_count = counts["wrong"]
        cv2.putText(frame, f"{class_name}: R={right_count} W={wrong_count}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30

    # Display the direction counts
    cv2.putText(frame, f"Right Direction: {direction_counts['right_direction']}", 
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y_offset += 30
    cv2.putText(frame, f"Wrong Direction: {direction_counts['wrong_direction']}", 
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# Main function to process the video
def process_video(video_path, model, line_y_blue, line_y_yellow, csv_file_path, width=1000, height=600):
    """
    Main function to process the video, track vehicles, and log their directions.
    """
    cap = cv2.VideoCapture(video_path)
    object_status = defaultdict(set)  # Track status of each vehicle by track ID
    direction_counts = {"right_direction": 0, "wrong_direction": 0}  # Initialize direction counts
    class_counts = defaultdict(lambda: {"right": 0, "wrong": 0})  # Initialize class-specific counts

    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Track ID', 'Vehicle Class', 'Direction'])  # Write CSV header

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = resize_frame(frame, width, height)  # Resize frame
            process_frame(frame, model, line_y_blue, line_y_yellow, object_status, direction_counts, csv_writer, class_counts)

            # Show the processed frame
            cv2.imshow("YOLO Object Tracking & Direction Detection", frame)

            # Exit loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()  # Release the video capture
    cv2.destroyAllWindows()  # Close all OpenCV windows

# Run the main function
if __name__ == "__main__":
    model = load_yolo_model()
    video_path = 'data/sample_video/input_video_2_15fps.mp4'  # Input video file
    csv_file_path = 'logs/processed_video/vehicle_counts.csv'  # Output CSV file for storing results
    # line_y_blue = 430  # Blue line y-coordinate
    # line_y_yellow = 380  # Yellow line y-coordinate
    line_y_blue = 230  # Blue line y-coordinate
    line_y_yellow = 200  # Yellow line y-coordinate
    process_video(video_path, model, line_y_blue, line_y_yellow, csv_file_path)
