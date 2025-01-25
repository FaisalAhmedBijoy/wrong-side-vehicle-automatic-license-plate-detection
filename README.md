# Real-Time YOLO License Plate Detection

This project demonstrates real-time vehicle and license plate detection using YOLO (You Only Look Once) and OpenCV. It processes videos to detect vehicles and license plates, and performs live tracking with bounding boxes and OCR (optical character recognition) for license plate text extraction. The output video is saved to disk, and live detection results are displayed in a resized OpenCV window.

---

## Features

- **YOLO-Based Detection**: Detects vehicles and license plates in uploaded or live videos.
- **Real-Time Processing**: Displays live output in a resized window for better performance.
- **Customizable FPS Reduction**: Allows users to reduce the number of processed frames for faster performance.
- **License Plate OCR**: Uses EasyOCR to extract text from detected license plates.
- **Video Saving**: Outputs the processed video with bounding boxes to disk.

---

## Prerequisites

Before running the application, ensure the following:

1. **Python Version**: Python 3.8 or higher.
2. **Dependencies**:
   - Install the required Python packages:
     ```bash
     pip install -r requirements.txt
     ```
   - Required libraries include:
     - `ultralytics`
     - `opencv-python`
     - `easyocr`
     - `gradio`
3. **Models**:
   - Place the YOLO vehicle detection model (`yolo11l.pt`) in the `models/` directory.
   - Place the license plate detection model (`license_plate_detector.pt`) in the `models/` directory.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/real-time-yolo-detection.git
   cd real-time-yolo-detection
   ```
