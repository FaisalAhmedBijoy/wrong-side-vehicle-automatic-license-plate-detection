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
     - `FastAPI`
     - `gradio`

---

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/real-time-yolo-detection.git
   cd real-time-yolo-detection
   ```

2. **Run Vehicle Detection Without GUI**:

   ```bash
      python -m app.license_plate_detection
   ```

   Output video will be saved in the `app/outputs` folder and csv be saved in the `app/outputs` folder.

3. **Run Vehicle Detection With GUI**:

   Run the following command to start the FastAPI server and access the web interface:

   ```bash
      python -m app.main
   ```

   Open your web browser and navigate to `http://localhost:8000` to access the application.

   Check the API documentation at `http://localhost:8000/docs` for more details on the available endpoints.

   ![](logs/readne_images/Screenshot%202025-04-26%20222211.png)

   Another approach is open the `webpage.html` file in your browser to access the web interface.
   The web interface allows you to upload a video file

   The processed video will be saved in the `app/results` folder and the csv file will be saved in the `app/results` folder.

   Server Response:

   ```json
   {
     "message": "Processing complete",
     "output_video": "app/results/processed_a15ef89a-8907-4089-9ec5-d9643946e45e.mp4",
     "results_csv": "app/results/output_2569bc76-ee74-4d5f-9681-51ab09f58ea9.csv"
   }
   ```
