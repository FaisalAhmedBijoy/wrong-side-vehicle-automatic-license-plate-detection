import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    def __init__(self):
        self.VEHICLE_DETECTION_MODEL = self.get_required_env("VEHICLE_DETECTION_MODEL")
        self.LICENSE_PLATE_DETECTION_MODEL = self.get_required_env("LICENSE_PLATE_DETECTION_MODEL")
        self.INPUT_SAMPLE_VIDEO_PATH = self.get_required_env("INPUT_SAMPLE_VIDEO_PATH")
        self.YOLO_OUTPUT_VIDEO_PATH = self.get_required_env("YOLO_OUTPUT_VIDEO_PATH") 
        self.OUTPUT_RESULTS_CSV_PATH = self.get_required_env("OUTPUT_RESULTS_CSV_PATH")

        self.FPS_REDUCTION = self.get_required_env("FPS_REDUCTION") 
        self.LINE_Y_BLUE = self.get_required_env("LINE_Y_BLUE") 
        self.LINE_Y_YELLOW = self.get_required_env("LINE_Y_YELLOW")

        self.WINDOW_WIDTH = self.get_required_env("WINDOW_WIDTH") 
        self.WINDOW_HEIGHT = self.get_required_env("WINDOW_HEIGHT") 

        
    def get_required_env(self, env_variable):
        value = os.getenv(env_variable)
        if value is None:
            error_message = f"Invalid or missing '{env_variable}' in the environment variables"
            return error_message
        return value