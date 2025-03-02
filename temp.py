import cv2
import numpy as np

def preprocess_plate(plate_crop):
    """Apply preprocessing to enhance the license plate image for OCR."""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive Thresholding for better contrast
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Morphological operations to remove noise
        kernel = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Resize for better OCR performance
        processed_plate = cv2.resize(morph, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        return processed_plate
    
    except Exception as e:
        print(f"Error in plate preprocessing: {e}")
        return plate_crop
if __name__ == "__main__":
    plate_crop = cv2.imread("data/images/license_plate.png")
    cv2.imshow("Processed Plate", plate_crop)
    processed_plate = preprocess_plate(plate_crop)
    cv2.imshow("Processed Plate", processed_plate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()