import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_road_lines(image):
    # Load image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)
    # cv2.imshow("Edges", edges)
    # plt.imshow(edges, cmap="gray")
    # plt.axis("off")
    # plt.show()
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)
    
    # plt.imshow(cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2))
    # plt.axis("off")
    # plt.show()
    
    # Find the largest contour (assuming it's the road)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding box for the road
    x, y, w, h = cv2.boundingRect(largest_contour)
    # plt.imshow(cv2.rectangle(image.copy(), (x, y), (x + w, y + h), (0, 255, 0), 2))
    # plt.axis("off")
    # plt.show()
    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Define line positions dynamically
    line_y_blue = y + int(h * 0.8)  # Blue line at 80% of detected road height
    line_y_yellow = y + int(h * 0.6)  # Yellow line at 60% of detected road height
    
    # return line_y_blue, line_y_yellow, image
    return line_y_blue, line_y_yellow

def draw_lines(image):
    line_y_blue, line_y_yellow = detect_road_lines(image)
    height, width, _ = image.shape
    
    # Draw lines on image
    cv2.line(image, (0, line_y_blue), (width, line_y_blue), (255, 0, 0), 3)  # Blue line
    cv2.line(image, (0, line_y_yellow), (width, line_y_yellow), (0, 255, 255), 3)  # Yellow line
    
    # Show the result
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"Blue: {line_y_blue}, Yellow: {line_y_yellow}")
    plt.show()


if __name__ == "__main__":
    # Example usage
    road_images = ["data/images/road1.jpg", 
                "data/images/road2.jpg", 
                "data/images/road3.jpg", 
                "data/images/road_with_cars.png", 
                "data/images/multiple_road.png"]
    for image_path in road_images:
        image = cv2.imread(image_path)
        draw_lines(image)
