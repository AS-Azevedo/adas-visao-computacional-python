# vehicle_detector.py

import cv2
from ultralytics import YOLO

# --- Constants for Distance Estimation (NEEDS CALIBRATION!) ---
# You need to calibrate this yourself. These are just example values.
KNOWN_CAR_WIDTH_PIXELS = 150 # Example: width of a car's bounding box at a known distance
KNOWN_DISTANCE_METERS = 15   # Example: the known distance in meters
AVG_CAR_WIDTH_REAL = 1.8     # Average car width in meters

# Calculate the focal length once
# Focal Length = (Pixel Width * Known Distance) / Real Width
FOCAL_LENGTH = (KNOWN_CAR_WIDTH_PIXELS * KNOWN_DISTANCE_METERS) / AVG_CAR_WIDTH_REAL

# This list contains the class IDs for vehicles in the COCO dataset, which YOLO was trained on.
# 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASSES = [2, 3, 5, 7]

def estimate_distance(box_width_pixels):
    """
    Estimates the distance to an object based on its bounding box width.
    This is an approximation and requires camera calibration.
    """
    if box_width_pixels == 0:
        return -1 # Avoid division by zero
    
    # Distance = (Real Width * Focal Length) / Pixel Width
    distance = (AVG_CAR_WIDTH_REAL * FOCAL_LENGTH) / box_width_pixels
    return distance

def detect_vehicles_yolo(frame, model):
    """
    Performs vehicle detection on a single frame using a YOLOv8 model.

    Args:
        frame: The input video frame.
        model: The pre-loaded YOLOv8 model object.

    Returns:
        The frame annotated with bounding boxes and distance information.
    """
    # Run inference on the frame
    # The model returns a list of Results objects
    results = model(frame)

    # Process results list
    for result in results:
        # Get the bounding boxes from the result object
        boxes = result.boxes

        for box in boxes:
            # Get the class ID
            class_id = int(box.cls[0])

            # Check if the detected object is a vehicle
            if class_id in VEHICLE_CLASSES:
                # Get coordinates of the bounding box (xyxy format)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get the confidence score
                confidence = float(box.conf[0])
                
                # Filter out low-confidence detections
                if confidence > 0.5:
                    # Get the class name from the model's names list
                    class_name = model.names[class_id]
                    
                    # Estimate the distance
                    box_width = x2 - x1
                    dist = estimate_distance(box_width)
                    
                    # --- Draw on the frame ---
                    # Draw the bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # Red box
                    
                    # Create the label text
                    label = f'{class_name} {confidence:.2f} | Dist: {dist:.1f}m'
                    
                    # Put the label above the bounding box
                    cv2.putText(frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return frame


# --- Main execution block for testing this module independently ---
if __name__ == '__main__':
    # Load the YOLOv8 model. 'yolov8n.pt' is the smallest and fastest version.
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')
    print("Model loaded.")

    video_path = 'assets/driving_video.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
    else:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process the frame for vehicles
            processed_frame = detect_vehicles_yolo(frame, model)

            # Display the result
            cv2.imshow("YOLOv8 Vehicle Detection Test", processed_frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()