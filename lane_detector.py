# lane_detector.py

import cv2
import numpy as np

def detect_lanes(frame):
    """
    Processes a single frame to detect road lanes.
    
    Args:
        frame: The input video frame (as a NumPy array).
        
    Returns:
        An image with the detected lanes drawn on it.
    """
    # 1. Convert to Grayscale
    # Image analysis is simpler and faster on a single color channel.
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Apply Gaussian Blur (Smoothing)
    # This helps reduce image noise and improves edge detection.
    # The (5, 5) kernel is the size of the blurring window.
    blur_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # 3. Canny Edge Detection
    # Canny is a popular algorithm to find sharp changes in intensity (edges).
    # 50 and 150 are the lower and upper thresholds for hysteresis.
    canny_edges = cv2.Canny(blur_frame, 50, 150)

    # 4. Define a Region of Interest (ROI)
    # We don't need to scan the whole image (e.g., sky, trees).
    # We create a trapezoidal mask to focus only on the road area.
    height, width = frame.shape[:2]
    
    # These vertices define the trapezoid. They may need tuning for your video!
    roi_vertices = np.array([
        [(0, height)],
        [(width * 0.45, height * 0.6)],
        [(width * 0.55, height * 0.6)],
        [(width, height)]
    ], np.int32)
    
    # Create a black mask with the same dimensions as the canny image
    mask = np.zeros_like(canny_edges)
    
    # Fill the ROI polygon with white (255)
    cv2.fillPoly(mask, [roi_vertices], 255)
    
    # Keep only the edges that are within our ROI mask (bitwise AND operation)
    roi_edges = cv2.bitwise_and(canny_edges, mask)

    # 5. Hough Transform to Detect Lines
    # This technique finds lines in the edge-detected image.
    # Parameters may need fine-tuning, but these are a good starting point.
    lines = cv2.HoughLinesP(
        roi_edges,
        rho=2,              # Distance precision in pixels
        theta=np.pi/180,    # Angle precision in radians
        threshold=100,      # Minimum "votes" to be considered a line
        minLineLength=40,   # Minimum length of a line in pixels
        maxLineGap=5        # Max gap between segments to be joined into one line
    )

    # 6. Draw the detected lines on a blank image
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Draw a green line with a thickness of 10 pixels
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)

    # 7. Combine the original frame with the drawn lines
    # The final image is a weighted overlay of the original with the lines.
    final_image = cv2.addWeighted(frame, 0.8, line_image, 1.0, 0.0)
    
    return final_image

# --- Main execution block for testing this module ---
if __name__ == '__main__':
    video_path = 'assets/driving_video.mp4'  # MAKE SURE THIS FILE EXISTS
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file. Check the path.")
    else:
        while cap.isOpened():
            ret, frame = cap.read()
            
            # If the video has ended, 'ret' will be False
            if not ret:
                break

            # Process the current frame
            processed_frame = detect_lanes(frame)

            # Display the result
            cv2.imshow('Phase 1 - Lane Detection', processed_frame)

            # Press 'q' to exit the loop
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Processing finished.")