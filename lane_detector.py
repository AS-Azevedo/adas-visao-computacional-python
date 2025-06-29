# lane_detector.py

import cv2
import numpy as np

last_left_fit = None
last_right_fit = None

def make_coordinates(image, line_parameters):
    try:
        slope, intercept = line_parameters
    except TypeError:
        return None
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    if abs(slope) < 1e-5:
        return None
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    global last_left_fit, last_right_fit
    left_fit = []
    right_fit = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            
            if abs(slope) < 0.5:
                continue
            
            # Lógica de inclinação corrigida: Esquerda (<0), Direita (>0)
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0) if left_fit else last_left_fit
    right_fit_average = np.average(right_fit, axis=0) if right_fit else last_right_fit
    
    last_left_fit = left_fit_average
    last_right_fit = right_fit_average
    
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    
    return [line for line in [left_line, right_line] if line is not None]

def detect_lanes(frame):
    lane_image = np.copy(frame)
    gray_frame = cv2.cvtColor(lane_image, cv2.COLOR_BGR2GRAY)
    blur_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    canny_edges = cv2.Canny(blur_frame, 50, 150)
    
    height, width = frame.shape[:2]
    
    # --- ROI AJUSTADA E CORRIGIDA ---
    roi_vertices = np.array([
        [ (width*0.2, height), (width*0.45, height*0.6), (width*0.55, height*0.6), (width*0.8, height) ]
    ], np.int32)
    
    mask = np.zeros_like(canny_edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    roi_edges = cv2.bitwise_and(canny_edges, mask)
    
    lines = cv2.HoughLinesP(
        roi_edges, rho=2, theta=np.pi/180, threshold=100, 
        minLineLength=40, maxLineGap=5
    )
    
    averaged_lines = average_slope_intercept(lane_image, lines)
    
    line_image = np.zeros_like(lane_image)
    if averaged_lines:
        for line in averaged_lines:
            x1, y1, x2, y2 = line
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
            
    final_image = cv2.addWeighted(frame, 0.8, line_image, 1.0, 0.0)
    
    return final_image