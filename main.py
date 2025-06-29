# main.py

import cv2
from ultralytics import YOLO

# Import functions from our custom modules
from lane_detector import detect_lanes
from vehicle_detector import detect_vehicles_yolo

def main():
    # --- Load Models Once ---
    print("Loading YOLOv8 model...")
    yolo_model = YOLO('yolov8n.pt')
    print("Model loaded successfully.")

    # --- Video I/O ---
    video_path = 'assets/driving_video.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    # Get original video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # --- CORREÇÃO 2: DEFINIR TAMANHO DA JANELA ---
    # Define a new, larger size for the display window.
    # You can change these values to fit your screen.
    display_width = 1280
    display_height = int(display_width * (frame_height / frame_width)) # Maintain aspect ratio

    # --- CORREÇÃO 2: CRIAR JANELA REDIMENSIONÁVEL ---
    # Create a named, resizable window before the loop
    window_name = 'ADAS Vision System'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_width, display_height)


    # Define the codec and create VideoWriter object for saving the output
    output_path = 'output_video.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # The output video will be saved with the original frame dimensions
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print("Processing video... Press 'q' or close the window to exit.")

    # --- Main Processing Loop ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Finished processing video.")
            break

        # --- Apply ADAS Functions ---
        processed_frame = detect_lanes(frame)
        processed_frame = detect_vehicles_yolo(processed_frame, yolo_model)
        
        # --- CORREÇÃO 2: REDIMENSIONAR O FRAME PARA EXIBIÇÃO ---
        # Resize the final frame to our desired display size
        display_frame = cv2.resize(processed_frame, (display_width, display_height))

        # --- Display and Save ---
        cv2.imshow(window_name, display_frame)
        out.write(processed_frame) # Save the original size frame

        # --- CORREÇÃO 1: CONDIÇÕES DE SAÍDA ---
        # 1. Check for 'q' key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("'q' key pressed. Exiting...")
            break
        
        # 2. Check if the user closed the window using the 'X' button
        # This checks if the window property is still valid/visible.
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed by user. Exiting...")
            break

    # --- Release Resources ---
    print("Releasing resources...")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Output video saved to {output_path}")

if __name__ == '__main__':
    main()