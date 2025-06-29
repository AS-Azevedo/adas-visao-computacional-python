# main.py

import cv2
import numpy as np
from ultralytics import YOLO

# Import custom modules
from lane_detector import detect_lanes
from tracker import Sort
from metrics_calculator import estimate_distance, calculate_speed_mps # Importa as novas funções

def main():
    # --- Load Models and Trackers ---
    print("Loading YOLOv8 model...")
    yolo_model = YOLO('yolov8n.pt')
    print("Model loaded.")
    tracker = Sort()
    
    # --- Video I/O ---
    video_path = 'assets/driving_video.mp4'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    time_delta_s = 1.0 / fps # Tempo em segundos entre cada frame

    display_width = 1280
    display_height = int(display_width * (frame_height / frame_width))

    window_name = 'ADAS Vision System'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_width, display_height)

    output_path = 'output_video.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print("Processing video... Press 'q' to exit.")

    # --- HISTÓRICO DE TRACKS PARA CÁLCULO DE VELOCIDADE ---
    # Dicionário para guardar o estado anterior de cada track
    # Formato: { track_id: {"dist": valor, "frame": número} }
    track_history = {}

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Finished processing video.")
            break
        
        frame_number += 1
        processed_frame = detect_lanes(frame)

        # --- Vehicle Detection and Tracking ---
        results = yolo_model(frame)
        detections = []
        VEHICLE_CLASSES = [2, 3, 5, 7]

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                if class_id in VEHICLE_CLASSES and confidence > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append([x1, y1, x2, y2])
        
        tracked_objects = tracker.update(np.array(detections))

        # --- CÁLCULO DE MÉTRICAS E VISUALIZAÇÃO ---
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, obj)
            
            # Estimar a distância atual
            box_width = x2 - x1
            current_dist_m = estimate_distance(box_width)
            
            speed_kmh = 0.0
            # Verificar se temos um histórico para esta track
            if track_id in track_history:
                prev_state = track_history[track_id]
                prev_dist_m = prev_state["dist"]
                prev_frame = prev_state["frame"]
                
                # Calcular a velocidade se a distância for válida e o tempo tiver passado
                if current_dist_m > 0 and prev_dist_m > 0 and frame_number > prev_frame:
                    time_diff_s = (frame_number - prev_frame) * time_delta_s
                    # O sinal negativo inverte para a nossa perspetiva (positivo = afastar-se)
                    speed_mps = calculate_speed_mps(prev_dist_m, current_dist_m, time_diff_s)
                    speed_kmh = speed_mps * 3.6 # Converter m/s para km/h
            
            # Atualizar o histórico com os dados atuais
            track_history[track_id] = {"dist": current_dist_m, "frame": frame_number}

            # Desenhar a caixa e a informação
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'ID: {track_id} | Vel: {speed_kmh:.1f} km/h'
            cv2.putText(processed_frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # --- Display and Save ---
        display_frame = cv2.resize(processed_frame, (display_width, display_height))
        cv2.imshow(window_name, display_frame)
        out.write(processed_frame)

        # --- Exit Condition ---
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Output video saved to {output_path}")

if __name__ == '__main__':
    main()