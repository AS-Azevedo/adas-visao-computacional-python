# tracker.py

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

def iou(bbox1, bbox2):
    x1_inter = max(bbox1[0], bbox2[0])
    y1_inter = max(bbox1[1], bbox2[1])
    x2_inter = min(bbox1[2], bbox2[2])
    y2_inter = min(bbox1[3], bbox2[3])
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = bbox1_area + bbox2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

class Track:
    def __init__(self, track_id, initial_bbox):
        self.track_id = track_id
        self.bbox = initial_bbox
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0], [0,1,0,0,0,1,0], [0,0,1,0,0,0,1], [0,0,0,1,0,0,0], [0,0,0,0,1,0,0], [0,0,0,0,0,1,0], [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0], [0,1,0,0,0,0,0], [0,0,1,0,0,0,0], [0,0,0,1,0,0,0]])
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.kf.x[:4] = self.convert_bbox_to_z(initial_bbox)

    def convert_bbox_to_z(self, bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        s = w * h
        r = w / float(h) if h != 0 else 0
        return np.array([x, y, s, r]).reshape((4, 1))

    def convert_x_to_bbox(self, x):
        h = np.sqrt(x[2] / x[3]) if x[3] != 0 else 0
        w = h * x[3]
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))

    def predict(self):
        self.kf.predict()
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        self.age += 1
        self.time_since_update += 1
        self.bbox = self.convert_x_to_bbox(self.kf.x)[0]
        return self.bbox

    def update(self, bbox):
        """
        Atualiza o vetor de estado com a caixa delimitadora (bbox) observada.
        """
        self.time_since_update = 0
        self.hits += 1
        self.kf.update(self.convert_bbox_to_z(bbox))
        self.bbox = self.convert_x_to_bbox(self.kf.x)[0]


class Sort:
    def __init__(self, max_age=5, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.frame_count = 0
        self.next_id = 1

    def update(self, detections):
        self.frame_count += 1
        
        valid_tracks = []
        predicted_bboxes = []
        for track in self.tracks:
            predicted_bbox = track.predict()
            if not np.any(np.isnan(predicted_bbox)):
                valid_tracks.append(track)
                predicted_bboxes.append(predicted_bbox)

        matched_indices = []
        if len(valid_tracks) > 0 and len(detections) > 0:
            iou_matrix = np.zeros((len(detections), len(predicted_bboxes)), dtype=np.float32)
            for d, det in enumerate(detections):
                for t, trk in enumerate(predicted_bboxes):
                    iou_matrix[d, t] = iou(det, trk)
            
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    matched_indices.append((r, c))
        
        unmatched_detections_idx = set(range(len(detections))) - set([r for r,c in matched_indices])
        
        for r, c in matched_indices:
            track_to_update = valid_tracks[c]
            track_to_update.update(detections[r])

        for i in unmatched_detections_idx:
            det = detections[i]
            self.tracks.append(Track(self.next_id, det))
            self.next_id += 1
            
        result = []
        for track in self.tracks:
            if track.time_since_update < 1 and (track.hits >= self.min_hits or self.frame_count <= self.min_hits):
                result.append(np.concatenate((track.bbox, [track.track_id])).tolist())

        self.tracks = [track for track in self.tracks if track.time_since_update < self.max_age]
        
        return np.array(result)