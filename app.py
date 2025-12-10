import streamlit as st
import cv2
import tempfile
import os
import numpy as np
import pandas as pd
import pickle
import math
from collections import deque
from skimage.feature import hog
import time

# ========== CONFIG & MODEL ==========
MODEL_PATH = "model_svm.pkl"

st.set_page_config(page_title="🚗 Vehicle Counter", layout="centered")

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model `{MODEL_PATH}` not found.")
    st.stop()

@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
        return data['svc'], data['scaler'], data['params']

svc, scaler, HOG_PARAMS = load_model()

# ========== CORE FUNCTIONS (tidak diubah) ==========
def preprocess_image(img):
    if img is None: return None
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    img_gray = cv2.equalizeHist(img_gray)
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
    return img_gray

def get_hog_features_detection(img):
    if img is None: return None
    img_proc = preprocess_image(img)
    img_resized = cv2.resize(img_proc, (64, 64))
    try:
        return hog(img_resized, **HOG_PARAMS, feature_vector=True, channel_axis=None)
    except:
        return None

def detect_vehicles_fast(frame, confidence_threshold=2.0):
    if frame is None: return []
    h, w = frame.shape[:2]
    all_boxes = []
    roi_y_start = int(h * 0.4)
    roi_y_end = h - 30
    scales = [1.5, 2.0, 2.2]
    step_size = 28

    for scale in scales:
        box_w, box_h = int(64 * scale), int(64 * scale)
        for y in range(roi_y_start, roi_y_end - box_h, step_size):
            for x in range(0, w - box_w, step_size):
                if (x < w * 0.15 or x > w * 0.85) and x % (step_size * 2) != 0:
                    continue
                window = frame[y:y+box_h, x:x+box_w]
                if window.shape[:2] != (box_h, box_w): continue
                feat = get_hog_features_detection(window)
                if feat is None: continue
                conf = svc.decision_function(scaler.transform(feat.reshape(1, -1)))[0]
                if conf > confidence_threshold:
                    all_boxes.append([x, y, x+box_w, y+box_h, conf])
    return all_boxes

def non_max_suppression_strong(boxes, overlap_thresh=0.5):
    if len(boxes) == 0: return []
    boxes = np.array(boxes)
    x1, y1, x2, y2, scores = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3], boxes[:,4]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)[::-1]
    pick = []
    while len(idxs) > 0:
        i = idxs[0]; pick.append(i)
        if len(idxs) == 1: break
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[1:]]
        idxs = idxs[1:][overlap <= overlap_thresh]
    return boxes[pick].astype("int")

def merge_nearby_boxes(boxes, threshold=50):
    if len(boxes) <= 1: return boxes
    boxes = boxes.tolist()
    merged, used = [], [False] * len(boxes)
    for i in range(len(boxes)):
        if used[i]: continue
        cluster, used[i] = [boxes[i]], True
        for j in range(i+1, len(boxes)):
            if used[j]: continue
            cx1 = (boxes[i][0] + boxes[i][2]) / 2
            cy1 = (boxes[i][1] + boxes[i][3]) / 2
            cx2 = (boxes[j][0] + boxes[j][2]) / 2
            cy2 = (boxes[j][1] + boxes[j][3]) / 2
            if math.sqrt((cx1-cx2)**2 + (cy1-cy2)**2) < threshold:
                cluster.append(boxes[j]); used[j] = True
        merged.append(max(cluster, key=lambda x: x[4]))
    return np.array(merged)

class VehicleTracker:
    def __init__(self):
        self.next_id = 0
        self.vehicles = {}
        self.counted_ids = set()
    
    def update(self, detections, line_y):
        current = [(((x1+x2)//2, (y1+y2)//2), (x1,y1,x2,y2,conf)) for x1,y1,x2,y2,conf in detections]
        updated, counted = {}, []
        for (cx,cy), box in current:
            matched, best_id = False, None
            for vid, data in self.vehicles.items():
                if data['history']:
                    lx, ly = data['history'][-1]
                    if math.sqrt((cx-lx)**2 + (cy-ly)**2) < 50:
                        data['history'].append((cx,cy))
                        if len(data['history']) > 10: data['history'].pop(0)
                        data['center'], data['box'] = (cx,cy), box
                        if not data['counted'] and len(data['history']) >= 2:
                            py = data['history'][-2][1]
                            if (py < line_y <= cy or py > line_y >= cy) and abs(py - cy) > 5:
                                data['counted'] = True
                                self.counted_ids.add(vid)
                                counted.append({'vehicle_id': vid, 'frame': frame_count_global[0], 'timestamp_sec': frame_count_global[0] / fps_global[0]})
                        updated[vid] = data
                        matched = True
                        break
            if not matched:
                self.vehicles[self.next_id] = {'center': (cx,cy), 'history': [(cx,cy)], 'box': box, 'counted': False}
                updated[self.next_id] = self.vehicles[self.next_id]
                self.next_id += 1
        self.vehicles = updated
        return counted

# ========== STREAMLIT ==========
st.title("🚗 Vehicle Counter")
st.caption("Upload video → Get count. Preview works in all browsers.")

if 'processed' not in st.session_state:
    st.session_state.processed = False
    st.session_state.counted = []
    st.session_state.video_bytes = None

if st.button("🔄 Reset"):
    st.session_state.clear()
    st.rerun()

uploaded = st.file_uploader("📤 Upload Video", type=["mp4", "avi"])

if uploaded and not st.session_state.processed:
    try:
        # Simpan input
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(uploaded.read())
            in_path = tmp.name

        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened(): raise ValueError("Cannot open video")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w, h = int(cap.get(3)), int(cap.get(4))
        total = int(cap.get(7))
        
        # ✅ 1. RESIZE OUTPUT KE 640x360
        out_w, out_h = 640, int(360 * h / w) // 2 * 2  # genap
        
        # ✅ 2. GUNAKAN CODEC 'MP4V' (KAPITAL!)
        fd, out_path = tempfile.mkstemp(suffix='.mp4')
        os.close(fd)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))
        
        if not out.isOpened(): raise RuntimeError("VideoWriter failed")

        # Process
        tracker = VehicleTracker()
        counted_list = []
        line_y = h // 2
        skip = 1
        frame_count = 0
        status = st.empty()
        bar = st.progress(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            frame_count_global = [frame_count]
            fps_global = [fps]
            
            status.text(f"Frame {frame_count}/{total} | Count: {len(counted_list)}")
            bar.progress(min(1.0, frame_count / total))

            # Resize frame to target size first
            frame_small = cv2.resize(frame, (w//2, h//2))
            raw = detect_vehicles_fast(frame_small, 2.0)
            
            final = []
            if raw:
                raw = [[x*2 for x in r[:4]] + [r[4]] for r in raw]
                nms = non_max_suppression_strong(np.array(raw))
                final = merge_nearby_boxes(nms)
            
            counted_now = tracker.update(final, line_y)
            counted_list.extend(counted_now)

            # Draw on original frame
            vis = frame.copy()
            cv2.line(vis, (0, line_y), (w, line_y), (0,255,255), 3)
            for vid, data in tracker.vehicles.items():
                if 'box' in data:
                    x1,y1,x2,y2,conf = data['box']
                    color = (0,0,255) if data['counted'] else (0,255,0) if conf > 3.0 else (0,165,255)
                    cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
                    cv2.circle(vis, data['center'], 4, (255,255,0), -1)
            
            # ✅ 3. RESIZE VIS KE OUTPUT SIZE SEBELUM TULIS
            vis_resized = cv2.resize(vis, (out_w, out_h))
            out.write(vis_resized)

        cap.release()
        out.release()

        # Baca file
        with open(out_path, "rb") as f:
            video_bytes = f.read()

        # Cleanup
        os.unlink(in_path)
        os.unlink(out_path)

        st.session_state.counted = counted_list
        st.session_state.video_bytes = video_bytes
        st.session_state.processed = True
        st.success(f"✅ Done! {len(counted_list)} vehicles counted.")

    except Exception as e:
        st.error(f"❌ {e}")

# Results
if st.session_state.processed:
    st.header("✅ Results")
    
    # ✅ 4. GUNAKAN st.video() LANGSUNG (BUKAN BASE64)
    st.video(st.session_state.video_bytes)
    
    st.download_button("📥 Download Video", st.session_state.video_bytes, "output.mp4", "video/mp4")
    
    if st.session_state.counted:
        df = pd.DataFrame(st.session_state.counted)
        df['time'] = pd.to_timedelta(df['timestamp_sec'], unit='s').apply(lambda x: str(x).split('.')[0])
        df = df[['vehicle_id', 'time', 'frame']].rename(columns={'vehicle_id':'ID','time':'Time','frame':'Frame'})
        st.dataframe(df, use_container_width=True)
        st.download_button("💾 Save CSV", df.to_csv(index=False).encode('utf-8'), "results.csv", "text/csv")
