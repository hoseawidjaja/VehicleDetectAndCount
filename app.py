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

MODEL_PATH = "model_svm.pkl"

# ✅ PASTIKAN LAYOUT TETAP "wide" seperti sebelumnya
st.set_page_config(page_title="🚗 Vehicle Counter", layout="wide")

if not os.path_exists(MODEL_PATH):
    st.error(f"❌ Model `{MODEL_PATH}` not found.")
    st.stop()

@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

model_data = load_model()
svc, scaler, HOG_PARAMS = model_data['svc'], model_data['scaler'], model_data['params']

# ========= FUNGSI UTAMA (SAMA PERSIS SEPERTI ASLI) =========
def preprocess_image(img):
    if img is None: return None
    if img.ndim == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(cv2.GaussianBlur(img, (3, 3), 0))

def get_hog_features(img):
    if img is None: return None
    img = cv2.resize(preprocess_image(img), (64, 64))
    try:
        return hog(img, 
                   orientations=HOG_PARAMS['orientations'],
                   pixels_per_cell=HOG_PARAMS['pixels_per_cell'],
                   cells_per_block=HOG_PARAMS['cells_per_block'],
                   transform_sqrt=HOG_PARAMS['transform_sqrt'],
                   feature_vector=True,
                   channel_axis=None)
    except:
        return None

def detect_vehicles(frame, threshold=2.0):
    if frame is None: return []
    h, w = frame.shape[:2]
    boxes = []
    roi_y_start = int(h * 0.4)
    roi_y_end = h - 30
    for scale in [1.5, 2.0, 2.2]:
        bw, bh = int(64*scale), int(64*scale)
        for y in range(roi_y_start, roi_y_end-bh, 28):
            for x in range(0, w-bw, 28):
                if (x < 0.15*w or x > 0.85*w) and x % 56 != 0: continue
                win = frame[y:y+bh, x:x+bw]
                if win.shape[:2] != (bh, bw): continue
                feat = get_hog_features(win)
                if feat is None: continue
                conf = svc.decision_function(scaler.transform(feat.reshape(1,-1)))[0]
                if conf > threshold:
                    boxes.append([x, y, x+bw, y+bh, conf])
    return boxes

def nms(boxes, thresh=0.5):
    if len(boxes) == 0: return []
    boxes = np.array(boxes)
    x1, y1, x2, y2, s = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3], boxes[:,4]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(s)[::-1]
    pick = []
    while len(idxs) > 0:
        i = idxs[0]; pick.append(i)
        if len(idxs) == 1: break
        xx1 = np.maximum(x1[i], x1[idxs[1:]]); yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]]); yy2 = np.minimum(y2[i], y2[idxs[1:]])
        w = np.maximum(0, xx2-xx1+1); h = np.maximum(0, yy2-yy1+1)
        overlap = (w * h) / area[idxs[1:]]
        idxs = idxs[1:][overlap <= thresh]
    return boxes[pick].astype(int)

def merge_boxes(boxes, dist=50):
    if len(boxes) <= 1: return boxes
    merged, used = [], [False]*len(boxes)
    for i in range(len(boxes)):
        if used[i]: continue
        cluster, used[i] = [boxes[i]], True
        for j in range(i+1, len(boxes)):
            if used[j]: continue
            c1 = ((boxes[i][0]+boxes[i][2])/2, (boxes[i][1]+boxes[i][3])/2)
            c2 = ((boxes[j][0]+boxes[j][2])/2, (boxes[j][1]+boxes[j][3])/2)
            if math.hypot(c1[0]-c2[0], c1[1]-c2[1]) < dist:
                cluster.append(boxes[j]); used[j] = True
        merged.append(max(cluster, key=lambda x: x[4]))
    return np.array(merged)

class Tracker:
    def __init__(self):
        self.next_id = 0
        self.vehicles = {}
    
    def update(self, dets, line_y):
        centers = [(((x1+x2)//2, (y1+y2)//2), (x1,y1,x2,y2,c)) for x1,y1,x2,y2,c in dets]
        updated, counted = {}, []
        for (cx,cy), box in centers:
            matched = False
            for vid, data in self.vehicles.items():
                if data['history']:
                    lx, ly = data['history'][-1]
                    if math.hypot(cx-lx, cy-ly) < 50:
                        data['history'].append((cx,cy))
                        if len(data['history']) > 10: data['history'].pop(0)
                        data['center'], data['box'] = (cx,cy), box
                        if not data['counted'] and len(data['history']) >= 2:
                            py = data['history'][-2][1]
                            if (py < line_y <= cy or py > line_y >= cy) and abs(py-cy) > 5:
                                data['counted'] = True
                                counted.append({'id': vid, 'frame': frame_count, 'time': frame_count / fps})
                        updated[vid] = data
                        matched = True
                        break
            if not matched:
                self.vehicles[self.next_id] = {'center': (cx,cy), 'history': [(cx,cy)], 'box': box, 'counted': False}
                updated[self.next_id] = self.vehicles[self.next_id]
                self.next_id += 1
        self.vehicles = updated
        return counted

# ========= STREAMLIT =========
st.title("🚗 Vehicle Counter — Upload Mode")
st.markdown("Upload video → Get count & download result. *Preview via last frame.*")

if 'done' not in st.session_state:
    st.session_state.done = False
    st.session_state.results = []
    st.session_state.video_bytes = None
    st.session_state.last_frame = None

col1, col2 = st.columns([5,1])
with col2:
    if st.button("🔄 Reset"):
        st.session_state.clear()
        st.rerun()

uploaded = st.file_uploader("📤 Upload Video", type=["mp4", "avi"])

if uploaded and not st.session_state.done:
    with st.spinner("Processing..."):
        # Save input
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
            f.write(uploaded.read())
            in_path = f.name

        cap = cv2.VideoCapture(in_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w, h = int(cap.get(3)), int(cap.get(4))
        total = int(cap.get(7))

        # Output file
        fd, out_path = tempfile.mkstemp(suffix='.mp4')
        os.close(fd)
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        # Process
        tracker = Tracker()
        results = []
        line_y = h // 2
        frame_count = 0
        last_frame = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1

            # Process every 2nd frame
            if frame_count % 2 == 0:
                small = cv2.resize(frame, (w//2, h//2))
                raw = detect_vehicles(small, 2.0)
                if raw:
                    raw = [[x*2 for x in r[:4]] + [r[4]] for r in raw]
                    nms_boxes = nms(np.array(raw))
                    final = merge_boxes(nms_boxes)
                else:
                    final = []
                counted = tracker.update(final, line_y)
                results.extend(counted)

            # Draw on frame
            vis = frame.copy()
            cv2.line(vis, (0, line_y), (w, line_y), (0,255,255), 2)
            for vid, data in tracker.vehicles.items():
                if 'box' in data:
                    x1,y1,x2,y2,c = data['box']
                    color = (0,0,255) if data['counted'] else (0,255,0) if c > 3.0 else (0,165,255)
                    cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
            out.write(vis)
            last_frame = vis  # simpan frame terakhir untuk thumbnail

        cap.release()
        out.release()

        # Save outputs
        with open(out_path, 'rb') as f:
            video_bytes = f.read()
        
        # Cleanup
        for p in [in_path, out_path]:
            if os.path.exists(p): os.unlink(p)

        st.session_state.done = True
        st.session_state.results = results
        st.session_state.video_bytes = video_bytes
        st.session_state.last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB) if last_frame is not None else None

        st.success(f"✅ Done! {len(results)} vehicles counted.")

# Results
if st.session_state.done:
    st.header("✅ Results")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Last Frame (Preview)")
        if st.session_state.last_frame is not None:
            st.image(st.session_state.last_frame, use_container_width=True)
        else:
            st.info("No preview available.")
    
    with col2:
        st.subheader("📥 Download")
        st.download_button(
            "📥 Download Video",
            st.session_state.video_bytes,
            "vehicle_count_output.mp4",
            "video/mp4",
            use_container_width=True
        )
    
    st.subheader("📋 Counted Vehicles")
    if st.session_state.results:
        df = pd.DataFrame(st.session_state.results)
        if not df.empty:
            df['time_str'] = pd.to_timedelta(df['time'], unit='s').apply(lambda x: str(x).split('.')[0])
            df = df.rename(columns={'id': 'Vehicle ID', 'time_str': 'Time', 'frame': 'Frame #'})[['Vehicle ID', 'Time', 'Frame #']]
            st.dataframe(df, use_container_width=True)
            
            st.download_button(
                "💾 Save as CSV",
                df.to_csv(index=False).encode('utf-8'),
                "results.csv",
                "text/csv",
                use_container_width=True
            )
    else:
        st.info("No vehicles detected.")

