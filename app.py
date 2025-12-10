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
import base64
import streamlit.components.v1 as components

MODEL_PATH = "model_svm.pkl"

st.set_page_config(page_title="🚗 Vehicle Counter (Upload)", layout="wide")

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file `{MODEL_PATH}` not found in current directory.")
    st.stop()

@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
        return data['svc'], data['scaler'], data['params']

try:
    svc, scaler, HOG_PARAMS = load_model()
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

def preprocess_image(img):
    if img is None:
        return None
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    img_gray = cv2.equalizeHist(img_gray)
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
    return img_gray

def get_hog_features_detection(img):
    if img is None:
        return None
    img_processed = preprocess_image(img)
    img_resized = cv2.resize(img_processed, (64, 64))
    try:
        features = hog(img_resized,
                       orientations=HOG_PARAMS['orientations'],
                       pixels_per_cell=HOG_PARAMS['pixels_per_cell'],
                       cells_per_block=HOG_PARAMS['cells_per_block'],
                       transform_sqrt=HOG_PARAMS['transform_sqrt'],
                       feature_vector=True,
                       channel_axis=None)
        return features
    except:
        return None

def detect_vehicles_fast(frame, confidence_threshold=2.0):
    if frame is None:
        return []
    height, width = frame.shape[:2]
    all_boxes = []
    roi_y_start = int(height * 0.4)
    roi_y_end = height - 30
    scales = [1.5, 2.0, 2.2]
    step_size = 28

    for scale in scales:
        w = int(64 * scale)
        h = int(64 * scale)
        for y in range(roi_y_start, roi_y_end - h, step_size):
            for x in range(0, width - w, step_size):
                if (x < width * 0.15 or x > width * 0.85) and x % (step_size * 2) != 0:
                    continue
                window = frame[y:y+h, x:x+w]
                if window.shape[0] != h or window.shape[1] != w:
                    continue
                features = get_hog_features_detection(window)
                if features is None:
                    continue
                scaled = scaler.transform(features.reshape(1, -1))
                conf = svc.decision_function(scaled)[0]
                if conf > confidence_threshold:
                    all_boxes.append([x, y, x+w, y+h, conf])
    return all_boxes

def non_max_suppression_strong(boxes, overlap_thresh=0.5):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    if len(boxes) == 0:
        return []
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)[::-1]
    while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)
        if len(idxs) == 1:
            break
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
    if len(boxes) <= 1:
        return boxes
    boxes = boxes.tolist()
    merged = []
    used = [False] * len(boxes)
    for i in range(len(boxes)):
        if used[i]:
            continue
        current = boxes[i]
        cluster = [current]
        used[i] = True
        for j in range(i+1, len(boxes)):
            if used[j]:
                continue
            cx1 = (current[0] + current[2]) / 2
            cy1 = (current[1] + current[3]) / 2
            cx2 = (boxes[j][0] + boxes[j][2]) / 2
            cy2 = (boxes[j][1] + boxes[j][3]) / 2
            distance = math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
            if distance < threshold:
                cluster.append(boxes[j])
                used[j] = True
        if cluster:
            best_box = max(cluster, key=lambda x: x[4])
            merged.append(best_box)
    return np.array(merged)

class VehicleTracker:
    def __init__(self):
        self.next_id = 0
        self.vehicles = {}
        self.counted_ids = set()
    
    def update(self, detections, line_y):
        current_centers = []
        for box in detections:
            x1, y1, x2, y2, conf = box
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            current_centers.append((center, box))
        
        updated_vehicles = {}
        counted_now = []
        
        for (center, box) in current_centers:
            cx, cy = center
            matched = False
            for vid, data in self.vehicles.items():
                last_center = data['history'][-1] if data['history'] else None
                if last_center:
                    dist = math.sqrt((cx - last_center[0])**2 + (cy - last_center[1])**2)
                    if dist < 50:
                        data['history'].append(center)
                        if len(data['history']) > 10:
                            data['history'].pop(0)
                        data['center'] = center
                        data['box'] = box
                        if not data['counted'] and len(data['history']) >= 2:
                            prev_y = data['history'][-2][1]
                            curr_y = cy
                            if (prev_y < line_y <= curr_y or prev_y > line_y >= curr_y) and abs(prev_y - curr_y) > 5:
                                data['counted'] = True
                                self.counted_ids.add(vid)
                                counted_now.append({
                                    'vehicle_id': vid,
                                    'frame': frame_count_global[0],
                                    'timestamp_sec': frame_count_global[0] / fps_global[0] if fps_global[0] > 0 else 0
                                })
                        updated_vehicles[vid] = data
                        matched = True
                        break
            if not matched:
                self.vehicles[self.next_id] = {
                    'center': center,
                    'history': [center],
                    'box': box,
                    'counted': False
                }
                updated_vehicles[self.next_id] = self.vehicles[self.next_id]
                self.next_id += 1
        self.vehicles = updated_vehicles
        return counted_now



st.title("🚗 Optimized Vehicle Counter — Upload Video")
st.markdown("Powered by **HOG + SVM + Path-Based Line Crossing** — *no flickering, full frame output*")

if 'processed' not in st.session_state:
    st.session_state.processed = False
    st.session_state.counted_vehicles = []
    st.session_state.video_bytes = None

col1, col2 = st.columns([6, 1])
with col2:
    if st.button("🔄 Reset", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

st.header("📤 1. Upload Video")
uploaded_video = st.file_uploader("Choose a video file (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

if uploaded_video and not st.session_state.processed:
    try:
        # ✅ 1. Simpan input ke tempfile (aman di cloud)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_in:
            tmp_in.write(uploaded_video.read())
            video_in_path = tmp_in.name

        cap = cv2.VideoCapture(video_in_path)
        if not cap.isOpened():
            raise ValueError("Cannot open uploaded video")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_global = [fps]

        st.info(f"🎥 {width}×{height} @ {fps:.1f} FPS | {total_frames} frames total")

        # ✅ 2. Buat output file di /tmp via mkstemp (100% cloud-safe)
        fd_out, out_file = tempfile.mkstemp(suffix='.mp4')
        os.close(fd_out)  # bebaskan file descriptor

        # ✅ 3. GUNAKAN 'mp4v' — satu-satunya codec yang pasti jalan di Streamlit Cloud
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_file, fourcc, fps, (width, height))

        # ✅ 4. Pastikan writer terbuka
        if not out.isOpened():
            raise RuntimeError("VideoWriter failed to open. Codec 'mp4v' may not be available.")

        tracker = VehicleTracker()
        counted_list = []
        line_y = height // 2
        skip_frames = 1
        frame_count = 0
        processed_frames = 0

        start_time = time.time()

        status_text = st.empty()
        progress_bar = st.progress(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame_count_global = [frame_count]

            elapsed = time.time() - start_time
            percent = (frame_count / total_frames) * 100
            status_text.text(f"Processing frame {frame_count}/{total_frames} ({percent:.1f}%) | ⏱️ {int(elapsed)}s")
            progress_bar.progress(min(1.0, frame_count / total_frames))

            if frame_count % (skip_frames + 1) != 0:
                display_frame = frame.copy()
                cv2.line(display_frame, (0, line_y), (width, line_y), (0, 255, 255), 3)
                cv2.putText(display_frame, f"TOTAL: {len(counted_list)}", (width - 220, 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                out.write(display_frame)
                continue

            processed_frames += 1
            small_frame = cv2.resize(frame, (width // 2, height // 2))
            raw_boxes = detect_vehicles_fast(small_frame, confidence_threshold=2.0)

            final_boxes = []
            if len(raw_boxes) > 0:
                raw_boxes = [[x * 2 for x in box[:4]] + [box[4]] for box in raw_boxes]
                boxes_nms = non_max_suppression_strong(np.array(raw_boxes), overlap_thresh=0.5)
                final_boxes = merge_nearby_boxes(boxes_nms, threshold=60)

            counted_now = tracker.update(final_boxes, line_y)
            counted_list.extend(counted_now)

            vis = frame.copy()

            cv2.line(vis, (0, line_y), (width, line_y), (0, 255, 255), 3)
            cv2.putText(vis, "COUNTING LINE", (width - 200, line_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            for vid, data in tracker.vehicles.items():
                if 'box' not in 
                    continue
                x1, y1, x2, y2, conf = data['box']
                if data['counted']:
                    color = (0, 0, 255)
                    thickness = 3
                elif conf > 3.0:
                    color = (0, 255, 0)
                    thickness = 2
                else:
                    color = (0, 165, 255)
                    thickness = 2
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
                cv2.circle(vis, data['center'], 5, (255, 255, 0), -1)
                cv2.putText(vis, f"ID:{vid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                if data['counted']:
                    cv2.putText(vis, "COUNTED", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            cv2.rectangle(vis, (10, 10), (380, 110), (0, 0, 0), -1)
            cv2.putText(vis, f"Frame: {frame_count}/{total_frames}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(vis, f"Counted: {len(counted_list)}", (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis, f"Active: {len(tracker.vehicles)}", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

            out.write(vis)

        # ✅ 5. RELEASE & FLUSH
        cap.release()
        out.release()

        # ✅ 6. Baca file SEKARANG — jangan hapus dulu
        if not os.path.exists(out_file):
            raise FileNotFoundError(f"Output file not created: {out_file}")
        if os.path.getsize(out_file) == 0:
            raise ValueError("Output video is 0 bytes — check frame writing.")

        with open(out_file, "rb") as f:
            video_bytes = f.read()

        # ✅ 7. Baru hapus file sementara
        os.unlink(video_in_path)
        os.unlink(out_file)

        # Simpan bytes ke session (bukan path)
        st.session_state.counted_vehicles = counted_list
        st.session_state.video_bytes = video_bytes
        st.session_state.processed = True

        st.success(f"✅ Done! {frame_count} frames, {len(counted_list)} vehicles.")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)
        # Cleanup
        for f in [video_in_path, out_file]:
            if 'f' in locals() and os.path.exists(f):
                os.unlink(f)
        st.stop()

# === RESULTS ===
if st.session_state.processed:
    st.header("✅ 2. Results")

    video_bytes = st.session_state.video_bytes

    # Tampilkan info ukuran
    st.info(f"📹 Video size: {len(video_bytes) // 1024} KB")

    # Preview hanya jika ukuran masuk akal (>1 KB)
    if len(video_bytes) > 1024:
        st.subheader("📹 Output Video")
        video_height = 480
        video_b64 = base64.b64encode(video_bytes).decode()
        video_html = f"""
        <div style="display: flex; justify-content: center; margin: 1rem 0;">
        <video controls style="max-width: 100%; height: {video_height}px; border-radius: 8px;">
            <source src="video/mp4;base64,{video_b64}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        </div>
        """
        components.html(video_html, height=video_height + 60)
    else:
        st.warning("⚠️ Video too small to preview — try downloading.")

    # Download selalu aman
    st.download_button(
        "📥 Download Video",
        video_bytes,
        file_name="vehicle_count_output.mp4",
        mime="video/mp4"
    )

    st.subheader("📋 Counted Vehicles")
    if st.session_state.counted_vehicles:
        df = pd.DataFrame(st.session_state.counted_vehicles)
        df['time_formatted'] = pd.to_timedelta(df['timestamp_sec'], unit='s').apply(
            lambda x: str(x).split('.')[0]
        )
        df = df[['vehicle_id', 'time_formatted', 'frame']].rename(columns={
            'vehicle_id': 'Vehicle ID',
            'time_formatted': 'Time (H:MM:SS)',
            'frame': 'Frame #'
        }).reset_index(drop=True)
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "💾 Save as CSV",
            csv,
            file_name="vehicle_count_results.csv",
            mime="text/csv"
        )
    else:
        st.info("No vehicles were counted.")

st.markdown("---")
st.caption("✅ Cloud-safe: uses 'mp4v' codec + tempfile.mkstemp() + byte storage")
