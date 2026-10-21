import streamlit as st
import cv2
import numpy as np
from PIL import Image
from collections import Counter
from ultralytics import YOLO

st.set_page_config(page_title="Smart Helmet Detection", layout="wide")
st.title("🏍️ Smart Helmet Compliance Tracker")
st.markdown("Upload a traffic image to detect riders, bikes, and helmet compliance instantly.")

# లోకల్ ఫోల్డర్ లో ఉన్న బెస్ట్ మోడల్ ని లోడ్ చేయడం
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error("Model లోడ్ అవ్వలేదు. 'best.pt' ఫైల్ app.py ఉన్న ఫోల్డర్ లోనే ఉందో లేదో చెక్ చేయండి.")

def compute_ior(small_box, large_box):
    xA = max(small_box[0], large_box[0])
    yA = max(small_box[1], large_box[1])
    xB = min(small_box[2], large_box[2])
    yB = min(small_box[3], large_box[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    small_area = max(1e-5, (small_box[2] - small_box[0]) * (small_box[3] - small_box[1]))
    return inter_area / small_area

def detect_and_count_optimized(orig_img, conf_thresh=0.30, iou_thresh=0.45):
    h, w = orig_img.shape[:2]
    target_h = 600
    scale_factor = target_h / h
    target_w = int(w * scale_factor)
    img = cv2.resize(orig_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    results = model.predict(source=img, conf=conf_thresh, iou=iou_thresh, agnostic_nms=False, verbose=False)[0]

    bikes, riders_raw, helmets_raw = [], [], []

    for box in results.boxes:
        cls_id, conf = int(box.cls[0]), float(box.conf[0])
        coords = box.xyxy[0].cpu().numpy().astype(int)
        if cls_id == 1: bikes.append((coords, conf))
        elif cls_id in [2, 4]: riders_raw.append((coords, cls_id, conf))
        elif cls_id in [0, 3, 5, 6]: helmets_raw.append((coords, cls_id, conf))

    valid_riders = []
    for r_box, r_cls, r_conf in riders_raw:
        rx1, ry1, rx2, ry2 = r_box
        seating_zone = [rx1, int(ry1 + 0.3 * (ry2 - ry1)), rx2, ry2]
        has_bike = len(bikes) == 0
        if not has_bike:
            for b_box, _ in bikes:
                bx1, by1, bx2, by2 = b_box
                ix1, iy1 = max(seating_zone[0], bx1), max(seating_zone[1], by1)
                ix2, iy2 = min(seating_zone[2], bx2), min(seating_zone[3], by2)
                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    seating_area = max(1, (seating_zone[2] - seating_zone[0]) * (seating_zone[3] - seating_zone[1]))
                    if (inter_area / seating_area) > 0.08:
                        has_bike = True
                        break
        if has_bike: valid_riders.append((r_box, r_cls, r_conf))

    compliant_count, non_compliant_count = 0, 0

    for b_box, _ in bikes:
        cv2.rectangle(img, (b_box[0], b_box[1]), (b_box[2], b_box[3]), (130, 130, 130), 2)

    for h_box, h_cls, h_conf in helmets_raw:
        is_compliant = h_cls in [0, 3]
        color = (0, 220, 0) if is_compliant else (0, 0, 255)
        cv2.rectangle(img, (h_box[0], h_box[1]), (h_box[2], h_box[3]), color, 2)

    for r_box, r_cls, r_conf in valid_riders:
        rx1, ry1, rx2, ry2 = r_box
        r_height = ry2 - ry1
        head_zone = [rx1, max(0, ry1 - int(0.1 * r_height)), rx2, ry1 + int(0.45 * r_height)]

        has_helmet = any(compute_ior(h[0], head_zone) >= 0.30 and h[1] in [0, 3] for h in helmets_raw)
        has_no_helmet = any(compute_ior(h[0], head_zone) >= 0.30 and h[1] in [5, 6] for h in helmets_raw)
        
        if r_cls in [0, 3]: has_helmet = True
        elif r_cls in [5, 6]: has_no_helmet = True

        if has_helmet and not has_no_helmet:
            compliant_count += 1
            color, status = (0, 220, 0), "Safe"
        else:
            non_compliant_count += 1
            color, status = (0, 0, 255), "Complaint"

        cv2.rectangle(img, (rx1, ry1), (rx2, ry2), color, 2)
        cv2.putText(img, status, (rx1, max(15, ry1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    panel_w = 340
    panel = np.full((target_h, panel_w, 3), (28, 28, 32), dtype=np.uint8)
    y = 40
    cv2.putText(panel, "TRAFFIC COMPLIANCE SUMMARY", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 215, 255), 2, cv2.LINE_AA)
    y += 47
    
    summary_lines = [
        (f"Total Riders Located : {len(valid_riders)}", (255, 255, 255)),
        (f"Complaint (No Helmet): {non_compliant_count}", (50, 50, 255)),
        (f"Safe (Helmet)        : {compliant_count}", (50, 235, 50)),
    ]
    for text_line, col in summary_lines:
        cv2.putText(panel, text_line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
        y += 30

    combined_img = np.hstack((img, panel))
    return combined_img, len(valid_riders), compliant_count, non_compliant_count

# Streamlit UI Components
col1, col2 = st.columns([1, 2])

with col1:
    st.write("### Upload Image")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        orig_img = cv2.imdecode(file_bytes, 1)
        
        with st.spinner('Analyzing Traffic Compliance...'):
            combined_img, total, safe, complaint = detect_and_count_optimized(orig_img)
            rgb_img = cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)
            
        st.write("### Live Metrics")
        st.metric("Total Riders Detected", total)
        st.metric("Safe (Wearing Helmet)", safe)
        st.metric("Complaint (No Helmet)", complaint)

with col2:
    if uploaded_file is not None:
        st.image(rgb_img, caption="AI Processed Image with Compliance Dashboard", use_container_width=True)
    else:
        st.info("Upload an image on the left to see the AI detection results here.")