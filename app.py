import streamlit as st
import tempfile
import threading
import time
import os
import pandas as pd
from pathlib import Path
from collections import deque
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import io
import threading

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import av
except ImportError:
    av = None

try:
    from streamlit_webrtc import WebRtcMode, VideoProcessorBase, webrtc_streamer
    WEBRTC_AVAILABLE = True
except ImportError:
    WebRtcMode = None
    VideoProcessorBase = object
    webrtc_streamer = None
    WEBRTC_AVAILABLE = False

try:
    import winsound
except ImportError:
    winsound = None

MODEL_INSTANCE = None
MODEL_LOCK = threading.Lock()

# ============================================================
# CONFIGURATION
# ============================================================
ALLOWED_CLASSES = ["person", "cell phone", "laptop", "book", "tablet", "bottle"]
THREAT_CLASSES = {"cell phone": 85, "laptop": 70, "book": 55, "tablet": 65}
CONFIDENCE_THRESHOLD = 0.6
ALERT_THREAT_THRESHOLD = 70
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="AI Surveillance", page_icon="ðŸŽ¥", layout="wide")

st.markdown("""
<style>
.report-container { background-color: #f4f6fb; }
.card { background-color: #ffffff; padding: 22px; border-radius: 22px; box-shadow: 0 18px 45px rgba(20, 40, 84, 0.08); margin: 16px 0; }
.metric .stMetricValue {
    font-size: 2.4rem !important;
}
.metric .stMetricLabel {
    color: #344767 !important;
}
img {
    border-radius: 16px;
    transition: transform 0.18s ease-in-out, box-shadow 0.18s ease-in-out;
}
img:hover {
    transform: scale(1.04);
    box-shadow: 0 18px 40px rgba(0,0,0,0.18);
}
.section-title {
    font-weight: 700;
    margin-bottom: 10px;
}
.summary-card {
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.18);
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
def init_session_state():
    """Initialize all session variables."""
    if "running" not in st.session_state:
        st.session_state.running = False
    if "model" not in st.session_state:
        st.session_state.model = None
    if "alert_count" not in st.session_state:
        st.session_state.alert_count = 0
    if "suspicious_events" not in st.session_state:
        st.session_state.suspicious_events = []
    if "threat_history" not in st.session_state:
        st.session_state.threat_history = deque(maxlen=100)
    if "screenshots" not in st.session_state:
        st.session_state.screenshots = []
    if "video_writer" not in st.session_state:
        st.session_state.video_writer = None
    if "cap" not in st.session_state:
        st.session_state.cap = None
    if "alarm_active" not in st.session_state:
        st.session_state.alarm_active = False
    if "last_threat_score" not in st.session_state:
        st.session_state.last_threat_score = 0

init_session_state()


def get_shared_model():
    """Load the YOLO model once and reuse it across app reruns and webcam workers."""
    global MODEL_INSTANCE
    if MODEL_INSTANCE is None:
        with MODEL_LOCK:
            if MODEL_INSTANCE is None:
                MODEL_INSTANCE = YOLO("yolov8s.pt")
    return MODEL_INSTANCE

# ============================================================
# UTILITIES
# ============================================================
def load_model():
    """Load YOLO model."""
    if st.session_state.model is None:
        try:
            st.session_state.model = get_shared_model()
        except Exception as e:
            st.error(f"âŒ Failed to load model: {e}")
    return st.session_state.model


class BrowserWebcamProcessor(VideoProcessorBase):
    """Process frames coming from the browser webcam."""

    def __init__(self):
        self.model = get_shared_model()
        self.alert_count = 0
        self.suspicious_events = []
        self.threat_history = deque(maxlen=100)
        self.screenshots = []
        self.last_threat_score = 0
        self.alarm_active = False
        self.frame_count = 0
        self.output_path = str(Path("browser_output.avi"))
        self.video_writer = None

    def _ensure_writer(self, frame_shape):
        if self.video_writer is not None:
            return

        height, width = frame_shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.video_writer = cv2.VideoWriter(self.output_path, fourcc, 20.0, (width, height))

    def close(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def recv(self, frame):
        if av is None:
            return frame

        frame_np = frame.to_ndarray(format="bgr24")
        processed_frame, detections, threat = process_frame(frame_np.copy(), self.model)
        processed_frame = draw_boxes(processed_frame, detections)

        self._ensure_writer(processed_frame.shape)
        if self.video_writer is not None:
            self.video_writer.write(processed_frame)

        self.frame_count += 1
        self.threat_history.append(threat)

        alert_spike = threat > ALERT_THREAT_THRESHOLD and self.last_threat_score <= ALERT_THREAT_THRESHOLD
        if alert_spike:
            self.alert_count += 1
            current_time = time.time()

            for det in detections:
                if det["label"] in THREAT_CLASSES:
                    self.suspicious_events.append({
                        "label": det["label"],
                        "threat": int(det["conf"] * 100),
                        "time": current_time,
                    })

            if len(self.screenshots) < 10:
                Path("screenshots").mkdir(exist_ok=True)
                path = f"screenshots/browser_{int(current_time * 1000)}.jpg"
                try:
                    cv2.imwrite(path, processed_frame)
                    self.screenshots.append(path)
                except Exception:
                    pass

            if winsound and not self.alarm_active:
                self.alarm_active = True
                try:
                    threading.Thread(target=play_alarm, daemon=True).start()
                except Exception:
                    pass
        else:
            self.alarm_active = False

        self.last_threat_score = threat
        return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")

def play_alarm():
    """Play beep sound."""
    if winsound:
        try:
            winsound.Beep(1000, 300)
            time.sleep(0.1)
            winsound.Beep(1200, 300)
        except:
            pass

def calculate_threat(detections):
    """Calculate threat 0-100."""
    threat = 0
    for det in detections:
        label = det["label"]
        if label in THREAT_CLASSES:
            threat += THREAT_CLASSES[label]
        elif label == "person":
            threat += 20
        else:
            threat += 10
    return min(threat, 100)

def draw_boxes(frame, detections):
    """Draw bounding boxes."""
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        conf = det["conf"]

        # Color
        if label in THREAT_CLASSES:
            color = (0, 0, 255)  # RED
        elif label == "person":
            color = (0, 255, 0)  # GREEN
        else:
            color = (255, 0, 0)  # BLUE

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Label
        label_text = f"{label} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(label_text, font, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, y1 - 30), (x1 + text_size[0] + 10, y1), color, -1)
        cv2.putText(frame, label_text, (x1 + 5, y1 - 8), font, 0.6, (255, 255, 255), 2)

    return frame

def process_frame(frame, model):
    """Detect objects in frame."""
    if frame is None or model is None:
        return frame, [], 0

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    try:
        results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
    except:
        return frame, [], 0

    detections = []

    for result in results:
        if not hasattr(result, "boxes"):
            continue

        for box in result.boxes:
            try:
                xyxy = box.xyxy[0].tolist() if hasattr(box.xyxy, 'tolist') else list(box.xyxy[0])
                x1, y1, x2, y2 = map(int, xyxy[:4])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if conf < CONFIDENCE_THRESHOLD:
                    continue

                label = model.names[cls_id] if hasattr(model, "names") else str(cls_id)

                # Fix phone/laptop confusion
                if label == "laptop":
                    area = (x2 - x1) * (y2 - y1)
                    if area < 5000:
                        label = "cell phone"

                if label not in ALLOWED_CLASSES:
                    continue

                detections.append({
                    "label": label,
                    "conf": conf,
                    "bbox": (x1, y1, x2, y2)
                })
            except:
                continue

    return frame, detections, calculate_threat(detections)

def export_csv_report(event_summary, total_alerts, screenshot_count):
    """Export CSV."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    rows = []
    for event, data in event_summary.items():
        risk = "HIGH" if data["max_threat"] > 70 else "MEDIUM" if data["max_threat"] > 50 else "SAFE"
        rows.append({
            "Event": event,
            "Count": data["count"],
            "Max Threat": data["max_threat"],
            "Risk": risk,
            "Timestamp": timestamp
        })

    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode()

def export_txt_report(event_summary, total_alerts, screenshot_count):
    """Export TXT."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    average_threat = int(sum(data['max_threat'] for data in event_summary.values()) / len(event_summary)) if event_summary else 0

    report = """
============================================================
AI SURVEILLANCE REPORT
============================================================
"""
    report += f"Generated: {timestamp}\n\n"
    report += "## ðŸ“Š SUMMARY\n"
    report += f"Total Alerts      : {total_alerts}\n"
    report += f"Screenshots       : {screenshot_count}\n"
    report += f"Unique Events     : {len(event_summary)}\n"
    report += f"Average Threat    : {average_threat}%\n\n"
    report += "## ðŸš¨ THREAT ANALYSIS\n"

    for event, data in event_summary.items():
        risk = "HIGH" if data["max_threat"] > 70 else "MEDIUM" if data["max_threat"] > 50 else "SAFE"
        report += f"\nðŸ”´ {event.upper()}\n"
        report += f"â€¢ Detections : {data['count']}\n"
        report += f"â€¢ Max Threat : {data['max_threat']}%\n"
        report += f"â€¢ Risk Level : {risk}\n"

    report += "\n---\n"
    report += "âš  Risk Levels:\n"
    report += "0â€“50   â†’ SAFE\n"
    report += "50â€“70  â†’ MEDIUM\n"
    report += "70+    â†’ HIGH\n"
    report += "============================================================\n"
    return report.encode()

def summarize_events(events):
    """Summarize detections."""
    summary = {}
    for event in events:
        label = event.get("label", "unknown")
        if label not in summary:
            summary[label] = {"count": 1, "max_threat": event.get("threat", 0)}
        else:
            summary[label]["count"] += 1
            summary[label]["max_threat"] = max(summary[label]["max_threat"], event.get("threat", 0))
    return summary


def format_risk_label(threat):
    if threat > 70:
        return "High"
    if threat > 50:
        return "Medium"
    return "Safe"


def render_event_cards(summary):
    cols = st.columns(2)
    for i, (event, data) in enumerate(summary.items()):
        risk = format_risk_label(data["max_threat"])
        color = "#ff6b6b" if risk == "High" else "#f7b955" if risk == "Medium" else "#5cd08d"
        with cols[i % 2]:
            st.markdown(f"""
            <div style="
                padding:18px;
                border-radius:18px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                box-shadow: 0 14px 28px rgba(0,0,0,0.18);
                margin-bottom: 14px;">
                <div style="font-size:18px;font-weight:700;margin-bottom:10px;">{event.title()}</div>
                <div style="font-size:14px;line-height:1.7;">
                    ðŸ” Count: {data['count']}<br>
                    âš¡ Max Threat: {data['max_threat']}%<br>
                    ðŸš¨ Risk: <span style="color:{color};font-weight:700;">{risk.upper()}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# UI HEADER
# ============================================================
st.title("ðŸŽ¥ AI Surveillance System")
st.markdown("Real-time YOLO detection + threat analysis")
st.markdown("---")

# ============================================================
# SIDEBAR (UNIQUE KEYS)
# ============================================================
with st.sidebar:
    st.title("âš™ï¸ Controls")
    mode = st.radio("ðŸ“‹ Select Mode:", ["ðŸŒ Browser Webcam", "ðŸ“· Local Camera", "ðŸ“¤ Upload Video"], key="main_mode_radio")
    st.markdown("---")

    st.markdown("**System Status:**")
    model = load_model()
    if model:
        st.success("âœ“ Model Loaded (YOLOv8s)")
    else:
        st.error("âœ— Model Failed")

    st.markdown("---")
    st.markdown("**Settings:**")
    alarm_enabled = st.checkbox("ðŸ”Š Enable Alarm", value=True, key="alarm_checkbox_main")

# ============================================================
# BROWSER WEBCAM MODE
# ============================================================
if mode == "ðŸŒ Browser Webcam":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("ðŸ“¹ Browser Webcam")
    st.caption("This mode captures from your browser camera, which is the cloud-friendly option.")
    st.markdown('</div>', unsafe_allow_html=True)

    if not WEBRTC_AVAILABLE:
        st.error("streamlit-webrtc is not installed. Install dependencies to use browser webcam mode.")
    else:
        st.info("Click Start to grant camera access in your browser. This works on Streamlit Cloud.")
        webrtc_ctx = webrtc_streamer(
            key="browser-webcam",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=BrowserWebcamProcessor,
            async_processing=True,
        )

        if webrtc_ctx.video_processor is not None:
            vp = webrtc_ctx.video_processor

            if not webrtc_ctx.state.playing:
                vp.close()

            if vp.frame_count > 0:
                st.markdown("---")
                st.markdown("### ðŸ“Š Browser Webcam Analytics")

                total_alerts = vp.alert_count
                unique_events = len(summarize_events(vp.suspicious_events))
                screenshot_count = len(vp.screenshots)
                avg_threat = int(np.mean(vp.threat_history)) if vp.threat_history else 0
                risk_level = format_risk_label(avg_threat)
                risk_color = "#5cd08d" if risk_level == "Safe" else "#f7b955" if risk_level == "Medium" else "#ff6b6b"

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("ðŸš¨ Total Alerts", total_alerts)
                with m2:
                    st.metric("âš  Unique Events", unique_events)
                with m3:
                    st.metric("ðŸ“¸ Screenshots", screenshot_count)

                st.markdown(f"<div class='card' style='padding:16px;border-radius:18px;margin-top:12px;'>" \
                            f"<strong>System Risk:</strong> <span style='color:{risk_color};font-weight:700;'>{risk_level}</span> | " \
                            f"Average Threat: {avg_threat}%</div>", unsafe_allow_html=True)

                if vp.threat_history:
                    st.markdown("#### ðŸ“ˆ Threat Timeline")
                    df = pd.DataFrame({
                        "Frame": range(len(vp.threat_history)),
                        "Threat %": list(vp.threat_history)
                    })
                    st.line_chart(df, x="Frame", y="Threat %", use_container_width=True)

                if vp.suspicious_events:
                    st.markdown("#### ðŸ“‹ Event Summary")
                    render_event_cards(summarize_events(vp.suspicious_events))

                st.divider()
                st.markdown("## ðŸ“„ Export Report")

                col1, col2 = st.columns(2)
                if vp.suspicious_events:
                    summary = summarize_events(vp.suspicious_events)
                    with col1:
                        csv = export_csv_report(summary, vp.alert_count, len(vp.screenshots))
                        st.download_button(
                            "â¬‡ Download CSV Report",
                            csv,
                            "report.csv",
                            "text/csv",
                            use_container_width=True,
                            key="dl_csv_browser"
                        )
                    with col2:
                        txt = export_txt_report(summary, vp.alert_count, len(vp.screenshots))
                        st.download_button(
                            "â¬‡ Download Text Report",
                            txt,
                            "report.txt",
                            "text/plain",
                            use_container_width=True,
                            key="dl_txt_browser"
                        )

                if os.path.exists(vp.output_path) and os.path.getsize(vp.output_path) > 0:
                    st.markdown("### ðŸŽ¥ Recorded Video")
                    with open(vp.output_path, "rb") as f:
                        st.download_button(
                            "â¬‡ï¸ Download Video",
                            f.read(),
                            "browser-surveillance.avi",
                            "video/x-msvideo",
                            use_container_width=True,
                            key="dl_video_browser"
                        )

                if vp.screenshots:
                    st.markdown("---")
                    st.markdown("## ðŸ“¸ Alert Snapshots")
                    display_screenshots = vp.screenshots[-6:]
                    cols = st.columns(3)
                    for i, path in enumerate(display_screenshots):
                        try:
                            img = cv2.imread(path)
                            if img is not None:
                                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                caption = f"Alert {i+1}"
                                if i < len(vp.suspicious_events):
                                    evt = vp.suspicious_events[i]
                                    caption += f" | {evt['label'].title()} | Threat: {evt['threat']}%"
                                cols[i % 3].image(img_rgb, caption=caption, use_container_width=True)
                        except:
                            pass

elif mode == "ðŸ“· Local Camera":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("ðŸ“¹ Live Feed")

    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("â–¶ Start Camera", key="btn_start", use_container_width=True)
    with col2:
        stop_btn = st.button("â¹ Stop Camera", key="btn_stop", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if start_btn:
        st.session_state.running = True
        st.session_state.alert_count = 0
        st.session_state.suspicious_events = []
        st.session_state.threat_history.clear()
        st.session_state.screenshots = []

    if stop_btn:
        st.session_state.running = False
        if st.session_state.video_writer is not None:
            st.session_state.video_writer.release()
            st.session_state.video_writer = None
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None

    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    metrics_placeholder = st.empty()

    if st.session_state.running:
        try:
            if st.session_state.cap is None:
                st.session_state.cap = cv2.VideoCapture(0)
                st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

            cap = st.session_state.cap

            if not cap.isOpened():
                st.error("âŒ Camera unavailable")
                st.session_state.running = False
            else:
                status_placeholder.info("ðŸ”„ Recording...")

                # Video writer
                if st.session_state.video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    st.session_state.video_writer = cv2.VideoWriter(
                        "output.avi",
                        fourcc,
                        20.0,
                        (FRAME_WIDTH, FRAME_HEIGHT)
                    )

                frame_limit = 30 * 10  # 10 seconds
                frame_num = 0

                while st.session_state.running and frame_num < frame_limit:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_num += 1
                    current_time = time.time()

                    # Process
                    processed_frame, detections, threat = process_frame(frame.copy(), model)
                    processed_frame = draw_boxes(processed_frame, detections)

                    # Write video
                    if st.session_state.video_writer is not None:
                        st.session_state.video_writer.write(processed_frame)

                    # Threat history
                    st.session_state.threat_history.append(threat)

                    # Alert spike detection
                    alert_spike = (threat > ALERT_THREAT_THRESHOLD and
                                 st.session_state.last_threat_score <= ALERT_THREAT_THRESHOLD)

                    if alert_spike:
                        st.session_state.alert_count += 1

                        # Add events
                        for det in detections:
                            if det["label"] in THREAT_CLASSES:
                                st.session_state.suspicious_events.append({
                                    "label": det["label"],
                                    "threat": int(det["conf"] * 100),
                                    "time": current_time
                                })

                        # Alarm
                        if alarm_enabled and not st.session_state.alarm_active:
                            st.session_state.alarm_active = True
                            threading.Thread(target=play_alarm, daemon=True).start()

                        # Screenshot
                        if len(st.session_state.screenshots) < 10:
                            Path("screenshots").mkdir(exist_ok=True)
                            path = f"screenshots/screenshot_{int(current_time * 1000)}.jpg"
                            try:
                                cv2.imwrite(path, processed_frame)
                                st.session_state.screenshots.append(path)
                            except:
                                pass
                    else:
                        st.session_state.alarm_active = False

                    st.session_state.last_threat_score = threat

                    # Display
                    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, use_container_width=True)

                    # Metrics
                    with metrics_placeholder.container():
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.metric("ðŸš¨ Alerts", st.session_state.alert_count)
                        with m2:
                            st.metric("âš ï¸ Events", len(st.session_state.suspicious_events))
                        with m3:
                            st.metric("ðŸ“¸ Screenshots", len(st.session_state.screenshots))

                    time.sleep(0.03)

                status_placeholder.success("âœ… Recording stopped")

        except Exception as e:
            st.error(f"âŒ Error: {e}")
            st.session_state.running = False

    # Results
    if not st.session_state.running and st.session_state.alert_count > 0:
        st.markdown("---")

        # Download video
        if os.path.exists("output.avi") and os.path.getsize("output.avi") > 0:
            st.markdown("### ðŸŽ¥ Recorded Video")
            st.info("Download recorded session below")
            with open("output.avi", "rb") as f:
                st.download_button(
                    "â¬‡ï¸ Download Video",
                    f.read(),
                    "surveillance.avi",
                    "video/x-msvideo",
                    use_container_width=True,
                    key="dl_video_live"
                )

        st.markdown("---")
        st.markdown("### ðŸ“Š Analytics Dashboard")
        summary = summarize_events(st.session_state.suspicious_events)
        total_alerts = st.session_state.alert_count
        unique_events = len(summary)
        screenshot_count = len(st.session_state.screenshots)
        avg_threat = int(np.mean(st.session_state.threat_history)) if st.session_state.threat_history else 0
        risk_level = format_risk_label(avg_threat)
        risk_color = "#5cd08d" if risk_level == "Safe" else "#f7b955" if risk_level == "Medium" else "#ff6b6b"

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("ðŸš¨ Total Alerts", total_alerts)
        with m2:
            st.metric("âš  Unique Events", unique_events)
        with m3:
            st.metric("ðŸ“¸ Screenshots", screenshot_count)

        st.markdown(f"<div class='card' style='padding:16px;border-radius:18px;margin-top:12px;'>" \
                    f"<strong>System Risk:</strong> <span style='color:{risk_color};font-weight:700;'>{risk_level}</span> | " \
                    f"Average Threat: {avg_threat}%</div>", unsafe_allow_html=True)

        if st.session_state.threat_history:
            st.markdown("#### ðŸ“ˆ Threat Timeline")
            df = pd.DataFrame({
                "Frame": range(len(st.session_state.threat_history)),
                "Threat %": list(st.session_state.threat_history)
            })
            st.line_chart(df, x="Frame", y="Threat %", use_container_width=True)

        if st.session_state.suspicious_events:
            st.markdown("#### ðŸ“‹ Event Summary")
            render_event_cards(summary)

        st.divider()
        st.markdown("## ðŸ“„ Export Report")

        col1, col2 = st.columns(2)
        if st.session_state.suspicious_events:
            summary = summarize_events(st.session_state.suspicious_events)

            with col1:
                csv = export_csv_report(summary, st.session_state.alert_count, len(st.session_state.screenshots))
                st.download_button(
                    "â¬‡ Download CSV Report",
                    csv,
                    "report.csv",
                    "text/csv",
                    use_container_width=True,
                    key="dl_csv_live"
                )

            with col2:
                txt = export_txt_report(summary, st.session_state.alert_count, len(st.session_state.screenshots))
                st.download_button(
                    "â¬‡ Download Text Report",
                    txt,
                    "report.txt",
                    "text/plain",
                    use_container_width=True,
                    key="dl_txt_live"
                )

        if st.session_state.screenshots:
            st.markdown("---")
            st.markdown("## ðŸ“¸ Alert Snapshots")
            display_screenshots = st.session_state.screenshots[-6:]
            cols = st.columns(3)
            for i, path in enumerate(display_screenshots):
                try:
                    img = cv2.imread(path)
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        caption = f"Alert {i+1}"
                        if i < len(st.session_state.suspicious_events):
                            evt = st.session_state.suspicious_events[i]
                            caption += f" | {evt['label'].title()} | Threat: {evt['threat']}%"
                        cols[i % 3].image(img_rgb, caption=caption, use_container_width=True)
                except:
                    pass

# ============================================================
# UPLOAD VIDEO MODE
# ============================================================
elif mode == "ðŸ“¤ Upload Video":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("ðŸ“¤ Analyze Video")
    st.markdown('</div>', unsafe_allow_html=True)

    file = st.file_uploader("Choose video", type=["mp4", "avi", "mov", "mkv"], key="upload_main")

    if file is not None:
        if st.button("ðŸ” Analyze", key="btn_analyze", use_container_width=True):
            temp = Path(tempfile.gettempdir()) / file.name
            with open(temp, "wb") as f:
                f.write(file.getbuffer())

            status = st.empty()
            prog = st.progress(0)
            status.info("ðŸ”„ Processing...")

            try:
                cap = cv2.VideoCapture(str(temp))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                frame_num = 0
                alerts = 0
                events = []
                threats = []
                shots = []

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_num += 1

                    _, detections, threat = process_frame(frame.copy(), model)
                    threats.append(threat)

                    if threat > ALERT_THREAT_THRESHOLD:
                        alerts += 1
                        for det in detections:
                            if det["label"] in THREAT_CLASSES:
                                events.append({
                                    "label": det["label"],
                                    "threat": int(det["conf"] * 100)
                                })

                        if len(shots) < 10:
                            Path("screenshots").mkdir(exist_ok=True)
                            path = f"screenshots/upload_{int(time.time() * 1000)}.jpg"
                            cv2.imwrite(path, frame)
                            shots.append(path)

                    prog.progress(min(frame_num / total, 1.0))

                cap.release()

                st.session_state.alert_count = alerts
                st.session_state.suspicious_events = events
                st.session_state.threat_history = deque(threats[-100:], maxlen=100)
                st.session_state.screenshots = shots

                status.success("âœ… Done!")

            except Exception as e:
                st.error(f"âŒ Error: {e}")
            finally:
                if temp.exists():
                    temp.unlink()

        # Results
        if st.session_state.alert_count > 0:
            st.markdown("---")
            st.markdown("### ðŸ“Š Analytics Dashboard")
            summary = summarize_events(st.session_state.suspicious_events)
            total_alerts = st.session_state.alert_count
            unique_events = len(summary)
            screenshot_count = len(st.session_state.screenshots)
            avg_threat = int(np.mean(st.session_state.threat_history)) if st.session_state.threat_history else 0
            risk_level = format_risk_label(avg_threat)
            risk_color = "#5cd08d" if risk_level == "Safe" else "#f7b955" if risk_level == "Medium" else "#ff6b6b"

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("ðŸš¨ Total Alerts", total_alerts)
            with c2:
                st.metric("âš  Unique Events", unique_events)
            with c3:
                st.metric("ðŸ“¸ Screenshots", screenshot_count)

            st.markdown(f"<div class='card' style='padding:16px;border-radius:18px;margin-top:12px;'>" \
                        f"<strong>System Risk:</strong> <span style='color:{risk_color};font-weight:700;'>{risk_level}</span> | " \
                        f"Average Threat: {avg_threat}%</div>", unsafe_allow_html=True)

            if st.session_state.threat_history:
                st.markdown("#### ðŸ“ˆ Threat Timeline")
                df = pd.DataFrame({
                    "Frame": range(len(st.session_state.threat_history)),
                    "Threat %": list(st.session_state.threat_history)
                })
                st.line_chart(df, x="Frame", y="Threat %", use_container_width=True)

            if st.session_state.suspicious_events:
                st.markdown("#### ðŸ“‹ Event Summary")
                render_event_cards(summary)

            st.divider()
            st.markdown("## ðŸ“„ Export Report")

            col1, col2 = st.columns(2)
            if st.session_state.suspicious_events:
                summary = summarize_events(st.session_state.suspicious_events)

                with col1:
                    csv = export_csv_report(summary, st.session_state.alert_count, len(st.session_state.screenshots))
                    st.download_button(
                        "â¬‡ Download CSV Report",
                        csv,
                        "report.csv",
                        "text/csv",
                        use_container_width=True,
                        key="dl_csv_upload"
                    )

                with col2:
                    txt = export_txt_report(summary, st.session_state.alert_count, len(st.session_state.screenshots))
                    st.download_button(
                        "â¬‡ Download Text Report",
                        txt,
                        "report.txt",
                        "text/plain",
                        use_container_width=True,
                        key="dl_txt_upload"
                    )

            if st.session_state.screenshots:
                st.markdown("---")
                st.markdown("## ðŸ“¸ Alert Snapshots")
                display_screenshots = st.session_state.screenshots[-6:]
                cols = st.columns(3)
                for i, path in enumerate(display_screenshots):
                    try:
                        img = cv2.imread(path)
                        if img is not None:
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            caption = f"Alert {i+1}"
                            if i < len(st.session_state.suspicious_events):
                                evt = st.session_state.suspicious_events[i]
                                caption += f" | {evt['label'].title()} | Threat: {evt['threat']}%"
                            cols[i % 3].image(img_rgb, caption=caption, use_container_width=True)
                    except:
                        pass

st.markdown("---")
st.markdown("<div style='text-align:center;color:gray;font-size:11px;'><b>ðŸŽ¥ AI Surveillance v4.0 FIXED</b><br>Production-Ready | YOLO | Real-time Analysis</div>", unsafe_allow_html=True)
