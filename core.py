import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import winsound
except ImportError:  # pragma: no cover
    winsound = None

# TARGET OBJECTS TO DETECT
TARGET_OBJECTS = {"person", "cell phone", "laptop", "tablet", "book", "dog", "cat"}
RESTRICTED_OBJECTS = {"cell phone", "laptop", "tablet", "book"}  # Devices to restrict
ANIMAL_OBJECTS = {"dog", "cat"}

# SUSPICIOUS OBJECTS
SUSPICIOUS_OBJECTS = [
    "cell phone",
    "laptop",
    "keyboard",
    "remote",
    "mouse",
    "book"
]

# TIMING AND LIMITS
COOLDOWN_SECONDS = 3.0  # Alarm cooldown
MAX_SCREENSHOTS = 5  # Max screenshots per session
MIN_CAPTURE_GAP = 2.0  # Gap between screenshots in seconds
ALERT_THREAT_THRESHOLD = 70  # Threat score threshold for alert

# HELPER FUNCTIONS

def _get_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Calculate center point of bounding box."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def _is_near(center1: Tuple[float, float], center2: Tuple[float, float], frame_size: Tuple[int, int]) -> bool:
    """Check if two centers are near each other (within 18% of frame diagonal)."""
    width, height = frame_size
    if width <= 0 or height <= 0:
        return False
    threshold = max(width, height) * 0.18
    return _distance(center1, center2) <= threshold


def _clamp_score(score: float) -> int:
    """Clamp threat score between 0-100."""
    return int(min(100, max(0, round(score))))


def behavior_suspicion(detections):
    suspicious = 0

    persons = [d for d in detections if d["label"] == "person"]
    objects = [d for d in detections if d["label"] != "person"]

    if len(persons) > 1:
        suspicious += 20

    if len(objects) > 2:
        suspicious += 25

    return suspicious


def _play_alarm() -> None:
    """Play a beep alarm sound."""
    if winsound is None:
        return
    try:
        # winsound.Beep(frequency, duration_ms)
        winsound.Beep(1000, 500)  # 1000 Hz for 500ms
    except Exception:
        pass


def load_model(model_path: Optional[Path] = None) -> Any:
    """Load YOLO model from file."""
    if model_path is None:
        model_path = Path("yolov8n.pt")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return YOLO(str(model_path))


# SIMPLIFIED ANOMALY DETECTION
# Rule: If person AND restricted object in same frame → anomaly
class SimpleAnomalyDetector:
    """Simple anomaly detection: person + restricted object = anomaly."""

    def __init__(self):
        self.enabled = True

    def detect_anomaly(self, detections: List[Dict[str, Any]]) -> bool:
        """
        Detect anomaly: person AND restricted object (phone, laptop, tablet, book) 
        in same frame = suspicious behavior.
        """
        has_person = any(d["object_name"] == "person" for d in detections)
        has_restricted = any(d["is_restricted"] for d in detections)
        return has_person and has_restricted



def process_frame(
    frame: Any,
    model: Any = None,
    draw_boxes: bool = True,
) -> Dict[str, Any]:
    """
    Process a single frame using YOLO detection.
    
    Returns:
        - detections: list of detected objects with threat scores
        - alert_triggered: True if any threat > threshold
        - suspicious_events: list of suspicious behavior events
    """
    if frame is None:
        return {"detections": [], "alert_triggered": False, "suspicious_events": []}

    # Load model if not provided
    if model is None:
        try:
            model = load_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            return {"detections": [], "alert_triggered": False, "suspicious_events": []}

    # Validate frame
    if not hasattr(frame, "shape") or len(frame.shape) < 2:
        return {"detections": [], "alert_triggered": False, "suspicious_events": []}

    frame_height, frame_width = frame.shape[:2]
    if frame_width == 0 or frame_height == 0:
        return {"detections": [], "alert_triggered": False, "suspicious_events": []}

    # Optimize frame processing
    frame = cv2.resize(frame, (640, 480))

    # Run YOLO detection
    try:
        results = model(frame, verbose=False)
    except Exception as e:
        print(f"Error running detection: {e}")
        return {"detections": [], "alert_triggered": False, "suspicious_events": []}

    detections: List[Dict[str, Any]] = []
    person_centers: List[Tuple[float, float]] = []
    suspicious_events: List[Dict[str, Any]] = []
    alert_triggered = False

    # PROCESS YOLO RESULTS
    for result in results:
        if not hasattr(result, "boxes") or len(result.boxes) == 0:
            continue

        # Loop through detections
        for box in result.boxes:
            try:
                # Extract bounding box coordinates
                xyxy = box.xyxy
                if xyxy is None or len(xyxy) == 0:
                    continue
                
                x1, y1, x2, y2 = map(int, xyxy[0].tolist())
                
                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                # Extract class and confidence
                cls_id = int(box.cls[0])
                conf_value = float(box.conf[0])
                confidence = max(0.0, min(1.0, conf_value))
                
                # Only consider detections with confidence >= 0.5
                if confidence < 0.5:
                    continue
                
                # Get label name
                label = model.names[cls_id] if hasattr(model, "names") else str(cls_id)
                
                # Only track target objects
                if label not in TARGET_OBJECTS:
                    continue

                # Calculate center and area
                bbox = (x1, y1, x2, y2)
                center = _get_center(bbox)
                area_ratio = ((x2 - x1) * (y2 - y1)) / max(1.0, frame_width * frame_height)

                # Create detection dict
                detection: Dict[str, Any] = {
                    "object_name": label,
                    "confidence": confidence,
                    "bbox": bbox,
                    "center": center,
                    "area_ratio": area_ratio,
                    "is_restricted": label in RESTRICTED_OBJECTS,
                    "is_animal": label in ANIMAL_OBJECTS,
                    "near_person": False,
                    "threat": 0,
                    "label": "",
                }

                # Track person centers for proximity checks
                if label == "person":
                    person_centers.append(center)

                detections.append(detection)

            except Exception as e:
                continue

    # CHECK PROXIMITY: Restricted object near person
    for detection in detections:
        if detection["object_name"] != "person" and detection["is_restricted"]:
            for person_center in person_centers:
                if _is_near(person_center, detection["center"], (frame_width, frame_height)):
                    detection["near_person"] = True
                    suspicious_events.append(
                        {
                            "object_name": detection["object_name"],
                            "confidence": int(detection["confidence"] * 100),
                            "reason": f"{detection['object_name']} detected near person",
                        }
                    )
                    break

    # CALCULATE THREAT SCORES - BEHAVIOR BASED (NOT confidence-based)
    threat_score = 0

    for det in detections:
        if det["label"] == "cell phone":
            threat_score += 60
        elif det["label"] == "laptop":
            threat_score += 50
        elif det["label"] == "book":
            threat_score += 40

    # Add behavior suspicion
    threat_score += behavior_suspicion(detections)

    # Limit threat score
    threat_score = min(threat_score, 100)

    for detection in detections:
        threat_score = 0.0
        object_name = detection["object_name"]
        
        # Base threat by object type (independent of detection confidence)
        if object_name == "person":
            threat_score = 25.0  # Neutral: just a person detected
        elif object_name in ANIMAL_OBJECTS:
            threat_score = 30.0  # Low: animals present
        elif object_name in RESTRICTED_OBJECTS:
            threat_score = 20.0  # Very low: device alone without person
        
        # BEHAVIORAL THREAT BOOSTS - when near person
        if detection["near_person"]:
            if object_name == "cell phone":
                threat_score = 85.0  # HIGH: Person using phone
            elif object_name in {"laptop", "tablet"}:
                threat_score = 65.0  # MEDIUM-HIGH: Person using device
            elif object_name == "book":
                threat_score = 55.0  # MEDIUM: Person reading/holding book
        
        detection["threat"] = _clamp_score(threat_score)
        # Display: separate confidence from threat
        detection["label"] = f"{object_name} | Conf:{int(detection['confidence'] * 100)}% | Threat:{detection['threat']}%"
        
        # Check if alert (threat > 70)
        if detection["threat"] > ALERT_THREAT_THRESHOLD:
            alert_triggered = True

    # SMART LABEL DISPLAY
    if threat_score > 60:
        status = "🔴 Suspicious"
    elif threat_score > 40:
        status = "🟡 Warning"
    else:
        status = "🟢 Normal"

    # DRAW BOUNDING BOXES (optional, for visualization)
    if draw_boxes:
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            
            # Color based on threat level: green <40, yellow 40-70, red >70
            threat = detection["threat"]
            if threat > 70:
                color = (0, 0, 255)  # Red for high threat
            elif threat >= 40:
                color = (0, 165, 255)  # Yellow/Orange for medium threat
            else:
                color = (0, 255, 0)  # Green for low threat
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Put label
            cv2.putText(
                frame,
                detection["label"],
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    return {
        "detections": detections,
        "alert_triggered": alert_triggered,
        "suspicious_events": suspicious_events,
        "threat_score": threat_score,
        "status": status,
    }


def _get_threat_type(detections: List[Dict[str, Any]]) -> str:
    """
    Determine threat type from detections.
    Returns a string identifying the type of threat detected.
    """
    # Find highest threat detection
    max_threat = max((d["threat"] for d in detections), default=0)
    if max_threat == 0:
        return "unknown"
    
    # Find detection with max threat
    threat_detection = next(d for d in detections if d["threat"] == max_threat)
    obj = threat_detection["object_name"]
    
    if threat_detection["near_person"] and obj == "cell phone":
        return "phone_usage"
    elif threat_detection["near_person"] and obj in {"laptop", "tablet"}:
        return "device_usage"
    elif threat_detection["near_person"] and obj == "book":
        return "reading"
    elif obj in ANIMAL_OBJECTS:
        return "animal_detected"
    else:
        return "unknown"


def analyze_video(
    video_path: Union[Path, int, str],
    model: Any = None,
    screenshot_folder: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Analyze video (file or webcam) for threats.
    
    Returns:
        - alert_count: number of UNIQUE alert events (with 3-sec deduplication)
        - screenshots: list of screenshot file paths
        - suspicious_events: list of suspicious events detected
    """
    if screenshot_folder is None:
        screenshot_folder = Path("screenshots")
    screenshot_folder.mkdir(parents=True, exist_ok=True)

    # Load model
    try:
        if model is None:
            model = load_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        return {"alert_count": 0, "screenshots": [], "suspicious_events": []}

    # Open video source
    if isinstance(video_path, Path):
        capture_source = str(video_path)
    else:
        capture_source = video_path
    
    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        print("Error: Cannot open video source")
        return {"alert_count": 0, "screenshots": [], "suspicious_events": []}

    # Tracking variables
    frame_index = 0
    alert_count = 0
    screenshots: List[str] = []
    suspicious_events: List[Dict[str, Any]] = []
    last_capture_time = 0.0
    last_alarm_time = 0.0
    last_alert_time = 0.0  # Track last alert time for deduplication
    last_threat_type = ""  # Track last threat type

    print("Processing video... Press 'q' to stop.")

    try:
        while True:
            success, frame = capture.read()
            if not success or frame is None:
                break
            
            frame_index += 1

            # Process frame - draw boxePrame
            result = process_frame(frame, model=model, draw_boxes=True)
            detections = result.get("detections", [])
            alert_triggered = result.get("alert_triggered", False)
            events = result.get("suspicious_events", [])
            suspicious_events.extend(events)

            # Check for alerts with deduplication
            current_time = time.time()
            if alert_triggered and detections:
                # Determine current threat type from detections
                current_threat_type = _get_threat_type(detections)
                
                # Check if this is a NEW alert (not a duplicate within cooldown)
                is_new_alert = (
                    current_time - last_alert_time >= COOLDOWN_SECONDS or
                    current_threat_type != last_threat_type
                )
                
                if is_new_alert:
                    alert_count += 1
                    last_alert_time = current_time
                    last_threat_type = current_threat_type
                    print(f"[ALERT #{alert_count}] Frame {frame_index}: {current_threat_type} detected!")
                    
                    # Play alarm
                    if current_time - last_alarm_time >= COOLDOWN_SECONDS:
                        last_alarm_time = current_time
                        threading.Thread(target=_play_alarm, daemon=True).start()
                    
                    # Save screenshot (with gaps)
                    if len(screenshots) < MAX_SCREENSHOTS and current_time - last_capture_time >= MIN_CAPTURE_GAP:
                        try:
                            screenshot_path = screenshot_folder / f"alert_{frame_index}_{int(current_time * 1000)}.jpg"
                            cv2.imwrite(str(screenshot_path), frame)
                            screenshots.append(str(screenshot_path))
                            last_capture_time = current_time
                            print(f"Screenshot saved: {screenshot_path}")
                        except Exception as e:
                            print(f"Error saving screenshot: {e}")

            # Display frame
            cv2.imshow("AI Surveillance System", frame)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Stopping...")
                break

    finally:
        capture.release()
        cv2.destroyAllWindows()

    # Return report
    report = {
        "alert_count": alert_count,
        "screenshots": screenshots,
        "suspicious_events": suspicious_events,
    }
    
    print(f"\n=== SURVEILLANCE REPORT ===")
    print(f"Total Unique Alerts: {alert_count}")
    print(f"Screenshots Captured: {len(screenshots)}")
    print(f"Suspicious Events: {len(suspicious_events)}")
    
    return report


def _test_webcam_capture():
    """Test webcam capture and display."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not available")
        return None

    print("Testing webcam. Press 'q' to stop.")
    frame_count = 0
    
    try:
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                print("Error: Failed to capture frame")
                break
            
            frame_count += 1
            cv2.imshow("Webcam Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Captured {frame_count} frames")
    return True


def test_backend():
    """
    Main test function: Load model, run detection on webcam, 
    display bounding boxes, trigger alarms, and save screenshots.
    """
    print("=== AI SURVEILLANCE SYSTEM TEST ===\n")
    
    # Test webcam availability
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Camera not available")
        return

    print("✓ Camera available")

    # Load model
    try:
        model = load_model()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"ERROR: Model loading failed: {e}")
        cap.release()
        return

    # Create screenshot folder
    screenshot_folder = Path("screenshots")
    screenshot_folder.mkdir(parents=True, exist_ok=True)
    print(f"✓ Screenshot folder: {screenshot_folder}")

    # Tracking variables
    frame_count = 0
    alert_count = 0
    screenshot_count = 0
    suspicious_events = []
    last_capture_time = 0.0
    last_alarm_time = 0.0
    last_alert_time = 0.0  # Track last alert time for deduplication
    last_threat_type = ""  # Track last threat type

    print("\n=== LIVE DETECTION ===")
    print("Press 'q' to stop and see report.\n")

    try:
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                break

            frame_count += 1

            # Process frame with detection and bounding boxes
            result = process_frame(frame, model=model, draw_boxes=True)
            detections = result.get("detections", [])
            alert_triggered = result.get("alert_triggered", False)
            events = result.get("suspicious_events", [])
            suspicious_events.extend(events)

            # Show detections in console (every 30 frames)
            if frame_count % 30 == 0:
                if detections:
                    print(f"Frame {frame_count}: {len(detections)} objects detected")
                    for det in detections:
                        print(f"  - {det['label']}")

            # Handle alerts with deduplication
            current_time = time.time()
            if alert_triggered and detections:
                # Determine current threat type from detections
                current_threat_type = _get_threat_type(detections)
                
                # Check if this is a NEW alert (not a duplicate within cooldown)
                is_new_alert = (
                    current_time - last_alert_time >= COOLDOWN_SECONDS or
                    current_threat_type != last_threat_type
                )
                
                if is_new_alert:
                    alert_count += 1
                    last_alert_time = current_time
                    last_threat_type = current_threat_type
                    print(f"🚨 ALERT #{alert_count} at frame {frame_count}: {current_threat_type}")

                    # Play alarm (with cooldown)
                    if current_time - last_alarm_time >= COOLDOWN_SECONDS:
                        last_alarm_time = current_time
                        print("  → Alarm triggered (beep)")
                        threading.Thread(target=_play_alarm, daemon=True).start()

                    # Save screenshot (with gaps)
                    if screenshot_count < MAX_SCREENSHOTS and current_time - last_capture_time >= MIN_CAPTURE_GAP:
                        try:
                            screenshot_path = screenshot_folder / f"alert_{frame_count}_{int(current_time * 1000)}.jpg"
                            cv2.imwrite(str(screenshot_path), frame)
                            screenshot_count += 1
                            last_capture_time = current_time
                            print(f"  → Screenshot saved [{screenshot_count}/{MAX_SCREENSHOTS}]")
                        except Exception as e:
                            print(f"  → Error saving screenshot: {e}")

            # Display frame with bounding boxes
            cv2.imshow("AI Surveillance System - Live Detection", frame)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nStopping...\n")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Generate and display report
    print("=== SURVEILLANCE REPORT ===")
    print(f"Duration: {frame_count} frames")
    print(f"Total Unique Alerts: {alert_count}")
    print(f"Screenshots Captured: {screenshot_count}")
    print(f"Suspicious Events: {len(suspicious_events)}")
    
    if suspicious_events:
        print("\nSuspicious Events Detected:")
        for event in suspicious_events[:10]:  # Show first 10
            print(f"  - {event.get('object_name', 'unknown')}: {event.get('reason', 'no reason')}")
        if len(suspicious_events) > 10:
            print(f"  ... and {len(suspicious_events) - 10} more events")

    print("\n✓ Test complete")


if __name__ == "__main__":
    test_backend()


