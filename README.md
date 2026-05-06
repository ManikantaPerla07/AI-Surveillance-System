# AI Surveillance System

<div align="center">

[![Live Demo](https://img.shields.io/badge/Live-Demo-blue.svg)](#local-setup)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?logo=opencv&logoColor=white)
![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-00A9CE)
![License](https://img.shields.io/badge/License-MIT-green)

Real-time AI surveillance for live camera or uploaded video. The app detects selected objects, scores threat levels, triggers alerts, captures screenshots, records video, and generates exportable reports.

Repository: [AI-Surveillance-System](https://github.com/ManikantaPerla07/AI-Surveillance-System)

</div>

## Table of Contents

- [Overview](#overview)
- [Feature Highlights](#feature-highlights)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Architecture Diagram](#architecture-diagram)
- [Screenshots](#screenshots)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Run Locally](#run-locally)
- [Configuration Notes](#configuration-notes)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Author](#author)

## Overview

AI Surveillance System is a Streamlit-based monitoring dashboard that uses YOLOv8 to detect selected objects in real time from a webcam or uploaded video. It is designed to surface suspicious activity through threat scoring, visual alerts, screenshots, recorded video, and downloadable reports.

## Feature Highlights

- Real-time object detection with bounding boxes.
- Threat scoring with Safe, Medium, and High risk states.
- Automatic alert generation with optional sound.
- Screenshot capture when a threat spike is detected.
- Session video recording with download support.
- Threat timeline analytics and event summaries.
- CSV and TXT report export.
- Browser webcam, local camera, and uploaded video workflows.

## How It Works

1. The app reads frames from the default camera or from an uploaded video file.
2. YOLOv8 detects objects in each frame.
3. The app filters detections to allowed classes and calculates a threat score.
4. If the score crosses the alert threshold, the system logs the event, saves a screenshot, and can play an alarm.
5. The Streamlit UI shows annotated frames, analytics, and report controls.

## Architecture

The app follows a simple capture, detect, score, alert, and report pipeline.

## Architecture Diagram

```mermaid
flowchart TD
	A[Webcam or Uploaded Video] --> B[Frame Capture]
	B --> C[YOLOv8 Inference]
	C --> D[Filter Allowed Objects]
	D --> E[Threat Scoring]
	E --> F{Threat >= Threshold?}
	F -- Yes --> G[Trigger Alert]
	G --> H[Save Screenshot]
	G --> I[Play Alarm]
	G --> J[Log Suspicious Event]
	F -- No --> K[Continue Monitoring]
	J --> L[Analytics Dashboard]
	H --> L
	I --> L
	K --> L
	L --> M[CSV / TXT Export]
```

```mermaid
sequenceDiagram
	participant U as User
	participant S as Streamlit App
	participant M as YOLOv8 Model
	participant C as Camera/Video

	U->>S: Start live camera or upload video
	S->>C: Read next frame
	S->>M: Run object detection
	M-->>S: Return detections
	S->>S: Calculate threat score
	alt Threat is high
		S->>S: Save screenshot and log alert
		S->>S: Play alarm sound
	end
	S-->>U: Render annotated frame and dashboard
```

## Screenshots

Current screenshots are stored in the `screenshots/` folder.

### Live Detection

![Live Detection](screenshots/live.png)

### Dashboard

![Dashboard](screenshots/dashboard.png)

### Threat Timeline

![Threat Timeline](screenshots/timeline.png)

### Event Summary

![Event Summary](screenshots/event.png)

### Export Report

![Export Report](screenshots/export.png)

### Alert Snapshots

![Alert Snapshots](screenshots/alerts.png)

## Tech Stack

- Python 3.11+
- Streamlit
- Ultralytics YOLOv8
- OpenCV
- NumPy
- Pandas
- scikit-learn

## Project Structure

```text
.
â”œâ”€â”€ app.py
â”œâ”€â”€ core.py
â”œâ”€â”€ utils.py
â”œâ”€â”€ generate_alarm.py
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ README.md
â”œâ”€â”€ alarm.wav
â”œâ”€â”€ screenshots/
â””â”€â”€ test_yolo.py
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/ManikantaPerla07/AI-Surveillance-System.git
cd AI-Surveillance-System
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the app

```bash
streamlit run app.py
```

### 4. Open the app

Streamlit will print a local URL in the terminal, usually:

```text
http://localhost:8501
```

## Usage Guide

### Browser Webcam Mode

- Choose `Browser Webcam` in the sidebar.
- Click `Start` in the Streamlit WebRTC widget and allow camera access in the browser.
- Review the annotated stream, threat score, alerts, screenshots, and analytics.
- Click `Stop` to end the session.

### Local Camera Mode

- Choose `Local Camera` in the sidebar.
- Click `Start Camera` to begin monitoring from the machine running Streamlit.
- Use this only when the app is running on a device that physically has the camera attached.

### Upload Video Mode

- Choose `Upload Video` in the sidebar.
- Upload an MP4, AVI, MOV, or MKV file.
- Click `Analyze` to process the video.
- Export reports once processing completes.

## Outputs

During a session, the app may create:

- Annotated frame previews inside the UI.
- `output.avi` recorded session video.
- Alert screenshots under `screenshots/`.
- CSV report export.
- TXT report export.

## Configuration Notes

- The model is loaded automatically from `yolov8s.pt` on first run.
- Windows users can hear the alarm sound through `winsound`.
- The browser webcam mode is the deployment-safe option and should be used on Streamlit Cloud.
- The local camera mode uses webcam index `0` and is only for machines that physically have the camera attached.
- The first startup may take longer while the model is downloaded.

## Troubleshooting

- If the camera does not open, close other apps that may be using the webcam.
- If the model fails to load, confirm that `ultralytics` is installed and the machine can reach the internet for the first download.
- If Streamlit does not open automatically, manually visit `http://localhost:8501`.

## Roadmap

- Multi-camera support.
- Face recognition.
- Cloud deployment.
- Mobile notifications.
- Role-based access and audit logs.

## Author

Manikanta Perla
