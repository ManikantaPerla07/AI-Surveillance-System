import csv
import io
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Union


def ensure_folder(path: Union[str, Path]) -> Path:
    """Create a folder if it does not exist and return a Path object."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logger(name: str = "surveillance", level: int = logging.INFO) -> logging.Logger:
    """Create a logger that is safe across import boundaries."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def get_timestamp_string() -> str:
    """Return a human-readable timestamp string for filenames."""
    return time.strftime("%Y%m%d_%H%M%S")


def report_to_csv(report: Dict[str, Any]) -> str:
    """Convert the report dictionary into a CSV string."""
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["field", "value"])
    for key in ["video_path", "annotated_video_path", "frames_processed", "duration_seconds", "processing_fps", "total_alerts", "screenshots_saved"]:
        writer.writerow([key, report.get(key, "")])

    writer.writerow([])
    writer.writerow(["alert_index", "frame_index", "track_id", "label", "confidence", "threat_score", "screenshot_path"])
    for index, alert in enumerate(report.get("alert_records", []), start=1):
        writer.writerow([
            index,
            alert.get("frame_index", ""),
            alert.get("track_id", ""),
            alert.get("label", ""),
            alert.get("confidence", ""),
            alert.get("threat_score", ""),
            alert.get("screenshot_path", ""),
        ])

    return output.getvalue()


def report_to_json(report: Dict[str, Any]) -> str:
    """Convert the report dictionary into a JSON string."""
    return json.dumps(report, indent=2)
