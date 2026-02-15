import base64
from collections import Counter, deque
import os
import threading
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

try:
    import yaml
except Exception:
    yaml = None

try:
    from fer import FER as FERDetector
except Exception:
    try:
        from fer.fer import FER as FERDetector
    except Exception:
        FERDetector = None

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

user_logged_in = True

MODEL_DIR = "python_Scripts"
YOLO_WEIGHTS = os.path.join(MODEL_DIR, "yolov3-tiny.weights")
YOLO_CFG = os.path.join(MODEL_DIR, "yolov3-tiny.cfg")


COCO_NAMES = os.path.join(MODEL_DIR, "coco.names")
CONFIG_PATH = "config.yaml"



DEFAULT_CONFIG: Dict[str, Any] = {
    "runtime": {
        "client_state_ttl_seconds": 300,
        "jpeg_quality": 72,
    },
    "tracking": {
        "max_distance": 85,
        "max_missed_frames": 12,
    },
    "yolo": {
        "profiles": {
            "fast": {"input_size": 416, "confidence": 0.50, "nms_threshold": 0.45},
            "balanced": {"input_size": 544, "confidence": 0.42, "nms_threshold": 0.45},
            "accurate": {"input_size": 608, "confidence": 0.35, "nms_threshold": 0.45},
        },
        "default_profile": "balanced",
    },
    "modes": {
        "object": {
            "default_profile": "balanced",
            "min_area_ratio": 0.0006,
            "count_ema_alpha": 0.35,
        },
        "human": {
            "default_profile": "balanced",
            "min_area_ratio": 0.0008,
            "count_ema_alpha": 0.30,
        },
        "vehicle": {
            "default_profile": "balanced",
            "min_area_ratio": 0.0012,
            "count_ema_alpha": 0.35,
        },
        "emotion": {
            "history_size": 10,
            "top_k": 3,
        },
        "movement": {
            "history": 700,
            "var_threshold": 30,
            "warmup_frames": 25,
            "min_area_ratio": 0.0025,
        },
    },
}


def deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    config = dict(DEFAULT_CONFIG)

    if not os.path.exists(path) or yaml is None:
        return config

    try:
        with open(path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if isinstance(loaded, dict):
            config = deep_merge(config, loaded)
    except Exception as exc:
        print(f"Config load warning: {exc}")

    return config


APP_CONFIG = load_config()

YOLO_NET = None
YOLO_OUTPUT_LAYERS = None
COCO_CLASSES = None
YOLO_LOCK = threading.Lock()

PERSON_LABELS = {"person"}
VEHICLE_LABELS = {"car", "motorbike", "bus", "truck"}
SUPPORTED_MODES = {"object", "human", "vehicle", "movement", "emotion"}

EMOTION_DETECTOR = None
EMOTION_DETECTOR_LOCK = threading.Lock()
FACE_CASCADE = None
EYE_CASCADE = None
SMILE_CASCADE = None
CASCADE_LOCK = threading.Lock()

CLIENT_MODE_STATE = {}
CLIENT_MODE_STATE_LOCK = threading.Lock()
CLIENT_STATE_TTL_SECONDS = int(APP_CONFIG.get("runtime", {}).get("client_state_ttl_seconds", 300))

RUNTIME_METRICS = {
    "started_at": time.time(),
    "modes": {
        mode: {
            "requests": 0,
            "errors": 0,
            "total_latency_ms": 0.0,
            "last_latency_ms": 0.0,
            "last_meta": {},
            "last_updated": None,
        }
        for mode in SUPPORTED_MODES
    },
}
RUNTIME_METRICS_LOCK = threading.Lock()


def update_runtime_metrics(mode: str, latency_ms: float, error: bool, meta: Dict[str, Any] = None) -> None:
    with RUNTIME_METRICS_LOCK:
        payload = RUNTIME_METRICS["modes"].setdefault(
            mode,
            {
                "requests": 0,
                "errors": 0,
                "total_latency_ms": 0.0,
                "last_latency_ms": 0.0,
                "last_meta": {},
                "last_updated": None,
            },
        )

        payload["requests"] += 1
        payload["total_latency_ms"] += float(latency_ms)
        payload["last_latency_ms"] = float(latency_ms)
        payload["last_updated"] = time.time()
        if error:
            payload["errors"] += 1
        if meta is not None:
            payload["last_meta"] = meta


def runtime_snapshot() -> Dict[str, Any]:
    with RUNTIME_METRICS_LOCK:
        now = time.time()
        modes = {}
        for mode, payload in RUNTIME_METRICS["modes"].items():
            requests_count = payload["requests"]
            avg_latency = payload["total_latency_ms"] / requests_count if requests_count else 0.0
            modes[mode] = {
                "requests": requests_count,
                "errors": payload["errors"],
                "avg_latency_ms": round(avg_latency, 2),
                "last_latency_ms": round(payload["last_latency_ms"], 2),
                "last_updated": payload["last_updated"],
                "last_meta": payload["last_meta"],
            }

        return {
            "started_at": RUNTIME_METRICS["started_at"],
            "uptime_seconds": round(now - RUNTIME_METRICS["started_at"], 2),
            "modes": modes,
        }


def encode_mjpeg_frame(frame):
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        return b""

    payload = buffer.tobytes()
    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"


def encode_frame_data_url(frame, quality=72):
    ok, buffer = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        return None

    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def decode_image_data_url(data_url):
    if not data_url or not isinstance(data_url, str):
        return None

    if "," in data_url:
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url

    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except Exception:
        return None

    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    return frame


def open_camera(index=0):
    backends = []
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(cv2.CAP_DSHOW)
    if hasattr(cv2, "CAP_MSMF"):
        backends.append(cv2.CAP_MSMF)
    backends.append(None)

    for backend in backends:
        cap = cv2.VideoCapture(index, backend) if backend is not None else cv2.VideoCapture(index)

        if cap is not None and cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap

        if cap is not None:
            cap.release()

    raise RuntimeError("Could not access webcam. Close apps like Zoom/Teams/Camera and retry.")


def load_coco_classes():
    global COCO_CLASSES

    if COCO_CLASSES is None:
        if not os.path.exists(COCO_NAMES):
            raise FileNotFoundError(f"Missing labels file: {COCO_NAMES}")

        with open(COCO_NAMES, "r", encoding="utf-8") as file:
            COCO_CLASSES = [line.strip() for line in file if line.strip()]

    return COCO_CLASSES


def load_yolo():
    global YOLO_NET, YOLO_OUTPUT_LAYERS

    required_files = [YOLO_WEIGHTS, YOLO_CFG, COCO_NAMES]
    missing = [path for path in required_files if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing model files: {', '.join(missing)}")

    if YOLO_NET is None or YOLO_OUTPUT_LAYERS is None:
        YOLO_NET = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CFG)
        YOLO_NET.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        YOLO_NET.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        layer_names = YOLO_NET.getLayerNames()
        output_layer_ids = np.array(YOLO_NET.getUnconnectedOutLayers()).flatten()
        YOLO_OUTPUT_LAYERS = [layer_names[int(index) - 1] for index in output_layer_ids]

        load_coco_classes()

    return YOLO_NET, YOLO_OUTPUT_LAYERS, COCO_CLASSES


def list_model_profiles() -> List[str]:
    profiles = APP_CONFIG.get("yolo", {}).get("profiles", {})
    return list(profiles.keys())


def resolve_profile_name(mode: str, requested_profile: str = None) -> str:
    available = list_model_profiles()
    default_profile = APP_CONFIG.get("modes", {}).get(mode, {}).get(
        "default_profile",
        APP_CONFIG.get("yolo", {}).get("default_profile", "balanced"),
    )

    if requested_profile in available:
        return requested_profile
    if default_profile in available:
        return default_profile
    return available[0] if available else "balanced"


def yolo_params_for(mode: str, requested_profile: str = None) -> Dict[str, Any]:
    profile_name = resolve_profile_name(mode, requested_profile)
    profile = APP_CONFIG.get("yolo", {}).get("profiles", {}).get(profile_name, {})
    mode_cfg = APP_CONFIG.get("modes", {}).get(mode, {})

    return {
        "profile": profile_name,
        "input_size": int(profile.get("input_size", 608)),
        "conf_threshold": float(profile.get("confidence", 0.4)),
        "nms_threshold": float(profile.get("nms_threshold", 0.45)),
        "min_area_ratio": float(mode_cfg.get("min_area_ratio", 0.001)),
    }


def detect_yolo(
    frame,
    allowed_labels=None,
    conf_threshold=0.4,
    nms_threshold=0.45,
    input_size=608,
    min_area_ratio=0.001,
):
    net, output_layers, classes = load_yolo()

    frame_height, frame_width = frame.shape[:2]
    min_area = int(frame_height * frame_width * min_area_ratio)
    allowed = set(allowed_labels) if allowed_labels else None

    blob = cv2.dnn.blobFromImage(
        frame,
        scalefactor=1 / 255.0,
        size=(input_size, input_size),
        mean=(0, 0, 0),
        swapRB=True,
        crop=False,
    )

    with YOLO_LOCK:
        net.setInput(blob)
        outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            class_scores = detection[5:]
            class_id = int(np.argmax(class_scores))
            class_prob = float(class_scores[class_id])
            objectness = float(detection[4])
            confidence = class_prob * objectness

            if confidence < conf_threshold:
                continue

            label = classes[class_id] if class_id < len(classes) else str(class_id)
            if allowed is not None and label not in allowed:
                continue

            box_w = int(detection[2] * frame_width)
            box_h = int(detection[3] * frame_height)
            if box_w <= 0 or box_h <= 0:
                continue
            if box_w * box_h < min_area:
                continue

            center_x = int(detection[0] * frame_width)
            center_y = int(detection[1] * frame_height)

            x = max(0, int(center_x - box_w / 2))
            y = max(0, int(center_y - box_h / 2))
            box_w = min(box_w, frame_width - x)
            box_h = min(box_h, frame_height - y)

            boxes.append([x, y, box_w, box_h])
            confidences.append(confidence)
            class_ids.append(class_id)

    detections = []
    if boxes:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        if len(indices) > 0:
            for idx in np.array(indices).flatten():
                class_id = class_ids[idx]
                label = classes[class_id] if class_id < len(classes) else str(class_id)
                detections.append({"box": boxes[idx], "confidence": confidences[idx], "label": label})

    return detections


def bbox_center(box: List[int]) -> Tuple[float, float]:
    x, y, w, h = box
    return x + (w / 2.0), y + (h / 2.0)


def update_tracks(detections: List[Dict[str, Any]], state: Dict[str, Any]) -> Tuple[int, int]:
    tracking_cfg = APP_CONFIG.get("tracking", {})
    max_distance = float(tracking_cfg.get("max_distance", 85))
    max_missed_frames = int(tracking_cfg.get("max_missed_frames", 12))

    tracks = state.setdefault("tracks", {})
    next_track_id = int(state.get("next_track_id", 1))

    centers = [bbox_center(det["box"]) for det in detections]
    assigned_track_ids = set()
    assigned_detection_ids = set()

    candidates = []
    for track_id, track in tracks.items():
        track_center = track.get("center", (0.0, 0.0))
        for det_idx, center in enumerate(centers):
            distance = float(np.linalg.norm(np.array(track_center) - np.array(center)))
            candidates.append((distance, track_id, det_idx))

    for distance, track_id, det_idx in sorted(candidates, key=lambda entry: entry[0]):
        if distance > max_distance:
            continue
        if track_id in assigned_track_ids or det_idx in assigned_detection_ids:
            continue

        assigned_track_ids.add(track_id)
        assigned_detection_ids.add(det_idx)
        tracks[track_id]["center"] = centers[det_idx]
        tracks[track_id]["missed"] = 0
        detections[det_idx]["track_id"] = track_id

    for det_idx, center in enumerate(centers):
        if det_idx in assigned_detection_ids:
            continue

        track_id = next_track_id
        next_track_id += 1
        tracks[track_id] = {"center": center, "missed": 0}
        detections[det_idx]["track_id"] = track_id
        assigned_track_ids.add(track_id)

    drop_ids = []
    for track_id, track in tracks.items():
        if track_id not in assigned_track_ids:
            track["missed"] = int(track.get("missed", 0)) + 1
            if track["missed"] > max_missed_frames:
                drop_ids.append(track_id)

    for track_id in drop_ids:
        tracks.pop(track_id, None)

    state["next_track_id"] = next_track_id
    unique_ids = state.setdefault("unique_track_ids", set())
    for detection in detections:
        if "track_id" in detection:
            unique_ids.add(detection["track_id"])

    return len(tracks), len(unique_ids)


def draw_detections(frame, detections, box_color=(0, 220, 70), text_color=(20, 20, 230)):
    for detection in detections:
        x, y, w, h = detection["box"]
        label = detection["label"]
        confidence = detection["confidence"]
        track_id = detection.get("track_id")

        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
        prefix = f"ID {track_id} " if track_id is not None else ""
        caption = f"{prefix}{label} {confidence * 100:.1f}%"
        cv2.putText(
            frame,
            caption,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            text_color,
            2,
            cv2.LINE_AA,
        )


def get_emotion_detector():
    global EMOTION_DETECTOR

    if EMOTION_DETECTOR is not None:
        return EMOTION_DETECTOR

    with EMOTION_DETECTOR_LOCK:
        if EMOTION_DETECTOR is not None:
            return EMOTION_DETECTOR

        if FERDetector is None:
            EMOTION_DETECTOR = None
        else:
            try:
                EMOTION_DETECTOR = FERDetector(mtcnn=False)
            except Exception as exc:
                print(f"FER initialization failed, cascade fallback enabled: {exc}")
                EMOTION_DETECTOR = None

    return EMOTION_DETECTOR


def get_haar_cascades():
    global FACE_CASCADE, EYE_CASCADE, SMILE_CASCADE

    if FACE_CASCADE is not None and EYE_CASCADE is not None and SMILE_CASCADE is not None:
        return FACE_CASCADE, EYE_CASCADE, SMILE_CASCADE

    with CASCADE_LOCK:
        if FACE_CASCADE is None:
            FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if EYE_CASCADE is None:
            EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        if SMILE_CASCADE is None:
            SMILE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

    return FACE_CASCADE, EYE_CASCADE, SMILE_CASCADE


def detect_emotion_with_cascades(frame):
    face_cascade, eye_cascade, smile_cascade = get_haar_cascades()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40))

    results = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y : y + h, x : x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=8, minSize=(15, 15))
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=24, minSize=(25, 25))

        if len(smiles) > 0:
            emotion = "happy"
            score = 0.70
        elif len(eyes) >= 2:
            emotion = "neutral"
            score = 0.55
        else:
            emotion = "unknown"
            score = 0.40

        results.append(
            {
                "box": [x, y, w, h],
                "label": emotion,
                "score": score,
                "top_emotions": [(emotion, score)],
            }
        )

    return results


def cleanup_client_states(now):
    stale_clients = []
    for client_id, payload in CLIENT_MODE_STATE.items():
        if now - payload.get("updated_at", now) > CLIENT_STATE_TTL_SECONDS:
            stale_clients.append(client_id)

    for client_id in stale_clients:
        CLIENT_MODE_STATE.pop(client_id, None)


def create_movement_state():
    movement_cfg = APP_CONFIG.get("modes", {}).get("movement", {})
    return {
        "subtractor": cv2.createBackgroundSubtractorMOG2(
            history=int(movement_cfg.get("history", 700)),
            varThreshold=float(movement_cfg.get("var_threshold", 30)),
            detectShadows=False,
        ),
        "open_kernel": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        "close_kernel": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        "dilate_kernel": cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        "frame_index": 0,
        "warmup_frames": int(movement_cfg.get("warmup_frames", 25)),
    }


def create_mode_state(mode):
    if mode in {"object", "human", "vehicle"}:
        return {
            "count_ema": 0.0,
            "tracks": {},
            "next_track_id": 1,
            "unique_track_ids": set(),
            "_last_frame_ts": None,
            "_fps_ema": 0.0,
        }
    if mode == "emotion":
        history_size = int(APP_CONFIG.get("modes", {}).get("emotion", {}).get("history_size", 10))
        return {"history": deque(maxlen=history_size), "_last_frame_ts": None, "_fps_ema": 0.0}
    if mode == "movement":
        state = create_movement_state()
        state["_last_frame_ts"] = None
        state["_fps_ema"] = 0.0
        return state
    return {"_last_frame_ts": None, "_fps_ema": 0.0}


def update_state_fps(state: Dict[str, Any]) -> float:
    now = time.time()
    last_ts = state.get("_last_frame_ts")
    if last_ts:
        delta = max(now - last_ts, 1e-3)
        instant_fps = 1.0 / delta
        prev = float(state.get("_fps_ema", instant_fps))
        state["_fps_ema"] = (0.8 * prev) + (0.2 * instant_fps)
    state["_last_frame_ts"] = now
    return float(state.get("_fps_ema", 0.0))


def get_client_mode_state(client_id, mode):
    now = time.time()

    with CLIENT_MODE_STATE_LOCK:
        cleanup_client_states(now)

        client_payload = CLIENT_MODE_STATE.setdefault(client_id, {"updated_at": now, "modes": {}})
        client_payload["updated_at"] = now

        mode_states = client_payload["modes"]
        if mode not in mode_states:
            mode_states[mode] = create_mode_state(mode)

        return mode_states[mode]


def process_object_frame(frame, state, requested_profile=None):
    params = yolo_params_for("object", requested_profile)
    detections = detect_yolo(
        frame,
        allowed_labels=None,
        conf_threshold=params["conf_threshold"],
        nms_threshold=params["nms_threshold"],
        input_size=params["input_size"],
        min_area_ratio=params["min_area_ratio"],
    )

    alpha = float(APP_CONFIG.get("modes", {}).get("object", {}).get("count_ema_alpha", 0.35))
    state["count_ema"] = ((1.0 - alpha) * state.get("count_ema", 0.0)) + (alpha * len(detections))
    count = int(round(state["count_ema"]))

    draw_detections(frame, detections)
    cv2.putText(frame, f"Objects: {count}", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return frame, {
        "count": count,
        "raw_count": len(detections),
        "profile": params["profile"],
        "input_size": params["input_size"],
    }


def process_human_frame(frame, state, requested_profile=None):
    params = yolo_params_for("human", requested_profile)
    detections = detect_yolo(
        frame,
        allowed_labels=PERSON_LABELS,
        conf_threshold=params["conf_threshold"],
        nms_threshold=params["nms_threshold"],
        input_size=params["input_size"],
        min_area_ratio=params["min_area_ratio"],
    )

    active_tracks, unique_tracks = update_tracks(detections, state)
    alpha = float(APP_CONFIG.get("modes", {}).get("human", {}).get("count_ema_alpha", 0.30))
    state["count_ema"] = ((1.0 - alpha) * state.get("count_ema", 0.0)) + (alpha * len(detections))
    count = int(round(state["count_ema"]))

    draw_detections(frame, detections, box_color=(255, 80, 0), text_color=(255, 255, 255))
    cv2.putText(frame, f"Humans: {count}", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"Tracked: {active_tracks} | Unique: {unique_tracks}",
        (12, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 215, 255),
        2,
        cv2.LINE_AA,
    )

    return frame, {
        "count": count,
        "raw_count": len(detections),
        "tracked": active_tracks,
        "unique": unique_tracks,
        "profile": params["profile"],
        "input_size": params["input_size"],
    }


def process_vehicle_frame(frame, state, requested_profile=None):
    params = yolo_params_for("vehicle", requested_profile)
    detections = detect_yolo(
        frame,
        allowed_labels=VEHICLE_LABELS,
        conf_threshold=params["conf_threshold"],
        nms_threshold=params["nms_threshold"],
        input_size=params["input_size"],
        min_area_ratio=params["min_area_ratio"],
    )

    active_tracks, unique_tracks = update_tracks(detections, state)
    alpha = float(APP_CONFIG.get("modes", {}).get("vehicle", {}).get("count_ema_alpha", 0.35))
    state["count_ema"] = ((1.0 - alpha) * state.get("count_ema", 0.0)) + (alpha * len(detections))
    count = int(round(state["count_ema"]))

    draw_detections(frame, detections, box_color=(255, 120, 0), text_color=(255, 255, 255))
    cv2.putText(frame, f"Vehicles: {count}", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"Tracked: {active_tracks} | Unique: {unique_tracks}",
        (12, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 215, 255),
        2,
        cv2.LINE_AA,
    )

    return frame, {
        "count": count,
        "raw_count": len(detections),
        "tracked": active_tracks,
        "unique": unique_tracks,
        "profile": params["profile"],
        "input_size": params["input_size"],
    }


def process_emotion_frame(frame, state, requested_profile=None):
    del requested_profile
    predictions = []
    detector = get_emotion_detector()
    emotion_cfg = APP_CONFIG.get("modes", {}).get("emotion", {})
    top_k = int(emotion_cfg.get("top_k", 3))

    if detector is not None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_predictions = detector.detect_emotions(rgb_frame)

        for item in raw_predictions:
            x, y, w, h = item.get("box", [0, 0, 0, 0])
            if w <= 0 or h <= 0:
                continue

            emotions = item.get("emotions", {})
            if not emotions:
                continue

            sorted_emotions = sorted(emotions.items(), key=lambda entry: entry[1], reverse=True)
            label, score = sorted_emotions[0]
            predictions.append(
                {
                    "box": [max(0, x), max(0, y), w, h],
                    "label": label,
                    "score": float(score),
                    "top_emotions": [(name, float(value)) for name, value in sorted_emotions[:top_k]],
                }
            )
    else:
        predictions = detect_emotion_with_cascades(frame)

    top_label = None
    top_summary = []
    if predictions:
        predictions.sort(key=lambda entry: entry["box"][2] * entry["box"][3], reverse=True)
        history = state.setdefault("history", deque(maxlen=int(emotion_cfg.get("history_size", 10))))
        history.append(predictions[0]["label"])
        top_label = Counter(history).most_common(1)[0][0]
        predictions[0]["label"] = top_label
        if predictions[0].get("top_emotions"):
            top_summary = [f"{name}:{score * 100:.1f}%" for name, score in predictions[0]["top_emotions"]]

    for prediction in predictions:
        x, y, w, h = prediction["box"]
        label = prediction["label"]
        score = prediction["score"]

        color = (0, 210, 255) if label in {"neutral", "unknown", "sad"} else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame,
            f"{label} {score * 100:.1f}%",
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        top_emotions = prediction.get("top_emotions", [])
        if top_emotions:
            second_line = " | ".join([f"{name}:{value * 100:.0f}%" for name, value in top_emotions[:top_k]])
            cv2.putText(
                frame,
                second_line,
                (x, min(frame.shape[0] - 8, y + h + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )

    if not predictions:
        cv2.putText(frame, "No face detected", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2, cv2.LINE_AA)

    return frame, {
        "faces": len(predictions),
        "emotion": top_label,
        "top_emotions": top_summary,
        "model": "FER" if detector is not None else "CascadeFallback",
    }


def process_movement_frame(frame, state, requested_profile=None):
    del requested_profile
    movement_cfg = APP_CONFIG.get("modes", {}).get("movement", {})
    frame = cv2.resize(frame, (960, 540))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    subtractor = state["subtractor"]
    fg_mask = subtractor.apply(gray)
    state["frame_index"] = state.get("frame_index", 0) + 1

    if state["frame_index"] <= state["warmup_frames"]:
        cv2.putText(
            frame,
            "Calibrating background...",
            (12, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return frame, {"moving_regions": 0, "warming_up": True}

    _, fg_mask = cv2.threshold(fg_mask, 245, 255, cv2.THRESH_BINARY)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, state["open_kernel"], iterations=2)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, state["close_kernel"], iterations=2)
    fg_mask = cv2.dilate(fg_mask, state["dilate_kernel"], iterations=1)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(1600, int(frame.shape[0] * frame.shape[1] * float(movement_cfg.get("min_area_ratio", 0.0025))))

    moving_regions = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if h == 0:
            continue

        aspect_ratio = w / float(h)
        if aspect_ratio > 14.0 or aspect_ratio < 0.08:
            continue

        moving_regions += 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(
        frame,
        f"Moving regions: {moving_regions}",
        (12, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return frame, {"moving_regions": moving_regions, "warming_up": False}


MODE_PROCESSORS = {
    "object": process_object_frame,
    "human": process_human_frame,
    "vehicle": process_vehicle_frame,
    "emotion": process_emotion_frame,
    "movement": process_movement_frame,
}


def get_client_id(payload):
    client_id = payload.get("client_id") if isinstance(payload, dict) else None
    if client_id:
        return str(client_id)

    forwarded_for = request.headers.get("X-Forwarded-For", "")
    source = forwarded_for.split(",")[0].strip() if forwarded_for else ""
    if source:
        return source

    return request.remote_addr or "anonymous"


def process_frame_by_mode(mode, frame, state, profile_name=None):
    if mode not in MODE_PROCESSORS:
        raise ValueError(f"Unsupported mode: {mode}")

    processor = MODE_PROCESSORS[mode]
    return processor(frame, state, profile_name)


def generate_mode_frames(mode):
    cap = None
    state = create_mode_state(mode)

    try:
        cap = open_camera(0)
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            processed_frame, _ = process_frame_by_mode(mode, frame, state)
            encoded = encode_mjpeg_frame(processed_frame)
            if not encoded:
                break

            yield encoded
    except Exception as exc:
        print(f"Error in {mode}_video_feed: {exc}")
        yield b""
    finally:
        if cap is not None:
            cap.release()


@app.route("/")
def home():
    return render_template("home.html", user_logged_in=user_logged_in)


@app.route("/benchmark")
def benchmark_page():
    return render_template("benchmark.html", user_logged_in=user_logged_in)


def render_detector_page(title, mode):
    return render_template(
        "detector.html",
        user_logged_in=user_logged_in,
        detector_title=title,
        detector_mode=mode,
        detector_profiles=list_model_profiles(),
        detector_default_profile=resolve_profile_name(mode, None),
    )


@app.route("/object")
def object():
    return render_detector_page("Object Detection", "object")


@app.route("/human")
def human():
    return render_detector_page("Human Detection", "human")


@app.route("/vehicle")
def vehicle():
    return render_detector_page("Vehicle Detection", "vehicle")


@app.route("/movement")
def movement():
    return render_detector_page("Movement Detection", "movement")


@app.route("/emotion")
def emotion():
    return render_detector_page("Emotion Detection", "emotion")


@app.route("/api/health")
def health_api():
    snapshot = runtime_snapshot()
    return jsonify({"ok": True, "message": "healthy", "uptime_seconds": snapshot["uptime_seconds"]})


@app.route("/api/config")
def config_api():
    return jsonify({"ok": True, "config": APP_CONFIG, "profiles": list_model_profiles()})


@app.route("/api/metrics")
def metrics_api():
    return jsonify({"ok": True, "metrics": runtime_snapshot()})


@app.route("/api/detect/<mode>", methods=["POST"])
def detect_api(mode):
    if mode not in SUPPORTED_MODES:
        return jsonify({"ok": False, "error": "Unsupported mode"}), 404

    payload = request.get_json(silent=True) or {}
    frame_data = payload.get("frame")
    requested_profile = payload.get("profile")
    if not frame_data:
        return jsonify({"ok": False, "error": "Missing frame payload"}), 400

    frame = decode_image_data_url(frame_data)
    if frame is None:
        return jsonify({"ok": False, "error": "Invalid frame data"}), 400

    started_at = time.perf_counter()
    try:
        client_id = get_client_id(payload)
        state = get_client_mode_state(client_id, mode)
        fps_estimate = update_state_fps(state)

        processed_frame, meta = process_frame_by_mode(mode, frame, state, requested_profile)
        meta = meta or {}
        meta["fps_estimate"] = round(fps_estimate, 2)

        jpeg_quality = int(APP_CONFIG.get("runtime", {}).get("jpeg_quality", 72))
        encoded_image = encode_frame_data_url(processed_frame, quality=jpeg_quality)

        if encoded_image is None:
            latency_ms = (time.perf_counter() - started_at) * 1000.0
            update_runtime_metrics(mode, latency_ms, error=True)
            return jsonify({"ok": False, "error": "Frame encoding failed"}), 500

        latency_ms = (time.perf_counter() - started_at) * 1000.0
        meta["latency_ms"] = round(latency_ms, 2)
        update_runtime_metrics(mode, latency_ms, error=False, meta=meta)

        return jsonify({"ok": True, "mode": mode, "image": encoded_image, "meta": meta})
    except Exception as exc:
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        update_runtime_metrics(mode, latency_ms, error=True)
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/object/video_feed")
def object_video_feed():
    return Response(generate_mode_frames("object"), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/human/video_feed")
def human_video_feed():
    return Response(generate_mode_frames("human"), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/vehicle/video_feed")
def vehicle_video_feed():
    return Response(generate_mode_frames("vehicle"), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/movement/video_feed")
def movement_video_feed():
    return Response(generate_mode_frames("movement"), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/emotion/video_feed")
def emotion_video_feed():
    return Response(generate_mode_frames("emotion"), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/sign")
def sign():
    return render_template("sign.html", user_logged_in=user_logged_in)


@app.route("/sign/video_feed")
def sign_video_feed():
    def generate_sign_frames():
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            "Sign detection is not implemented yet",
            (24, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        encoded = encode_mjpeg_frame(frame)
        if encoded:
            while True:
                yield encoded

    return Response(generate_sign_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
