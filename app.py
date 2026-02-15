import base64
from collections import Counter, deque
import os
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

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
YOLO_WEIGHTS = os.path.join(MODEL_DIR, "yolov3.weights")
YOLO_CFG = os.path.join(MODEL_DIR, "yolov3.cfg")
COCO_NAMES = os.path.join(MODEL_DIR, "coco.names")

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
CLIENT_STATE_TTL_SECONDS = 300


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


def draw_detections(frame, detections, box_color=(0, 220, 70), text_color=(20, 20, 230)):
    for detection in detections:
        x, y, w, h = detection["box"]
        label = detection["label"]
        confidence = detection["confidence"]

        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
        caption = f"{label} {confidence * 100:.1f}%"
        cv2.putText(
            frame,
            caption,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
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

        results.append({"box": [x, y, w, h], "label": emotion, "score": score})

    return results


def cleanup_client_states(now):
    stale_clients = []
    for client_id, payload in CLIENT_MODE_STATE.items():
        if now - payload.get("updated_at", now) > CLIENT_STATE_TTL_SECONDS:
            stale_clients.append(client_id)

    for client_id in stale_clients:
        CLIENT_MODE_STATE.pop(client_id, None)


def create_movement_state():
    return {
        "subtractor": cv2.createBackgroundSubtractorMOG2(history=700, varThreshold=30, detectShadows=False),
        "open_kernel": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        "close_kernel": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        "dilate_kernel": cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        "frame_index": 0,
        "warmup_frames": 25,
    }


def create_mode_state(mode):
    if mode in {"object", "human", "vehicle"}:
        return {"count_ema": 0.0}
    if mode == "emotion":
        return {"history": deque(maxlen=8)}
    if mode == "movement":
        return create_movement_state()
    return {}


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


def process_object_frame(frame, state):
    detections = detect_yolo(
        frame,
        allowed_labels=None,
        conf_threshold=0.35,
        nms_threshold=0.45,
        input_size=608,
        min_area_ratio=0.0006,
    )

    state["count_ema"] = (0.65 * state.get("count_ema", 0.0)) + (0.35 * len(detections))
    count = int(round(state["count_ema"]))

    draw_detections(frame, detections)
    cv2.putText(frame, f"Objects: {count}", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return frame, {"count": count, "raw_count": len(detections)}


def process_human_frame(frame, state):
    detections = detect_yolo(
        frame,
        allowed_labels=PERSON_LABELS,
        conf_threshold=0.42,
        nms_threshold=0.45,
        input_size=608,
        min_area_ratio=0.0008,
    )

    state["count_ema"] = (0.70 * state.get("count_ema", 0.0)) + (0.30 * len(detections))
    count = int(round(state["count_ema"]))

    draw_detections(frame, detections, box_color=(255, 80, 0), text_color=(255, 255, 255))
    cv2.putText(frame, f"Humans: {count}", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return frame, {"count": count, "raw_count": len(detections)}


def process_vehicle_frame(frame, state):
    detections = detect_yolo(
        frame,
        allowed_labels=VEHICLE_LABELS,
        conf_threshold=0.42,
        nms_threshold=0.45,
        input_size=608,
        min_area_ratio=0.0012,
    )

    state["count_ema"] = (0.65 * state.get("count_ema", 0.0)) + (0.35 * len(detections))
    count = int(round(state["count_ema"]))

    draw_detections(frame, detections, box_color=(255, 120, 0), text_color=(255, 255, 255))
    cv2.putText(frame, f"Vehicles: {count}", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return frame, {"count": count, "raw_count": len(detections)}


def process_emotion_frame(frame, state):
    predictions = []
    detector = get_emotion_detector()

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

            label, score = max(emotions.items(), key=lambda entry: entry[1])
            predictions.append({"box": [max(0, x), max(0, y), w, h], "label": label, "score": float(score)})
    else:
        predictions = detect_emotion_with_cascades(frame)

    top_label = None
    if predictions:
        predictions.sort(key=lambda entry: entry["box"][2] * entry["box"][3], reverse=True)
        history = state.setdefault("history", deque(maxlen=8))
        history.append(predictions[0]["label"])
        top_label = Counter(history).most_common(1)[0][0]
        predictions[0]["label"] = top_label

    for prediction in predictions:
        x, y, w, h = prediction["box"]
        label = prediction["label"]
        score = prediction["score"]

        color = (0, 210, 255) if label in {"neutral", "unknown"} else (0, 255, 0)
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

    if not predictions:
        cv2.putText(frame, "No face detected", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2, cv2.LINE_AA)

    return frame, {"faces": len(predictions), "emotion": top_label}


def process_movement_frame(frame, state):
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
    min_area = max(1600, int(frame.shape[0] * frame.shape[1] * 0.0025))

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


def process_frame_by_mode(mode, frame, state):
    if mode not in MODE_PROCESSORS:
        raise ValueError(f"Unsupported mode: {mode}")

    processor = MODE_PROCESSORS[mode]
    return processor(frame, state)


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


def render_detector_page(title, mode):
    return render_template(
        "detector.html",
        user_logged_in=user_logged_in,
        detector_title=title,
        detector_mode=mode,
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


@app.route("/api/detect/<mode>", methods=["POST"])
def detect_api(mode):
    if mode not in SUPPORTED_MODES:
        return jsonify({"ok": False, "error": "Unsupported mode"}), 404

    payload = request.get_json(silent=True) or {}
    frame_data = payload.get("frame")
    if not frame_data:
        return jsonify({"ok": False, "error": "Missing frame payload"}), 400

    frame = decode_image_data_url(frame_data)
    if frame is None:
        return jsonify({"ok": False, "error": "Invalid frame data"}), 400

    try:
        client_id = get_client_id(payload)
        state = get_client_mode_state(client_id, mode)

        processed_frame, meta = process_frame_by_mode(mode, frame, state)
        encoded_image = encode_frame_data_url(processed_frame, quality=72)

        if encoded_image is None:
            return jsonify({"ok": False, "error": "Frame encoding failed"}), 500

        return jsonify({"ok": True, "mode": mode, "image": encoded_image, "meta": meta})
    except Exception as exc:
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
