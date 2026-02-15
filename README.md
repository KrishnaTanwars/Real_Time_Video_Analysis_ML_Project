# Realtime Video Analysis

A comprehensive real-time video analysis system that can perform multiple computer vision tasks including human detection, object detection, emotion recognition, and vehicle detection.

## Features

- Human Detection
- Object Detection
- Emotion Recognition
- Vehicle Detection
- Motion Detection
- Real-time Video Processing
- Web-based Interface
- Config-driven model profiles (`fast`, `balanced`, `accurate`)
- Tracking IDs and unique counts for human/vehicle modes
- Runtime benchmark metrics API (`/api/metrics`)
- Offline evaluation pipeline (`evaluate.py`)

## Prerequisites

- Python 3.7+
- Webcam or Video Input Device
- Modern Web Browser

## Installation

1. Clone the repository:
```bash
git clone https://github.com/[your-username]/realtime-video-analysis.git
cd realtime-video-analysis
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Download YOLOv3 weights:
   - Download the YOLOv3 weights file from [here](https://pjreddie.com/media/files/yolov3.weights)
   - Place the downloaded file in the `python_Scripts` directory

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Select the desired analysis mode:
   - Human Detection
   - Object Detection
   - Emotion Recognition
   - Vehicle Detection
   - Motion Detection

4. Optional benchmark dashboard:
   - Open `http://localhost:5000/benchmark`

5. API endpoints:
   - `GET /api/health`
   - `GET /api/config`
   - `GET /api/metrics`
   - `POST /api/detect/<mode>` with JSON `{ "client_id": "...", "profile": "balanced", "frame": "data:image/jpeg;base64,..." }`

## Browser Camera (Deployment-Ready)

This project now supports browser-based camera inference for cloud deployments.

- The frontend captures frames using `getUserMedia`.
- Frames are sent to backend API routes:
  - `POST /api/detect/object`
  - `POST /api/detect/human`
  - `POST /api/detect/vehicle`
  - `POST /api/detect/movement`
  - `POST /api/detect/emotion`
- Backend returns annotated frame images (`data:image/jpeg;base64,...`) and metadata.

This avoids dependency on server-side webcam (`cv2.VideoCapture(0)`) for deployed apps.

## Evaluation Pipeline

Run offline model evaluation on annotated images:

```bash
python evaluate.py --mode object --annotations dataset/annotations.json --images-dir dataset/images --profile balanced --iou-threshold 0.5 --output-dir results
```

Expected annotation format (`dataset/annotations.json`):

```json
{
  "images": [
    {
      "file": "image1.jpg",
      "annotations": [
        {"label": "person", "bbox": [100, 120, 80, 160]}
      ]
    }
  ]
}
```

Outputs:
- `results/metrics_<mode>.json`
- `results/confusion_<mode>.csv`

## Deploy (Free)

### Option A: Render (Free Tier)

1. Push this project to GitHub.
2. In Render, create a new **Blueprint** and select this repository.
3. Render will read `render.yaml` and deploy automatically.

Notes:
- The build step auto-downloads `python_Scripts/yolov3.weights` if it is missing.
- Start command uses `gunicorn` with `PORT` binding.

### Option B: Docker Platforms (Hugging Face Spaces / Railway / Fly.io)

1. Use the included `Dockerfile`.
2. Build and run:
```bash
docker build -t realtime-video-analysis .
docker run -p 7860:7860 realtime-video-analysis
```
3. Open `http://localhost:7860`

Notes:
- Docker image installs required system libs for OpenCV.
- Docker build also auto-downloads YOLO weights if needed.

## Project Structure

```
realtime/
├── app.py                 # Main Flask application
├── python_Scripts/        # Core analysis scripts
│   ├── human_yolov3.py   # Human detection module
│   ├── object_yolov3.py  # Object detection module
│   └── traffic_yolov3.py # Vehicle detection module
├── static/               # Static files (CSS, JS, images)
└── templates/            # HTML templates
```

## Technologies Used

- Flask: Web framework
- OpenCV: Computer vision tasks
- TensorFlow: Deep learning framework
- YOLOv3: Object detection
- FER: Facial emotion recognition
- NumPy: Numerical computations

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- YOLOv3 for object detection
- FER for emotion recognition
- OpenCV community
