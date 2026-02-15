import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("python_Scripts/yolov3.weights", "python_Scripts/yolov3.cfg")  # Path to YOLO weights and config file
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class labels (coco.names should be in the same folder)
with open("python_Scripts/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Open the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

# Check if the webcam was opened correctly
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Analyze the detections
    vehicle_count = 0
    height, width, channels = frame.shape
    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and (class_id in (2,3,5,7)):  # 0 corresponds to 'car','motorbike','bus','truck' classes in COCO dataset
                # Get the bounding box
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maxima suppression to eliminate redundant overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw rectangles around detected humans
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            vehicle_count += 1

    # Display the human count
    cv2.putText(frame, f'Vehicles detected: {vehicle_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('YOLO Vehicle Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
