import cv2
from transformers import pipeline
from PIL import Image
import numpy as np

# Load the deep fake detection pipeline
pipe = pipeline("image-classification", model="Wvolf/ViT_Deepfake_Detection")

# Load the pre-trained face detector model (Caffe model)
face_detector = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",  # Path to prototxt file
    "res10_300x300_ssd_iter_140000.caffemodel"  # Path to caffemodel file
)

# Initialize list to store trackers and bounding boxes
trackers = []
faces_boxes = []

# Function to detect faces and add to tracker
def detect_and_track_faces(frame):
    global trackers, faces_boxes
    trackers = []
    faces_boxes = []
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            face_box = (x, y, x1-x, y1-y)
            # Initialize tracker for the face
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, face_box)
            trackers.append(tracker)
            faces_boxes.append(face_box)

# Function to classify tracked faces
def classify_tracked_faces(frame):
    results = []
    for i, tracker in enumerate(trackers):
        # Update tracker and get new bounding box
        success, box = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            face = frame[y:y+h, x:x+w]
            rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_face)
            result = pipe(pil_image)
            results.append((box, result[0]))
    return results

# Process video in real-time
def process_video(video_path):
    global trackers
    video = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # If tracking is successful, classify tracked faces
        if trackers:
            results = classify_tracked_faces(frame)
            for box, result in results:
                (x, y, w, h) = [int(v) for v in box]
                label = result['label']
                score = result['score']
                color = (0, 255, 0) if label == 'Real' else (0, 0, 255)
                text = f"{label}: {score:.2f}"

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Detect faces and initialize trackers if it's the first frame or trackers are lost
        if frame_count % 30 == 0 or not trackers:
            detect_and_track_faces(frame)

        # Display the video frame with detections
        cv2.imshow('Video Stream', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    # Release the video capture object and close display window
    video.release()
    cv2.destroyAllWindows()

# Path to your video file
video_path = 'download.mp4'

# Process video in real-time
process_video(video_path)