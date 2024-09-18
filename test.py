import cv2
from transformers import pipeline
from PIL import Image

# Load the deep fake detection pipeline
pipe = pipeline("image-classification", model="Wvolf/ViT_Deepfake_Detection")

# Function to extract frames from a video
def extract_frames(video_path, interval=30):
    video = cv2.VideoCapture(video_path)
    frames = []
    success, frame = video.read()
    count = 0

    while success:
        if count % interval == 0:
            frames.append(frame)
        success, frame = video.read()
        count += 1

    video.release()
    return frames

# Function to classify frames
def classify_frames(frames):
    results = []
    for frame in frames:
        # Convert frame to RGB and then to a PIL image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        # Classify frame
        result = pipe(pil_image)
        results.append(result)
    return results

# Path to your video file
video_path = 'download.mp4'

# Extract frames
frames = extract_frames(video_path)

# Classify frames
results = classify_frames(frames)

# Output the results
for i, result in enumerate(results):
    print(f"Frame {i}: {result}")
