import cv2
import numpy as np
from picamera2 import Picamera2
import time
import os

# Create videos folder if not exists
output_folder = "videos"
os.makedirs(output_folder, exist_ok=True)

# Filename with timestamp
timestamp = time.strftime("%Y%m%d-%H%M%S")
video_path = os.path.join(output_folder, f"video_{timestamp}.mp4")

# Initialize Picamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
time.sleep(2)  # Let camera warm up

# OpenCV VideoWriter for MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30
out = cv2.VideoWriter(video_path, fourcc, fps, (1020, 500))

frame_count = 0

print("Recording started... Press ESC to stop.")

while True:
    # Capture frame from Pi Camera
    frame = picam2.capture_array()
    frame = cv2.flip(frame, -1)  # Flip vertically and horizontally if needed

#    frame_count += 1
#    if frame_count % 3 != 0:  # Skip frames if needed for performance
#        continue

    # Resize for display and saving
    frame_resized = cv2.resize(frame, (1020, 500))

    # Write to video file
    out.write(frame_resized)

    # Show live frame
    cv2.imshow("FRAME", frame_resized)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
picam2.stop()
picam2.close()
out.release()
cv2.destroyAllWindows()

print(f"Video saved at: {video_path}")

