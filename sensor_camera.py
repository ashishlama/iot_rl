import cv2
import os

# Use laptop's default camera (index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(" Camera not accessible.")
    exit()

ret, frame = cap.read()

if ret:
    filename = 'sample_data/sample_image.jpg'
    cv2.imwrite(filename, frame)
    print(f" Image saved to: {filename}")
else:
    print(" Failed to capture image.")

cap.release()
