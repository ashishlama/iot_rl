import cv2
import pyttsx3
import datetime
import time
from ultralytics import YOLO

engine = pyttsx3.init()

def announce(message):
    print("yes", message)
    engine.say(message)
    engine.runAndWait()

# 1. Live Detection: Low Light + Crowd + Motion
def detect_live_events():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    first_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)

        # --- Low Light Detection ---
        brightness = gray.mean()
        if brightness < 50:
            announce("Low visibility detected. Turning on night mode.")

        # --- Motion Detection ---
        if first_frame is None:
            first_frame = blur
            continue

        delta = cv2.absdiff(first_frame, blur)
        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        now = datetime.datetime.now()
        current_time_minutes = now.hour * 60 + now.minute
        night_start = 20 * 60     # 8:00 PM = 1200
        night_end = 6 * 60 + 30   # 6:30 AM = 390

        if contours:
            for contour in contours:
                if cv2.contourArea(contour) > 2000:
                    # if current_time_minutes >= night_start or current_time_minutes < night_end:
                    if True:
                        announce("Unauthorized movement detected! Security alert triggered.")
                        print("Motion Alert at", now.strftime("%Y-%m-%d %H:%M:%S"))
                        time.sleep(5)
                        break

        # --- Crowd Detection ---
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        print(f"Faces detected: {len(faces)}")
        if len(faces) >= 2:
            announce("High crowd density detected. Maintain social distancing.")

        time.sleep(5)  # Delay between detection cycles

    cap.release()
    cv2.destroyAllWindows()


# 2. Static Detection: Face Mask / Animal / Suspicious Object
def detect_from_image(image_path):
    model = YOLO('yolov8n.pt')
    results = model(image_path)

    labels = results[0].names
    detected_classes = results[0].boxes.cls.cpu().numpy()

    for cls_id in detected_classes:
        label = labels[int(cls_id)]

        # Face Mask Detection (simulate via label)
        if label == "no-mask":
            announce("Face mask not detected! Please wear a mask.")

        # Animal Intrusion
        if label in ['dog', 'cat', 'bird']:
            announce("Animal intrusion detected! Stay alert.")

        # Suspicious Object Detection
        if label in ['backpack', 'handbag', 'suitcase']:
            announce("Unattended object detected. Please inspect.")

        # Person (for completeness)
        if label == 'person':
            print("Person detected")


# MAIN
if __name__ == "__main__":
    # Live detection (motion + brightness + crowd)
    detect_live_events()

    # Static image detection (mask, animal, object)
    # detect_from_image("sample_data/test_image.jpg")

 