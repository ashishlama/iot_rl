import cv2
import pyttsx3
import datetime
import time

engine = pyttsx3.init()

def announce(msg):
    print("yes", msg)
    engine.say(msg)
    engine.runAndWait()

def detect_motion_and_crowd(restricted_hours=(20, 8)):
    cap = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    first_frame = None

    print("Starting camera... Press Ctrl+C to stop")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)

        # Motion detection
        if first_frame is None:
            first_frame = blur
            continue

        delta = cv2.absdiff(first_frame, blur)
        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Motion alert
        now = datetime.datetime.now().hour
        for contour in contours:
            if cv2.contourArea(contour) < 2000:
                continue
            if now >= restricted_hours[0] or now < restricted_hours[1]:
                announce("Unauthorized movement detected! Security alert triggered.")
                break

        # Crowd detection
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) >= 5:
            announce("High crowd density detected. Maintain social distancing.")

        # Optional: show live feed
        # cv2.imshow("Live", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        time.sleep(3)  # To prevent alert spamming

    cap.release()
    cv2.destroyAllWindows()

detect_motion_and_crowd()

 