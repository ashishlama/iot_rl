import cv2
import pyttsx3

engine = pyttsx3.init()

def announce(msg):
    print("audio", msg)
    engine.say(msg)
    engine.runAndWait()

def detect_camera_events(image_path='sample_data/sample_image.jpg'):
    img = cv2.imread(image_path)

    if img is None:
        print("Could not load image.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()

    # Check for low light
    if brightness < 50:
        announce("Low visibility detected. Turning on night mode.")

    # Face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        announce("Unrecognized person at the door. Proceed with caution.")
    elif len(faces) > 5:
        announce("High crowd density detected. Maintain social distancing.")
    else:
        announce(f"{len(faces)} person(s) detected.")

# Example Test:
detect_camera_events()
