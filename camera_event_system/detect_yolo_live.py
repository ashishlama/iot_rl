import cv2
import pyttsx3
import time
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("runs/detect/train/weights/best.pt")  # or your own trained path
 

# Setup speech engine
engine = pyttsx3.init()
def announce(message):
    print("yes", message)
    engine.say(message)
    engine.runAndWait()

# Open camera
cap = cv2.VideoCapture(0)

# Detection loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    boxes = results[0].boxes
    classes = results[0].names
    detected = []

    for box in boxes:
        cls_id = int(box.cls[0].item())
        label = classes[cls_id]
        detected.append(label)

        # Draw box
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
        cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
                # Trigger based on label
        if label == "without_mask":
            announce("Face mask not detected! Please wear a mask.")
        elif label == "with_mask":
            print("Person with mask detected.")
        elif label in ["backpack", "handbag", "suitcase"]:
            announce("Unattended object detected. Please inspect.")
        elif label in ["dog", "cat", "bird"]:
            announce("Animal intrusion detected! Stay alert.")
        elif label == "person":
            print("Person detected")


    # Show frame
    cv2.imshow("YOLOv8 Live Detection", frame)

    # Press 'q' to quit 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
