import time
from detectors.audio_detector import detect_audio_event
from utils.announcer import announce

event_messages = {
    "firealarm": "Fire alarm detected! Evacuate immediately.",
    "breaking_glass": "Possible intrusion detected! Alerting security.",
    "baby_crying": "Baby crying detected. Notifying guardian.",
    "doorbell": "Doorbell detected. Please check the entrance.",
    "gun_shot": "Gunshot detected! Take cover and call emergency services."
}

if __name__ == "__main__":
    while True:
        event = detect_audio_event()
        if event in event_messages:
            announce(event_messages[event])
        time.sleep(1)
