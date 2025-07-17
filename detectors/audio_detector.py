import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sounddevice as sd

# Load YAMNet model from TensorFlow Hub
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

# Load YAMNet class names
class_map_path = tf.keras.utils.get_file(
    'yamnet_class_map.csv',
    'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
)
class_names = [line.strip().split(',')[2] for line in open(class_map_path).readlines()[1:]]

# Define keywords and messages for your use case
event_keywords = {
    "smoke alarm": "Fire alarm detected! Evacuate immediately.",
    "glass": "Possible intrusion detected! Alerting security.",
    "baby cry": "Baby crying detected. Notifying guardian.",
    "doorbell": "Doorbell detected. Please check the entrance.",
    "gunshot": "Gunshot detected! Take cover and call emergency services."
}

# Record audio for a few seconds
def record_audio(duration=2, sample_rate=16000):
    print("🎙️ Recording audio...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(audio)

# Detect sound event using YAMNet
def detect_audio_event():
    waveform = record_audio()
    waveform = waveform / np.max(np.abs(waveform))  # Normalize

    scores, embeddings, spectrogram = yamnet_model(waveform)
    prediction = np.mean(scores, axis=0)

    top_index = np.argmax(prediction)
    top_class = class_names[top_index]
    confidence = prediction[top_index]

    print(f"✅ Detected: {top_class} ({confidence:.2f})")

    for keyword, message in event_keywords.items():
        if keyword in top_class.lower() and confidence > 0.4:
            return keyword  # Return internal keyword (e.g., 'glass')

    return None
