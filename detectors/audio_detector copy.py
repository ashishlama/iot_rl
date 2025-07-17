import sounddevice as sd
import numpy as np
import librosa
import os

SAMPLE_RATE = 16000
DURATION = 5  # seconds
THRESHOLD = 0.45  # similarity threshold

# Preload reference audio features (MFCCs)
def load_reference_mfccs():
    folder = "sample"
    events = ["firealarm", "breaking_glass", "baby_crying", "doorbell", "gun_shot"]
    ref_mfccs = {}

    for event in events:
        path = os.path.join(folder, f"{event}.wav")
        if os.path.exists(path):
            y, _ = librosa.load(path, sr=SAMPLE_RATE)
            mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE)
            ref_mfccs[event] = mfcc
        else:
            print(f"❌ Missing sample: {event}.wav")
    
    return ref_mfccs

reference_mfccs = load_reference_mfccs()

def cosine_similarity(mfcc1, mfcc2):
    min_rows = min(mfcc1.shape[0], mfcc2.shape[0])
    min_cols = min(mfcc1.shape[1], mfcc2.shape[1])
    mfcc1 = mfcc1[:min_rows, :min_cols]
    mfcc2 = mfcc2[:min_rows, :min_cols]

    # Flatten both MFCCs to vectors
    v1 = mfcc1.flatten()
    v2 = mfcc2.flatten()

    # Compute cosine similarity
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)

def detect_audio_event():
    print("🎙️ Listening for event...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    y = audio.flatten()

    try:
        mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=13)
        print("MFCC shape:", mfcc.shape)
    except Exception as e:
        print("MFCC extraction failed:", e)
        return None

    best_match = None
    best_score = 0

    for label, ref_mfcc in reference_mfccs.items():
        score = cosine_similarity(mfcc, ref_mfcc)
        print(score)
        if score > best_score:
            best_score = score
            best_match = label

    if best_score > THRESHOLD:
        print(f"✅ Detected: {best_match} (Score: {best_score:.2f})")
        return best_match
    else:
        print("🔇 No match")
        return None
