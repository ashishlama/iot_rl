import pyttsx3

def announce(message):
    print("🔊", message)
    engine = pyttsx3.init()
    engine.say(message)
    engine.runAndWait()
