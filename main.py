import pyttsx3

engine = pyttsx3.init()

## 日本語音声があったらそれを選択
for voice in engin.getProperty('voices'):
	if "Japanese" in voice.name or "Haruka" in voice.name:
		engine.setProperty('voice',voice.id)
		break

engine.say("こんにちは、私はオフラインのAIです。")
engine.runAndWait()
