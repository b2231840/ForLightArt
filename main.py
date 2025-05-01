import speech_recognition as sr
import pyttsx3

##voice_input
r = sr.Recognizer()
with sr.Microphone() as source:
    print("話しかけてください...")
    audio = r.listen(source)

    try:
        text = r.recognize_google(audio, language="ja-JP")
        print("認識結果:", text)
    except sr.UnknownValueError:
        print("認識できませんでした")

##responce
def get_response(text):
    if "こんにちは" in text:
        return "こんにちは！今日はどうですか？"
    elif "さようなら" in text:
        return "またね〜！"
    else:
        return "ごめん、よくわからないや"


##voice_output
engine = pyttsx3.init()

## 日本語音声があったらそれを選択
for voice in engine.getProperty('voices'):
	if "Japanese" in voice.name or "Haruka" in voice.name:
		engine.setProperty('voice',voice.id)
		break

engine.say("こんにちは、私はオフラインのAIです。")
engine.runAndWait()
