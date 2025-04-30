import speech_recognition as sr

r = sr.Recognizer()
with sr.Microphone() as source:
    print("話しかけてください...")
    audio = r.listen(source)

    try:
        text = r.recognize_google(audio, language="ja-JP")
        print("認識結果:", text)
    except sr.UnknownValueError:
        print("認識できませんでした")


def get_response(text):
    if "こんにちは" in text:
        return "こんにちは！今日はどうですか？"
    elif "さようなら" in text:
        return "またね〜！"
    else:
        return "ごめん、よくわからないや"


