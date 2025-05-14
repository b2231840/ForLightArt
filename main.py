
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
        exit(0)

##responce
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# モデルとトークナイザの読み込み
tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-1b")
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-1b")

def get_response(text):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=100, #生成トークンの最大長
            do_sample=True, #ランダム性を持つかどうか
            top_p=0.95, #より自然に
            temperature=0.8, #ランダム性の度合い
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    reply = tokenizer.decode(output[0], skip_special_tokens=True)
    # ユーザーの入力ごと含まれるので削除
    return reply.replace(text, "").strip()


##voice_output
engine = pyttsx3.init()

## 日本語音声があったらそれを選択
for voice in engine.getProperty('voices'):
	if "Japanese" in voice.name or "Haruka" in voice.name:
		engine.setProperty('voice',voice.id)
		break

##engine.say("こんにちは、私はオフラインのAIです。")
engine.say(get_response(text)) ##responseの関数で返答
engine.runAndWait()


