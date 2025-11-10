import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import speech_recognition as sr
import pyttsx3

# ===== モデル準備 =====
MODEL_NAME = "rinna/japanese-gpt-neox-3.6b-instruction-sft"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
if torch.cuda.is_available():
    model = model.to("cuda")

# ===== 音声合成準備 =====
engine = pyttsx3.init()
for voice in engine.getProperty('voices'):
    if "Japanese" in voice.name or "Haruka" in voice.name:
        engine.setProperty('voice', voice.id)
        break
engine.setProperty('rate', 180)

# ===== 音声認識 =====
def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 話しかけてください...")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio, language="ja-JP")
        print("👂 認識結果:", text)
    except sr.UnknownValueError:
        text = ""
        print("⚠️ 認識できませんでした")
    return text

# ===== 応答生成 =====
def generate_response(user_input, history, max_new_tokens=60):
    # 会話履歴からプロンプト生成
    prompt = "以下は人間（ユーザー）とAIの会話です。AIは短く自然な日本語で答えます。\n\n"
    for turn in history[-3:]:  # 直近3ターンのみ保持
        prompt += f"ユーザー: {turn['user']}\nAI: {turn['ai']}\n"
    prompt += f"ユーザー: {user_input}\nAI:"

    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "AI:" in text:
        text = text.split("AI:")[-1].strip()

    # 文を自然に止める
    stop_puncts = ["。", "？", "！"]
    for p in stop_puncts:
        if p in text:
            text = text[: text.index(p) + 1]
            break
    return text

# ===== 音声出力 =====
def speak(text):
    print("💬 AI:", text)
    engine.say(text)
    engine.runAndWait()

# ===== メインループ =====
def main():
    history = []
    while True:
        user_input = listen()
        if not user_input:
            continue
        if user_input in ["終了", "終わり", "さようなら","またね"]:
            speak("はい、またお話ししましょう。")
            break

        response = generate_response(user_input, history)
        speak(response)
        history.append({"user": user_input, "ai": response})

if __name__ == "__main__":
    main()
