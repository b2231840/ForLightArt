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
        text = "祝福に感謝します(^^)"
        #exit(0)


##responce
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# モデルとトークナイザの読み込み
tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-medium")
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")

#使わないかも↓
def get_response(text):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=80, #生成トークンの最大長
            do_sample=True, #ランダム性を持つかどうか
            top_p=0.9, #より自然に
            temperature=0.8, #ランダム性の度合い
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    reply = tokenizer.decode(output[0], skip_special_tokens=True)
    # ユーザーの入力ごと含まれるので削除
    return reply.replace(text, "").strip()

#こっち使うかも↓
def generateAndResponce(model, tokenizer, input_ids, max_length, engine):
    responses = ""
    buffer = ""

    for _ in range(max_length):
        with torch.no_grad():
            output = model(input_ids.to(model.device))
        logits = output.logits

        indices_to_remove = logits < torch.topk(logits, 50)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')

        probs = torch.nn.functional.softmax(logits[..., -1, :], dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat((input_ids, next_token_id), dim=-1)
        output_str = tokenizer.decode(next_token_id[0])
        buffer += output_str
        responses += output_str

        print(output_str.replace("<NL>", "\n"), end='', flush=True)

        if "。" in output_str:
            engine.say(buffer)
            engine.runAndWait()
            buffer = ""

        if "</s>" in output_str:
            break

    # 最後に残ったバッファも読み上げ
    if buffer:
        engine.say(buffer)
        engine.runAndWait()

    return responses



##voice_output
engine = pyttsx3.init()

## 日本語音声があったらそれを選択
for voice in engine.getProperty('voices'):
        if "Japanese" in voice.name or "Haruka" in voice.name:
                engine.setProperty('voice',voice.id)
                break

pronpt = f"""ユーザーが以下のようにあなたに話しかけています。あなたはユーザーに対して長すぎず、自然に、簡潔に、やさしく、敬語で返答します。\n\nユーザーの質問:{text}"""
input_ids = tokenizer.encode(pronpt, return_tensors="pt")
print(pronpt,"\n")

##engine.say("こんにちは、私はオフラインのAIです。")
res = generateAndResponce(model,tokenizer,input_ids,300,engine)

##engine.say(res) ##responseの関数で返答
print("\n\n返答結果：",res)
engine.runAndWait()