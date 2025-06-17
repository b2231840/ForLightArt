from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# モデルとトークナイザの読み込み
tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-medium")
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")

def get_response(text):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=300, #生成トークンの最大長
            do_sample=True, #ランダム性を持つかどうか
            top_p=0.95, #より自然に
            temperature=0.8, #ランダム性の度合い
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    reply = tokenizer.decode(output[0], skip_special_tokens=True)
    # ユーザーの入力ごと含まれるので削除
    return reply.replace(text, "").strip()

def generate_and_responce(model,toknizer,input_ids,max_length):
    responces = ""
    for _ in range(max_length):
        with torch.no_grad():
            #出力用モデルを入手
            output = model(input_ids.to(model.device))

        #予測確率(logits)を入手
        logits = output.logits

        #Top-Kサンプリングの適用
        indices_to_remove = logits <  torch.topk(logits, 50)[0][...,-1,None]
        logits[indices_to_remove] = float('-inf')

        #次のトークンをサンプリング
        probs = torch.nn.functional.softmax(logits[...,-1,:], dim = -1)
        next_token_id = torch.multinomial(probs,num_samples=1)

        #次のトークンを追加し、文字に変換
        input_ids = torch.cat((input_ids,next_token_id),dim=-1)
        output_str = tokenizer.decode(next_token_id[0])
        responces += output_str

        #結果を表示
        print(output_str.replace("<NL>","\n"), end='', flush = True)

        #終了
        if "</s>" in output_str:
            break

    return responces
