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
