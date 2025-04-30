def get_response(text):
    if "こんにちは" in text:
        return "こんにちは！今日はどうですか？"
    elif "さようなら" in text:
        return "またね〜！"
    else:
        return "ごめん、よくわからないや"

