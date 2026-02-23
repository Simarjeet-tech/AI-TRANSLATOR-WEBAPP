from flask import Flask, request, render_template
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)

models = {
    "en-hi": "Helsinki-NLP/opus-mt-en-hi",
    "hi-en": "Helsinki-NLP/opus-mt-hi-en",
    "en-fr": "Helsinki-NLP/opus-mt-en-fr",
    "en-ur": "Helsinki-NLP/opus-mt-en-ur"
}

loaded = {}

def translate(text, lang):
    if lang not in loaded:
        tokenizer = MarianTokenizer.from_pretrained(models[lang])
        model = MarianMTModel.from_pretrained(models[lang])
        loaded[lang] = (tokenizer, model)

    tokenizer, model = loaded[lang]
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)

    return tokenizer.decode(translated[0], skip_special_tokens=True)

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    text = ""

    if request.method == "POST":
        text = request.form["text"]
        lang = request.form["lang"]
        result = translate(text, lang)

    return render_template("index.html", result=result, text=text)

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)