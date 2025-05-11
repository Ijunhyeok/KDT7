from flask import Flask, request, render_template
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- 설정 ---
MODEL_DIR = "./best_model.pth" # !! 실제 최고 성능 모델 저장 디렉토리 경로로 수정하세요 !!
SOURCE_LANG = "en"
TARGET_LANG = "ko"

app = Flask(__name__)
tokenizer = None
model = None
device = "cpu" # CPU로 명시

# --- 모델 및 토크나이저 로드 (애플리케이션 시작 시) ---
def load_model():
    global tokenizer, model
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(device)
        print(f"'{MODEL_DIR}'에서 모델 및 토크나이저 로드 성공 (CPU 사용).")
    except Exception as e:
        print(f"모델 로드 실패: {e}")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/translate", methods=["POST"])
def translate():
    if not tokenizer or not model:
        return "모델 로딩에 실패했습니다.", 500

    text_to_translate = request.form["text"]
    source_lang = request.form["source_lang"]
    target_lang = request.form["target_lang"]

    try:
        model.eval()
        input_ids = tokenizer(text_to_translate, return_tensors="pt", truncation=True, padding=True).to(device)

        if tokenizer.name_or_path.startswith("google/mt5"):
            source_lang_token = tokenizer.lang_code_to_id[source_lang]
            input_ids['input_ids'] = torch.cat([
                torch.tensor([[source_lang_token]]).to(device), input_ids['input_ids']
            ], dim=-1)

        with torch.no_grad():
            outputs = model.generate(
                **input_ids,
                max_length=128,
                num_beams=5,
                early_stopping=True,
                return_dict_in_generate=True
            )

        translated_tokens = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        translated_text = translated_tokens[0] if translated_tokens else "번역 실패"
        return render_template("index.html", original_text=text_to_translate, translated_text=translated_text, source_lang=source_lang, target_lang=target_lang)

    except Exception as e:
        print(f"번역 오류: {e}")
        return "번역 중 오류가 발생했습니다.", 500

if __name__ == "__main__":
    load_model()
    app.run(debug=True)

# from flask import Flask, render_template

# app = Flask(__name__)

# @app.route("/", methods=["GET"])
# def index():
#     return render_template("index.html")

# @app.route("/translate", methods=["POST"])
# def translate():
#     # 모델 로딩 및 번역 기능은 아직 구현되지 않았으므로
#     # 임시 메시지를 표시하거나 이전 페이지로 리다이렉트합니다.
#     return render_template("index.html", error="번역 기능은 준비 중입니다.")

# if __name__ == "__main__":
#     app.run(debug=True)