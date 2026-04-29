from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import io

app = Flask(__name__)
CORS(app)

# =========================
# LOAD MODELS
# =========================
print("Loading Whisper model...")
whisper_model = whisper.load_model("tiny")

print("Loading T5 model...")
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
print("All models loaded! Server ready.")


# =========================
# SUMMARIZATION FUNCTION
# =========================
def summarize_text(text):
    # T5 has 512 token limit — chunk if needed
    max_chars = 1800
    if len(text) > max_chars:
        text = text[:max_chars]

    inputs = tokenizer(
        "summarize: " + text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )
    outputs = model.generate(
        inputs["input_ids"],
        max_length=150,
        min_length=30,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# =========================
# ROUTE 1: TXT / Book file upload
# =========================
@app.route("/upload-book", methods=["POST"])
def upload_book():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = file.filename.lower()

    # Read text from TXT file
    if filename.endswith(".txt"):
        try:
            text = file.read().decode("utf-8", errors="ignore").strip()
        except Exception as e:
            return jsonify({"error": f"Could not read file: {str(e)}"}), 400

        if not text:
            return jsonify({"error": "File is empty"}), 400

        summary = summarize_text(text)
        return jsonify({
            "transcription": text[:2000],   # send first 2000 chars as preview
            "summary": summary
        })

    else:
        return jsonify({"error": "Only .txt files supported on this route"}), 400


# =========================
# ROUTE 2: Audio file → Transcription + Summary
# =========================
@app.route("/summarize", methods=["POST"])
def summarize_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = "temp_audio" + os.path.splitext(file.filename)[1]
    file.save(file_path)

    try:
        result = whisper_model.transcribe(file_path)
        text = result["text"]
        summary = summarize_text(text)
    except Exception as e:
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    return jsonify({
        "transcription": text,
        "summary": summary
    })


# =========================
# ROUTE 3: Plain text / pasted text
# =========================
@app.route("/summarize-text", methods=["POST"])
def summarize_text_route():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400

    summary = summarize_text(text)
    return jsonify({
        "transcription": text,
        "summary": summary
    })


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(debug=True, port=5000)