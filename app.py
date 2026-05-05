from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
import io

# PDF and DOCX extractors
import pdfplumber
import docx

app = Flask(__name__)
CORS(app)

# =========================
# LOAD MODELS (LOW MEMORY FIX)
# =========================
print("Loading T5 model...")

tokenizer = AutoTokenizer.from_pretrained("t5-small", legacy=True)

model = AutoModelForSeq2SeqLM.from_pretrained(
    "t5-small",
    low_cpu_mem_usage=True,
    torch_dtype=torch.float32
)

model.eval()

print("T5 loaded! Server ready.")

# =========================
# TEXT EXTRACTION
# =========================
def extract_txt(file):
    return file.read().decode("utf-8", errors="ignore").strip()

def extract_pdf(file):
    text = ""
    with pdfplumber.open(io.BytesIO(file.read())) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "")
    return text.strip()

def extract_docx(file):
    doc = docx.Document(io.BytesIO(file.read()))
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()]).strip()

def extract_text(file, filename):
    name = filename.lower()

    if name.endswith(".txt"):
        return extract_txt(file)

    elif name.endswith(".pdf"):
        return extract_pdf(file)

    elif name.endswith(".docx"):
        return extract_docx(file)

    else:
        raise ValueError("Unsupported file type. Use .txt, .pdf, .docx")

# =========================
# SUMMARIZATION FUNCTION
# =========================
def summarize_text(text):

    CHUNK_SIZE = 2000
    MAX_CHUNKS = 40
    MERGE_CAP = 1800

    def run_model(chunk):
        inputs = tokenizer(
            "summarize: " + chunk,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=100,
                min_length=20,
                num_beams=2,
                length_penalty=1.5,
                early_stopping=True
            )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    text = text.strip()

    # Short text
    if len(text) <= CHUNK_SIZE:
        return run_model(text)

    # Chunking
    chunks = []
    start = 0
    while start < len(text) and len(chunks) < MAX_CHUNKS:
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE

    summaries = [run_model(c) for c in chunks]

    combined = " ".join(summaries)

    if len(combined) > MERGE_CAP:
        combined = combined[:MERGE_CAP]

    return run_model(combined)

# =========================
# ROUTE: FILE UPLOAD
# =========================
@app.route("/upload-book", methods=["POST"])
def upload_book():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = file.filename or ""

    try:
        text = extract_text(file, filename)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    if not text:
        return jsonify({"error": "Empty file"}), 400

    summary = summarize_text(text)

    return jsonify({
        "transcription": text[:2000],
        "summary": summary
    })

# =========================
# ROUTE: TEXT INPUT
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