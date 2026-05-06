#!/usr/bin/env bash
# =============================================================================
# setup.sh — Prepare the environment for Book Summarization Training
# =============================================================================
set -e

echo "============================================================"
echo " Audio-Based Book Summarization System — Environment Setup"
echo "============================================================"

# 1. Create directory structure
mkdir -p data outputs/bart_book_summarizer saved_model logs

# 2. Copy dataset
if [ -f "/mnt/user-data/uploads/booksummarization.csv" ]; then
    cp /mnt/user-data/uploads/booksummarization.csv data/
    echo "[✓] Dataset copied to data/"
else
    echo "[!] Place booksummarization.csv in the data/ directory"
fi

# 3. Install Python dependencies
echo ""
echo "Installing dependencies…"
pip install -q --break-system-packages \
    torch transformers datasets evaluate \
    sentencepiece sacrebleu rouge_score \
    accelerate pandas numpy tqdm pdfplumber

echo ""
echo "[✓] Setup complete!"
echo ""
echo "To train the model:"
echo "    python train.py"
echo ""
echo "To run inference:"
echo "    python inference.py --demo"
echo "    python inference.py --text 'Your book plot here…'"
echo "    python inference.py --file my_book.txt --tts"
echo ""
echo "To evaluate:"
echo "    python evaluate_model.py --model_path saved_model"