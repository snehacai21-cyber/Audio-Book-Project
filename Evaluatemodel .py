"""
=============================================================================
Audio-Based Book Summarization System — Standalone Evaluation Script
=============================================================================
Run AFTER training to get a detailed ROUGE report and qualitative samples.

Usage:
    python evaluate_model.py --model_path saved_model \
                             --data_path data/booksummarization.csv \
                             --num_samples 200
=============================================================================
"""

import json
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import evaluate as hf_evaluate
from transformers import BartTokenizer, BartForConditionalGeneration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────

def load_eval_data(data_path: str, num_samples: int, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(
        data_path, sep="\t", header=None,
        names=["id","book_id","title","author","date","genres","summary"],
        on_bad_lines="skip",
    )
    df = df.dropna(subset=["summary"])
    df = df[df["summary"].str.strip().str.len() > 50].reset_index(drop=True)
    df["summary"] = df["summary"].str.strip()
    df["title"] = df["title"].fillna("Unknown").str.strip()
    # Use last 10% as eval split (same split as train.py)
    n_train = int(len(df) * 0.9)
    val_df = df[n_train:].reset_index(drop=True)
    return val_df.sample(min(num_samples, len(val_df)), random_state=seed).reset_index(drop=True)


def generate_summary(
    model, tokenizer, text: str, title: str,
    device: str, max_input: int = 512, max_gen: int = 128,
    num_beams: int = 4, length_penalty: float = 2.0,
    no_repeat_ngram_size: int = 3,
) -> str:
    prompt = f"summarize: {title}: {text}"
    inputs = tokenizer(
        prompt, max_length=max_input, truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            min_length=30,
            max_length=max_gen,
            early_stopping=True,
        )
    return tokenizer.decode(ids[0], skip_special_tokens=True)


# ─────────────────────────────────────────────
# main evaluation
# ─────────────────────────────────────────────

def evaluate_model(
    model_path: str,
    data_path: str,
    num_samples: int = 200,
    output_file: str = "evaluation_report.json",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Load model
    tokenizer = BartTokenizer.from_pretrained(model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path).to(device)
    model.eval()
    logger.info(f"Model loaded from: {model_path}")

    # Load eval split
    val_df = load_eval_data(data_path, num_samples)
    logger.info(f"Evaluating on {len(val_df)} samples…")

    rouge = hf_evaluate.load("rouge")

    predictions, references = [], []
    qualitative_samples = []

    for i, row in val_df.iterrows():
        if i % 20 == 0:
            logger.info(f"  [{i+1}/{len(val_df)}]")

        # Reference: first ~80 words (same proxy used in training)
        reference = " ".join(str(row["summary"]).split()[:80])
        # Prediction
        prediction = generate_summary(
            model, tokenizer,
            text=str(row["summary"]),
            title=str(row["title"]),
            device=device,
        )

        predictions.append(prediction)
        references.append(reference)

        # Save first 5 for qualitative review
        if len(qualitative_samples) < 5:
            qualitative_samples.append({
                "title": row["title"],
                "author": row.get("author", ""),
                "source_excerpt": str(row["summary"])[:300] + "…",
                "reference_summary": reference,
                "model_summary": prediction,
            })

    # Compute ROUGE
    result = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )
    rouge_scores = {k: round(v * 100, 4) for k, v in result.items()}

    # ── Print Report ──────────────────────────────────────────────
    border = "=" * 60
    print(f"\n{border}")
    print("  EVALUATION REPORT — Book Summarization (BART)")
    print(f"  Model   : {model_path}")
    print(f"  Samples : {len(val_df)}")
    print(border)
    print(f"  {'Metric':<12} {'Score':>8}")
    print(f"  {'-'*20}")
    for metric in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
        if metric in rouge_scores:
            print(f"  {metric:<12} {rouge_scores[metric]:>8.4f}")
    print(border)

    print("\n  QUALITATIVE SAMPLES")
    print(f"  {'-'*56}")
    for s in qualitative_samples:
        print(f"\n  📖  {s['title']}  ({s['author']})")
        print(f"  Source  : {s['source_excerpt'][:120]}…")
        print(f"  Reference: {s['reference_summary'][:120]}")
        print(f"  Model   : {s['model_summary']}")

    # ── Save JSON ─────────────────────────────────────────────────
    report = {
        "model_path": model_path,
        "num_eval_samples": len(val_df),
        "rouge_scores": rouge_scores,
        "qualitative_samples": qualitative_samples,
    }
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nFull report saved to: {output_file}")

    return rouge_scores


# ─────────────────────────────────────────────
# entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="saved_model")
    parser.add_argument("--data_path",  type=str, default="data/booksummarization.csv")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--output_file", type=str, default="evaluation_report.json")
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        data_path=args.data_path,
        num_samples=args.num_samples,
        output_file=args.output_file,
    )