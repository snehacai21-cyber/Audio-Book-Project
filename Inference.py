"""
=============================================================================
Audio-Based Book Summarization System — Inference Script
=============================================================================
Supports:
  • Single text summarization
  • Long-document chunked summarization
  • PDF text extraction + summarization
  • Optional Text-to-Speech (gTTS or pyttsx3)
=============================================================================
"""

import os
import re
import logging
import warnings
from pathlib import Path
from typing import List, Optional

import torch
from transformers import BartTokenizer, BartForConditionalGeneration

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. SUMMARIZER CLASS
# ─────────────────────────────────────────────

class BookSummarizer:
    """
    BART-based abstractive book summarizer.

    Usage:
        summarizer = BookSummarizer("saved_model")
        summary = summarizer.summarize("Long book text here…")
    """

    def __init__(
        self,
        model_path: str = "saved_model",
        device: Optional[str] = None,
        # Generation hyper-parameters (mirror the paper's beam search settings)
        num_beams: int = 4,
        length_penalty: float = 2.0,
        no_repeat_ngram_size: int = 3,
        min_length: int = 30,
        max_length: int = 128,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.min_length = min_length
        self.max_length = max_length

        logger.info(f"Loading model from: {model_path}  |  device: {self.device}")
        self.tokenizer = BartTokenizer.from_pretrained(model_path)
        self.model = BartForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model ready.")

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _chunk_tokens(self, text: str, max_tokens: int = 512, overlap: int = 50) -> List[str]:
        """Split long text into overlapping token windows."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            return [text]
        stride = max_tokens - overlap
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunks.append(
                self.tokenizer.decode(tokens[start:end], skip_special_tokens=True)
            )
            if end == len(tokens):
                break
            start += stride
        return chunks

    def _summarize_single_chunk(self, text: str, title: str = "") -> str:
        """Summarize one chunk of text (≤ 512 tokens)."""
        prompt = f"summarize: {title + ': ' if title else ''}{text}"
        inputs = self.tokenizer(
            prompt,
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                num_beams=self.num_beams,
                length_penalty=self.length_penalty,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                min_length=self.min_length,
                max_length=self.max_length,
                early_stopping=True,
            )

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # ── Public API ───────────────────────────────────────────────

    def summarize(
        self,
        text: str,
        title: str = "",
        chunk_size: int = 450,      # slightly below 512 for safety
        overlap: int = 50,
        merge_strategy: str = "concat",   # "concat" | "recursive"
    ) -> str:
        """
        Summarize an arbitrarily long text.

        Args:
            text           : Raw book/document text.
            title          : Optional book title (prepended to prompt).
            chunk_size     : Max tokens per chunk (≤ 512 for BART).
            overlap        : Overlapping tokens between consecutive chunks.
            merge_strategy : How to combine chunk summaries.
                             "concat"    → join chunk summaries with ". "
                             "recursive" → summarize the joined summaries again

        Returns:
            Final summary string.
        """
        text = self.clean_text(text)
        chunks = self._chunk_tokens(text, max_tokens=chunk_size, overlap=overlap)
        logger.info(f"Processing {len(chunks)} chunk(s)…")

        chunk_summaries = [
            self._summarize_single_chunk(chunk, title=title if i == 0 else "")
            for i, chunk in enumerate(chunks)
        ]

        if len(chunk_summaries) == 1:
            return chunk_summaries[0]

        combined = " ".join(chunk_summaries)

        if merge_strategy == "recursive":
            logger.info("Recursively summarizing chunk summaries…")
            return self._summarize_single_chunk(combined, title=title)
        else:
            return combined

    def summarize_file(self, file_path: str, title: str = "") -> str:
        """Read a plain-text file and summarize it."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        text = path.read_text(encoding="utf-8", errors="ignore")
        return self.summarize(text, title=title or path.stem)

    def summarize_pdf(self, pdf_path: str, title: str = "") -> str:
        """
        Extract text from a PDF and summarize it.
        Requires: pip install pdfplumber
        """
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("Install pdfplumber: pip install pdfplumber")

        logger.info(f"Extracting text from PDF: {pdf_path}")
        text_pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_pages.append(page_text)
        full_text = "\n".join(text_pages)
        logger.info(f"Extracted {len(full_text.split())} words from PDF.")
        return self.summarize(full_text, title=title)


# ─────────────────────────────────────────────
# 2. TEXT-TO-SPEECH (OPTIONAL)
# ─────────────────────────────────────────────

class SummaryToSpeech:
    """
    Convert a text summary to an audio file.

    Two backends are supported:
      • "gtts"   — Google Text-to-Speech (requires internet)
      • "pyttsx3"— Offline TTS engine

    Install:
        pip install gTTS           # online
        pip install pyttsx3        # offline
    """

    def __init__(self, backend: str = "gtts", language: str = "en"):
        self.backend = backend
        self.language = language

    def speak(self, text: str, output_path: str = "summary_audio.mp3") -> str:
        """
        Convert text to speech and save to file.

        Returns:
            Path to the saved audio file.
        """
        if self.backend == "gtts":
            return self._gtts(text, output_path)
        elif self.backend == "pyttsx3":
            return self._pyttsx3(text, output_path)
        else:
            raise ValueError(f"Unknown TTS backend: {self.backend}")

    def _gtts(self, text: str, output_path: str) -> str:
        try:
            from gtts import gTTS
        except ImportError:
            raise ImportError("Install gTTS: pip install gTTS")
        tts = gTTS(text=text, lang=self.language, slow=False)
        tts.save(output_path)
        logger.info(f"Audio saved to: {output_path}")
        return output_path

    def _pyttsx3(self, text: str, output_path: str) -> str:
        try:
            import pyttsx3
        except ImportError:
            raise ImportError("Install pyttsx3: pip install pyttsx3")
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)   # words per minute
        engine.save_to_file(text, output_path)
        engine.runAndWait()
        logger.info(f"Audio saved to: {output_path}")
        return output_path


# ─────────────────────────────────────────────
# 3. DEMO — END-TO-END PIPELINE
# ─────────────────────────────────────────────

def demo_pipeline(
    model_path: str = "saved_model",
    tts_enabled: bool = False,
    tts_backend: str = "gtts",
):
    """Full pipeline: text → summary → (optionally) audio."""

    # ── Sample book plot (Animal Farm) ──────────────────────────
    sample_text = """
    Old Major, the old boar on the Manor Farm, calls the animals on the farm for a
    meeting, where he compares the humans to parasites and teaches the animals a
    revolutionary song, 'Beasts of England'. When Major dies, two young pigs, Snowball
    and Napoleon, assume command and turn his dream into a philosophy. The animals
    revolt and drive the drunken and irresponsible Mr Jones from the farm, renaming it
    "Animal Farm". They adopt Seven Commandments of Animalism, the most important of
    which is, "All animals are equal". Snowball attempts to teach the animals reading
    and writing; food is plentiful, and the farm runs smoothly. The pigs elevate
    themselves to positions of leadership and Napoleon declares himself leader, using
    Squealer as a mouthpiece to spread propaganda. Over time, the pigs adopt the
    lifestyle of humans, rewriting the commandments to suit themselves, and the farm
    ends up being indistinguishable from a human-run estate. The animals realise that
    they cannot tell the difference between the pigs and the men who once oppressed them.
    """

    # ── Step 1: Summarize ────────────────────────────────────────
    summarizer = BookSummarizer(model_path=model_path)
    summary = summarizer.summarize(sample_text, title="Animal Farm")

    print("\n" + "=" * 60)
    print("INPUT TEXT (truncated):")
    print(sample_text[:300].strip() + "…")
    print("\nGENERATED SUMMARY:")
    print(summary)
    print("=" * 60)

    # ── Step 2: Text-to-Speech (optional) ────────────────────────
    if tts_enabled:
        tts = SummaryToSpeech(backend=tts_backend)
        audio_path = tts.speak(summary, output_path="animal_farm_summary.mp3")
        print(f"\nAudio summary saved to: {audio_path}")

    return summary


# ─────────────────────────────────────────────
# 4. ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Book Summarizer Inference")
    parser.add_argument("--model_path", type=str, default="saved_model")
    parser.add_argument("--text", type=str, default=None, help="Direct text to summarize")
    parser.add_argument("--file", type=str, default=None, help="Path to .txt file")
    parser.add_argument("--pdf",  type=str, default=None, help="Path to .pdf file")
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--tts", action="store_true", help="Enable TTS output")
    parser.add_argument("--tts_backend", type=str, default="gtts", choices=["gtts", "pyttsx3"])
    parser.add_argument("--demo", action="store_true", help="Run built-in demo")
    args = parser.parse_args()

    if args.demo or (not args.text and not args.file and not args.pdf):
        demo_pipeline(model_path=args.model_path, tts_enabled=args.tts, tts_backend=args.tts_backend)
    else:
        summarizer = BookSummarizer(model_path=args.model_path)

        if args.text:
            result = summarizer.summarize(args.text, title=args.title)
        elif args.file:
            result = summarizer.summarize_file(args.file, title=args.title)
        elif args.pdf:
            result = summarizer.summarize_pdf(args.pdf, title=args.title)

        print("\nSUMMARY:\n", result)

        if args.tts:
            tts = SummaryToSpeech(backend=args.tts_backend)
            tts.speak(result, output_path="output_summary.mp3")