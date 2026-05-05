# =========================
# INSTALL (run in terminal once)
# pip install transformers datasets evaluate rouge_score matplotlib torch
# =========================

from transformers import pipeline
from datasets import Dataset
import evaluate
import matplotlib.pyplot as plt

# =========================
# SMALL CUSTOM DATASET (NO INTERNET HEAVY LOAD)
# =========================
data = {
    "article": [
        "Artificial Intelligence is transforming industries by automating tasks and improving efficiency across multiple sectors.",
        "Machine learning allows systems to learn from data and continuously improve performance without explicit programming.",
        "Deep learning is an advanced form of machine learning that uses neural networks for complex pattern recognition.",
        "Natural language processing helps computers understand, interpret, and generate human language effectively.",
        "Artificial intelligence is widely used in healthcare, finance, education, and many other industries."
    ],
    "summary": [
        "Artificial intelligence improves efficiency across industries.",
        "Machine learning enables systems to learn and improve from data.",
        "Deep learning uses neural networks for pattern recognition.",
        "Natural language processing allows computers to understand human language.",
        "Artificial intelligence is widely used in multiple industries."
    ]
}

dataset = Dataset.from_dict(data)

# =========================
# LOAD MODEL (LIGHT + SAFE)
# =========================
summarizer = pipeline("summarization", model="t5-small")

# =========================
# ROUGE SETUP
# =========================
rouge = evaluate.load("rouge")

predictions = []
references = []

# =========================
# RUN MODEL
# =========================
for i in range(len(dataset)):
    article = dataset[i]["article"]
    reference = dataset[i]["summary"]

    pred = summarizer(article, max_length=50, min_length=10)[0]["summary_text"]

    predictions.append(pred)
    references.append(reference)

# =========================
# COMPUTE ROUGE
# =========================
results = rouge.compute(predictions=predictions, references=references)

scores = {
    "ROUGE-1": results["rouge1"] * 100,
    "ROUGE-2": results["rouge2"] * 100,
    "ROUGE-L": results["rougeL"] * 100
}

print("\n📊 ROUGE SCORES:")
for k, v in scores.items():
    print(f"{k}: {round(v,2)}")

# =========================
# GRAPH
# =========================
names = list(scores.keys())
values = list(scores.values())

plt.bar(names, values)
plt.title("ROUGE Score")
plt.ylabel("Score (%)")

for i, v in enumerate(values):
    plt.text(i, v + 1, f"{v:.1f}", ha='center')

plt.show()