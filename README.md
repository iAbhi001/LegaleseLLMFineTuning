# Legalese-Simplifier: Fine-Tuning Llama 3.2 for Legal Accessibility

A professional-grade LLM fine-tuning project that transforms dense, archaic legalese into plain, actionable English using **QLoRA** and **Llama 3.2-3B**.

## 🚀 Project Overview
Legal documents are often inaccessible to the general public. This project demonstrates how to specialize a general-purpose Large Language Model for the legal domain. By fine-tuning on a curated dataset of contract clauses, the model learns to maintain legal intent while drastically improving readability.

### Key Technical Features:
- **Base Model:** Llama 3.2-3B-Instruct
- **Fine-tuning Method:** QLoRA (4-bit Quantized Low-Rank Adaptation)
- **Framework:** Unsloth (Optimized for 2x faster training & 70% less VRAM)
- **Dataset:** 400+ high-quality pairs of complex legal clauses vs. simplified equivalents.

---

## 📂 Project Structure
```text
legalese-simplifier/
├── .gitignore               # Prevents large model weights from bloating the repo
├── README.md                # Project documentation
├── requirements.txt         # Environment dependencies
├── data/
│   └── legal_train.jsonl    # Specialized legal dataset (JSONL format)
├── notebooks/
│   └── Training_Llama3.ipynb# Colab-ready training pipeline
├── model_cards/
│   └── adapter_config.json  # PEFT/LoRA configuration details
└── app/
    └── main.py              # Gradio-based inference UI