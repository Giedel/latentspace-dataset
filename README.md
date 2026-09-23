# LatentSpace Agentic Training Dataset Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giedel/latentspace-dataset/blob/master/latentspace_dataset/notebook/latentspace_dataset_builder.ipynb)

A complete offline semantic middleware and agentic assistant training dataset pipeline for **LatentSpace**.

The pipeline ingests 8 heterogeneous datasets (23,384 records across text, speech transcripts, and OCR documents), preserves original source fields, classifies records into a controlled domain/category taxonomy, performs agentic schema completion using the canonical `dispatch_actions` universal tool contract, validates examples, extracts synthetic agentic fields, and exports clean OpenAI-format JSONL datasets (80% Train, 10% Validation, 10% Test).

---

### Quick Start: Open Directly in Google Colab

Click the badge above or navigate to:  
👉 **[Open in Google Colab](https://colab.research.google.com/github/Giedel/latentspace-dataset/blob/master/latentspace_dataset/notebook/latentspace_dataset_builder.ipynb)**

### In Google Colab:
1. Select **Runtime** > **Run all** (`Ctrl+F9`).
2. *(Optional)* Add your `OPENROUTER_API_KEY` to Colab's **Secrets** (🔑 Key icon on the left sidebar) if you want to perform online LLM calls.
3. Once finished, download the outputs directly from the last cell or via `latentspace_outputs.zip`.

---

### Baseline Source Datasets (23,384 records)
- **BANKING77**: 7,000 records (Text)
- **CLINC150**: 8,000 records (Text)
- **HWU64**: 5,000 records (Text)
- **MINDS-14 US**: 563 records (Audio transcripts)
- **MINDS-14 EXT**: 1,246 records (Audio transcripts)
- **CORD-v2**: 800 records (Receipt OCR)
- **SROIE**: 626 records (Receipt OCR)
- **FUNSD**: 149 records (Form OCR)

---

### Repository Structure
```text
latentspace-dataset/
├── README.md
├── build_notebook.py
├── latentspace_pipeline.py
├── latentspace_dataset/
│   ├── notebook/
│   │   └── latentspace_dataset_builder.ipynb
│   ├── config/
│   │   └── intent_registry.json
│   ├── prompts/
│   │   ├── annotation_prompt.txt
│   │   ├── synthetic_generation_prompt.txt
│   │   └── validation_prompt.txt
│   └── output/
│       ├── original_dataset.csv
│       ├── synthetic_dataset.csv
│       ├── combined_dataset.csv
│       ├── rejected_records.csv
│       ├── dataset_report.json
│       ├── latentspace_complete_dataset.jsonl
│       ├── latentspace_train.jsonl
│       ├── latentspace_validation.jsonl
│       └── latentspace_test.jsonl
```
