# Addressee Recognition and Reference Style Classification in MPDD

This repository contains code for evaluating large language models (LLMs) on **addressee recognition** and **utterance reference style classification** using the MPDD (Multi-Party Dialogue Dataset).

---

## 📁 Project Structure

```
.
├── addressee recognition/        # Listener prediction task
│   ├── listener.py               # Main entry for running listener prediction
│   ├── utils.py                  # Utilities for prompt construction and formatting
│   └── prompt.yaml               # Prompt templates used in listener.py
│
├── type classification/          # Reference style classification (A/B/C)
│   └── classify.py               # Classifies utterances into reference types
│
├── mpdd/                         # Dataset files
│   ├── dialogue.json             # Processed dialogue data (English or general version)
│   ├── dialogue_chinese.json     # Original Chinese dialogue version
│   ├── metadata.json             # Metadata including speakers, listeners, emotion, etc.
│   ├── whole_dialogue_prepro.json # Full dialogue format for whole-context setting
│   └── readme.txt                # Dataset description (optional or placeholder)
│
├── postprocess/                  # Evaluation and analysis scripts
│   ├── calculate_f1.py           # Computes Precision, Recall, F1, Exact Match
│   ├── chars_num.py              # Analyzes number of characters per dialogue
│   ├── heatmap.py                # Generates heatmap visualizations for accuracy
│   ├── pvalue.py                 # Performs significance testing (e.g., t-test)
│   └── ref_type.py               # Reference-type-wise breakdown for listener prediction
```

---

## 🧠 Tasks Overview

### 1. Addressee Recognition

Predicts the intended **addressee(s)** of a given utterance in multi-party dialogue.

#### Example command:

```bash
python listener.py \
  --output_file results/R1_llama/relation_name_noinfo_whole.json \
  --prompt_mode new_task \
  --start 0 \
  --anonymity name \
  --info_level none \
  --context_mode whole
```

**Arguments**:

* `--output_file`: Path to store model predictions
* `--prompt_mode`: Prompt template mode (`new_task`)
* `--start`: Starting index for batch processing
* `--anonymity`: `name` or `alias`
* `--info_level`: `none`, `position`, or `relation`
* `--context_mode`: `pre` or `whole`

---

### 2. Addressee Mention Type Classification

Classifies how the speaker refers to the addressee in a given utterance:

* **A**: Name/Nickname
* **B**: Role title (e.g., Mom, Boss)
* **C**: No explicit reference

#### Run:

```bash
python classify.py
```

This will output predictions (A/B/C) for all utterances in MPDD with ≥3 characters.

---

### 3. Postprocessing & Analysis

Located in `postprocess/`:

* `calculate_f1.py`: Metric calculation (Precision, Recall, F1, Exact Match)
* `heatmap.py`: Visualization of model accuracy (e.g., heatmap grouped by prompt setting)
* `ref_type.py`: Aggregates results by reference type (A/B/C)
* `pvalue.py`: Statistical significance testing

---

## 📦 Dataset: MPDD

The **MPDD** folder contains the full dialogue annotations. Key files:

* `dialogue.json`: Base data used for listener prediction
* `whole_dialogue_prepro.json`: Used in full-context experiments
* `metadata.json`: Speaker-listener relations and emotions
