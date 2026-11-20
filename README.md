# Quantized Legal Tagger

A fine-tuned GPT-2 model for legal clause classification using the LEDGAR dataset, with dynamic INT8 quantization to reduce model storage and inference latency.

## Overview

This project explores the optimization of a transformer-based language model for legal text classification. GPT-2 was fine-tuned on selected categories from the LexGLUE / LEDGAR dataset and subsequently optimized through dynamic quantization.

A Streamlit dashboard visualizes model size and inference latency before and after quantization.

## Objectives

- Fine-tune GPT-2 on legal clause data.
- Classify legal clauses into predefined categories.
- Apply dynamic INT8 quantization.
- Compare model size and inference latency.
- Visualize performance metrics using Streamlit.

## Dataset

- **Dataset:** LexGLUE – LEDGAR
- **Task:** Legal clause classification
- **Selected categories:** Amendments, Governing Laws, Survival, Entire Agreements, Severability, Notices, Counterparts and Assignments.
- **Training sample:** 2,500 rows
- **Test sample:** 500 rows

## Model Optimization Results

| Metric | Original Model | Dynamic INT8 |
|---|---:|---:|
| Model Size | 476.20 MB | 268.60 MB |
| Mean Inference Latency | 78.94 ms | 61.27 ms |

### Optimization Improvements

- **43.60% reduction** in model size
- **22.38% improvement** in measured inference latency
- Approximately **207.60 MB** of storage saved

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Scikit-learn
- Pandas
- Streamlit

## Project Structure

```text
Quantized_Legal_Tagger/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── original/
│   └── dynamic_quantized/
├── src/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── quantize.py
│   ├── benchmark.py
│   └── utils.py
├── app.py
├── benchmark_results.csv
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Quantized_Legal_Tagger.git
cd Quantized_Legal_Tagger
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Pipeline

### 1. Dataset preparation

```bash
python src/prepare_data.py
```

### 2. Fine-tuning

```bash
python src/train.py
```

### 3. Evaluation

```bash
python src/evaluate.py
```

### 4. Quantization

```bash
python src/quantize.py
```

### 5. Benchmarking

```bash
python src/benchmark.py
```

### 6. Dashboard

```bash
python -m streamlit run app.py
```

## Dashboard

The dashboard displays:

- Original and quantized model sizes
- Inference latency comparisons
- Storage reduction
- Benchmark data

## Key Findings

Dynamic INT8 quantization reduced the model's storage footprint by 43.60% and measured inference latency by 22.38% on the local benchmarking setup.

These results demonstrate the potential of model quantization to improve the efficiency of transformer-based NLP models.

## Limitations

- Results depend