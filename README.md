# 🏥 Weakly-Supervised Diagnosis Identification from Clinical Discharge Letters

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research-orange)

This repository implements a **weakly supervised NLP pipeline** for identifying diseases from clinical discharge letters **without requiring document-level manual annotation**.

The approach combines:
- Transformer-based embeddings
- Unsupervised clustering of diagnosis strings
- Keyword-based weak supervision
- Transformer fine-tuning for classification

## ⚡ Quick Start (5 minutes)

```bash
# 1. Install environment
conda env create -f environment_fse.yml
conda activate fse

# 2. Run clustering
python -m experiments.run_clustering EMBEDDING_MODEL_NAME

# 3. Assign weak labels
python -m experiments.run_weak_label_assignment \
  --input_csv df_fse_hdbscan_grid.csv \
  --cluster_col hdb_mcs10_ms5 \
  --output_label_col weak_label

# 4. Train classifier
python -m experiments.run_classification EMBEDDING_MODEL_NAME \
  --input_csv df_with_weak_labels.csv \
  --weak_label_col weak_label \
  --gold_label_col gold_label \
  --text_col testo_clean
```

---

## 🧠 Method Overview

The pipeline follows three main stages.

### 0️⃣ TPT — Transformer Pre-Training

Domain adaptation of a transformer model on Italian medical text via masked language modeling.

---

### 1️⃣ SAL — Semi-Automatic Labelling

#### 1.1 Diagnosis Extraction

* Extract diagnosis strings using rule-based patterns
* Handles heterogeneous discharge letter formats

#### 1.2 Diagnosis Clustering

* Transformer embeddings
* PCA dimensionality reduction
* HDBSCAN clustering
* Optional second-level clustering

This step groups semantically similar diagnoses into clusters.

#### 1.3 Mapping Clusters to Diseases

Clusters are mapped using keyword rules:

* Positive keywords → must appear
* Negative keywords → must not appear

Weak labels are assigned as:

```text
ỹ_i = 1 if cluster ∈ selected_clusters
      0 otherwise
```

Keyword matching is applied at the **cluster level**, improving robustness to lexical variability.

---

### 2️⃣ WLC — Weakly Labelled Classification

* Train transformer classifier on weak labels
* Input: full discharge letter
* Loss: binary cross-entropy

## 📁 Repository Structure

```text
.
├── experiments/
│   ├── run_clustering.py
│   ├── run_weak_label_assignment.py
│   ├── run_classification.py
│   ├── run_other_clustering_algorithms.py
│   ├── cluster_robustness.py
│
├── pipeline/
│   ├── diagnosis_extraction.py
│   ├── diagnosis_clustering.py
│   ├── weak_labels_selection.py
│   ├── classification.py
│   ├── compute_embeddings.py
│   ├── unsupervised_baselines.py
│
├── utils/
│   └── utils.py
│
├── environment_fse.yml
└── README.md
```

---

## 📊 Input Data Format

Required:

* `text` → discharge letter text

Optional:

* `gold_label` → ground truth label

---

## 🔁 Generalization to New Diseases

The pipeline is **disease-agnostic**.

To adapt it:

1. Modify keyword definitions in `pipeline/weak_labels_selection.py`
2. Run the pipeline unchanged

Document-level manual annotations are required only if you want to evaluate results against gold labels.

---

## 🧪 Tested Use Cases

* Bronchiolitis
* Bronchitis

Can be extended to any disease with keyword definitions.


---

## 📬 Contact

Vittorio Torri
MOX - Politecnico di Milano
vittorio.torri [at] polimi [dot] com
