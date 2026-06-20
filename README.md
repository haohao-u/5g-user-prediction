# 5G User Prediction

This repository contains a course project for predicting whether a telecom user is a 5G user. The task is formulated as a binary classification problem and evaluated with AUC.

## Dataset

Place the course dataset at the project root:

```text
train.csv
```

Columns:

- `id`: sample identifier
- `cat_0` to `cat_19`: categorical features
- `num_0` to `num_37`: numerical features
- `target`: binary label

## Models

The project implements two machine learning algorithms:

- Logistic Regression with balanced class weights
- Random Forest with balanced subsampling

Current validation results using 50000 rows:

| Model | AUC |
|---|---:|
| Logistic Regression | 0.8428 |
| Random Forest | 0.8716 |

Best model: `random_forest`.

## Quick Start

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run tests:

```bash
python -m pytest tests -q
```

Train on a quick sample:

```bash
python scripts/train_models.py --data train.csv --output outputs --max-rows 50000
```

Train on all rows:

```bash
python scripts/train_models.py --data train.csv --output outputs --max-rows 0
```

Generated artifacts are saved in `outputs/`, including metrics, plots, and the best model.
