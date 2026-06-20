# 5G User Prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible 5G user prediction coursework project with at least two AI algorithms, AUC evaluation, plots, a Jupyter Notebook, and written report materials.

**Architecture:** Core reusable modeling functions live in `src/modeling.py`; `scripts/train_models.py` runs the experiment and writes metrics, charts, and the best model. The notebook demonstrates the same workflow for submission, while report files summarize the problem, methods, results, and division of labor.

**Tech Stack:** Python, pandas, scikit-learn, matplotlib, seaborn, joblib, nbformat, pytest.

## Global Constraints

- Use `train.csv` as the local dataset.
- Predict `target` from `cat_0` to `cat_19` and `num_0` to `num_37`.
- Implement at least two algorithms.
- Evaluate with AUC.
- Keep generated code and experiment outputs reproducible with `random_state=42`.
- This folder is not a git repository, so verification replaces commit steps.

---

### Task 1: Core Modeling API

**Files:**
- Create: `src/__init__.py`
- Create: `src/modeling.py`
- Test: `tests/test_modeling.py`

**Interfaces:**
- Produces: `CAT_FEATURES: list[str]`, `NUM_FEATURES: list[str]`, `TARGET: str`
- Produces: `load_training_data(csv_path, max_rows=None) -> pandas.DataFrame`
- Produces: `split_features_target(df) -> tuple[pandas.DataFrame, pandas.Series]`
- Produces: `build_preprocessor() -> sklearn.compose.ColumnTransformer`
- Produces: `create_models(random_state=42) -> dict[str, sklearn.pipeline.Pipeline]`
- Produces: `evaluate_models(models, x_train, y_train, x_valid, y_valid) -> dict[str, float]`

- [x] **Step 1: Write failing tests**
- [x] **Step 2: Run tests and confirm failure because `src` does not exist**
- [x] **Step 3: Implement the core modeling module**
- [x] **Step 4: Run tests and confirm pass**

### Task 2: Training Script and Outputs

**Files:**
- Create: `scripts/train_models.py`
- Create after run: `outputs/metrics.json`
- Create after run: `outputs/model_scores.csv`
- Create after run: `outputs/best_model.joblib`
- Create after run: `outputs/target_distribution.png`
- Create after run: `outputs/auc_comparison.png`
- Create after run: `outputs/roc_curves.png`

**Interfaces:**
- Consumes: `src.modeling.create_models`, `src.modeling.split_features_target`
- Produces: reproducible command-line experiment runner

- [x] **Step 1: Implement CLI arguments for data path, output directory, row cap, split size, and random seed**
- [x] **Step 2: Train Logistic Regression and Random Forest pipelines**
- [x] **Step 3: Save metrics, model, and charts**
- [x] **Step 4: Run a sample training command and inspect output**

### Task 3: Submission Materials

**Files:**
- Create: `5g_user_prediction.ipynb`
- Create: `README.md`
- Create: `report/5G用户预测实验报告.md`
- Create: `report/5G用户预测实验报告.docx`
- Create: `report/成员分工与小组自评表.md`
- Create: `report/成员分工与小组自评表.docx`

**Interfaces:**
- Consumes: `outputs/metrics.json` and chart files when available
- Produces: coursework-facing notebook and report assets

- [x] **Step 1: Generate a notebook that loads data, explores imbalance, trains two models, evaluates AUC, and visualizes results**
- [x] **Step 2: Generate a concise Chinese report without source code**
- [x] **Step 3: Generate a README with English usage instructions**
- [x] **Step 4: Verify notebook syntax and report file existence**
