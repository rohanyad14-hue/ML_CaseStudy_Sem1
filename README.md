# MNIST Complexity Reduction Over Structural Coordinate Spaces

**Course:** MACSE504 – Machine Learning (Course-Based Design Project)  
**Question:** 7 – Complexity Reduction Over Structural Coordinate Spaces  
**Student Name:** Rohan Yadav  
**Register Number:** 26MML0020  

---

## 📌 Project Overview

This project investigates **complexity reduction** and **trade-off optimization** in high-dimensional coordinate spaces using the standard **MNIST 70,000 handwritten digit dataset**.

By applying **Principal Component Analysis (PCA)** to transform the raw $784$-dimensional spatial pixel coordinates into reduced orthogonal subspaces ($k \in \{5, 50, 100, 200\}$), we benchmark the computational and classification performance of **Multinomial Logistic Regression** against the full-dimensional baseline ($k=784$).

### Key Exploration Dimensions
1. **Dimensionality vs. Information Preservation:** Scree plots, cumulative explained variance ratio, and digit reconstruction fidelity.
2. **Classification Performance:** Multi-class accuracy, precision, recall, confusion matrices, and macro/micro F1-scores.
3. **Edge Deployment Trade-offs:** Model memory footprint (disk/RAM), training duration, and per-sample inference latency.
4. **Interactive Dashboard:** Real-time handwritten digit recognition with canvas drawing, digit reconstruction viewer, and live benchmarking.

---

## 🚀 Quick Start (One-Click Run)

### 🪟 Windows
Double-click `run.bat` or run in Command Prompt / PowerShell:
```cmd
run.bat
```
*This will automatically detect your virtual environment or system Python, verify dependencies, and launch the Streamlit dashboard in your default browser.*

### 🐧 Linux & 🍎 macOS
Make the script executable and run:
```bash
chmod +x run.sh
./run.sh
```

---

## 🛠️ Manual Installation & Execution

If you prefer to configure and launch manually:

### 1. Clone or Open the Project
```bash
cd ML_CaseStudy
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Interactive Streamlit App
```bash
streamlit run app.py
```
The application will be accessible at `http://localhost:8501`.

---

## 📂 Project Structure

```text
ML_CaseStudy/
├── app.py                                                # Interactive Streamlit Web Application
├── train_and_benchmark.py                                # End-to-end training & benchmarking script
├── execute_notebook.py                                   # Headless notebook execution runner
├── generate_notebook.py                                  # Script generating/updating notebook cells
├── MACSE504_Project_Question7_26MML0020_Rohan_Yadav.ipynb # Primary submission notebook with complete outputs
├── MNIST_PCA_LogisticRegression_CaseStudy7.ipynb         # Standalone reference notebook
├── requirements.txt                                      # Project Python dependencies
├── run.bat                                               # One-click launcher for Windows
├── run.sh                                                # One-click launcher for Linux/macOS
├── .gitignore                                            # Git ignore file for Python & ML artifacts
├── models/                                               # Serialized trained pipelines (.joblib)
│   ├── best_edge_pipeline_k50.joblib
│   ├── best_edge_pipeline_k100.joblib
│   ├── pipeline_baseline_784.joblib
│   ├── pipeline_pca_5.joblib
│   ├── pipeline_pca_50.joblib
│   ├── pipeline_pca_100.joblib
│   ├── pipeline_pca_200.joblib
│   └── pca_full_200.joblib
├── artifacts/                                            # Benchmark plots & evaluation figures
│   ├── benchmark_metrics.json
│   ├── cumulative_explained_variance.png
│   ├── reconstruction_comparison.png
│   ├── tradeoff_benchmarking_curves.png
│   ├── confusion_matrices_comparison.png
│   └── eigen_digits.png
└── sample_digits/                                        # Pre-extracted test samples for quick evaluation
```

---

## 🔬 Reproducing Training and Benchmarks

To retrain all pipelines, generate high-resolution comparison plots, and produce the benchmark metrics JSON:
```bash
python train_and_benchmark.py
```

To re-execute and update the Jupyter notebooks with fresh outputs:
```bash
python execute_notebook.py
```

---

## 📊 Core Findings & Summary

| Model Configuration | Input Dimensions | Dimension Reduction (%) | Retained Variance (%) | Test Accuracy (%) | Approx. Model Size |
|:---------------------|:----------------:|:-----------------------:|:---------------------:|:-----------------:|:------------------:|
| **PCA ($k=5$)**      | $5$              | $99.36\%$               | $\sim 31.3\%$         | $\sim 64.8\%$     | $\sim 21 \text{ KB}$ |
| **PCA ($k=50$)**     | $50$             | $93.62\%$               | $\sim 82.5\%$         | $\sim 91.8\%$     | $\sim 166 \text{ KB}$ |
| **PCA ($k=100$)** *(Optimal)* | $100$   | $87.24\%$               | $\sim 91.5\%$         | $\sim 92.5\%$     | $\sim 327 \text{ KB}$ |
| **PCA ($k=200$)**    | $200$            | $74.49\%$               | $\sim 96.6\%$         | $\sim 92.7\%$     | $\sim 650 \text{ KB}$ |
| **Baseline (Raw)**   | $784$            | $0.00\%$                | $100.0\%$             | $\sim 92.4\%$     | $\sim 64 \text{ KB}$ (weights only) |

> **Edge Recommendation:** $k=100$ components achieves optimal trade-off balance, retaining $>91.5\%$ variance and exceeding baseline classification accuracy while removing noise and redundancy from the input space.
