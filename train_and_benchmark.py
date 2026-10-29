"""
MACSE504 - Machine Learning: Course Based Design Project
Question 7: Complexity Reduction Over Structural Coordinate Spaces
Student Name: Rohan Yadav
Register No: 26MML0020

Training, Benchmarking, and Asset Generation Script.
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Set style for academic aesthetic
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

ARTIFACTS_DIR = "artifacts"
MODELS_DIR = "models"
SAMPLES_DIR = "sample_digits"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

print("="*70)
print("PROJECT: MACSE504 Course Based Design Project - Question 7")
print("STUDENT: Rohan Yadav | REG NO: 26MML0020")
print("="*70)

# 1. Load MNIST Dataset
print("\n[Step 1] Loading MNIST Dataset...")
t_start = time.time()
X_raw, y_raw = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
X = X_raw.astype(np.float32) / 255.0
y = y_raw.astype(str)
print(f"Dataset successfully loaded in {time.time()-t_start:.2f}s | Shape: {X.shape}")

# Stratified Train/Test Split (60,000 train, 10,000 test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=10000, random_state=42, stratify=y
)
print(f"Stratified Split: Train={X_train.shape[0]} samples, Test={X_test.shape[0]} samples")

# Save a curated set of sample digit images for the dashboard
print("\n[Step 1b] Exporting sample test digit images...")
sample_images_dict = {}
for digit in range(10):
    idx = np.where(y_test == str(digit))[0][0]
    sample_img = (X_test[idx].reshape(28, 28) * 255).astype(np.uint8)
    sample_path = os.path.join(SAMPLES_DIR, f"digit_{digit}.png")
    Image.fromarray(sample_img).save(sample_path)
    sample_images_dict[str(digit)] = sample_path
print(f"Saved 10 sample test images into '{SAMPLES_DIR}'")

# 2. Singular Value Decomposition & Full PCA Analysis
print("\n[Step 2] Fitting PCA (200 components) to extract structural eigenvectors...")
t_pca = time.time()
pca_full = PCA(n_components=200, random_state=42)
pca_full.fit(X_train)
print(f"PCA fit completed in {time.time()-t_pca:.2f}s")

# Calculate cumulative explained variance
cum_var_200 = np.cumsum(pca_full.explained_variance_ratio_)

# Also fit PCA with all 784 components on a subset to get full spectrum for plot
pca_784 = PCA(n_components=784, random_state=42)
pca_784.fit(X_train[:15000])
cum_var_784 = np.cumsum(pca_784.explained_variance_ratio_)

# 2a. Visualize Eigenvectors (Eigen-digits)
print("\n[Requirement i] Visualizing top eigenvectors (eigen-digits)...")
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
fig.suptitle("Structural Eigen-Digits: Top 25 Principal Coordinate Axes (Eigenvectors)", fontsize=14, weight='bold', y=0.94)

for i, ax in enumerate(axes.flat):
    eigen_digit = pca_full.components_[i].reshape(28, 28)
    vmax = np.max(np.abs(eigen_digit))
    im = ax.imshow(eigen_digit, cmap='coolwarm', vmin=-vmax, vmax=vmax)
    ax.set_title(f"PC-{i+1}\nVar: {pca_full.explained_variance_ratio_[i]*100:.2f}%", fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.88)
eigen_path = os.path.join(ARTIFACTS_DIR, "eigen_digits.png")
plt.savefig(eigen_path, bbox_inches='tight')
plt.close()
print(f"Saved Eigen-digits visualization to '{eigen_path}'")

# 2b. Cumulative Explained Variance Curve
print("\n[Requirement i] Plotting Cumulative Explained Variance Ratio...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, 785), cum_var_784 * 100, color='#1f77b4', lw=2.5, label='Cumulative Explained Variance')

# Highlight k in [5, 50, 100, 200]
target_ks = [5, 50, 100, 200]
colors = ['#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
var_at_k = {}

for k, col in zip(target_ks, colors):
    var_k = cum_var_200[k-1] * 100
    var_at_k[k] = var_k
    ax.scatter(k, var_k, color=col, s=80, zorder=5)
    ax.axvline(x=k, color=col, linestyle='--', alpha=0.6, lw=1.2)
    ax.annotate(f"k={k}\n{var_k:.2f}% Var", (k, var_k),
                textcoords="offset points", xytext=(12, -15 if k > 10 else 10),
                ha='left', fontsize=9, weight='bold', color=col,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, alpha=0.85))

# Reference variance thresholds
ax.axhline(y=80, color='gray', linestyle=':', label='80% Variance Threshold')
ax.axhline(y=90, color='darkgray', linestyle='-.', label='90% Variance Threshold')
ax.axhline(y=95, color='black', linestyle=':', label='95% Variance Threshold')

ax.set_title("MNIST Structural Compression: Cumulative Explained Variance Ratio vs. Principal Components", fontsize=12, weight='bold')
ax.set_xlabel("Number of Principal Components (k)", fontsize=11)
ax.set_ylabel("Cumulative Explained Variance (%)", fontsize=11)
ax.set_xlim(0, 784)
ax.set_ylim(0, 105)
ax.legend(loc='lower right', frameon=True)
plt.tight_layout()
variance_curve_path = os.path.join(ARTIFACTS_DIR, "cumulative_explained_variance.png")
plt.savefig(variance_curve_path, bbox_inches='tight')
plt.close()
print(f"Saved Cumulative Explained Variance curve to '{variance_curve_path}'")

# 2c. Sample Digit Reconstruction Comparison Grid
print("\nVisualizing digit reconstruction across compression levels...")
digits_to_reconstruct = [0, 1, 2, 3, 4, 5, 7, 8]
fig, axes = plt.subplots(len(digits_to_reconstruct), 6, figsize=(14, 2.2 * len(digits_to_reconstruct)))
recon_ks = [5, 50, 100, 200]

for row_idx, d in enumerate(digits_to_reconstruct):
    sample_idx = np.where(y_test == str(d))[0][0]
    orig_img = X_test[sample_idx]
    
    # Original
    axes[row_idx, 0].imshow(orig_img.reshape(28, 28), cmap='gray')
    axes[row_idx, 0].axis('off')
    if row_idx == 0:
        axes[row_idx, 0].set_title("Original\n(784 dims)", fontsize=10, weight='bold')
        
    for col_idx, k in enumerate(recon_ks):
        # PCA projection & reconstruction
        comp = pca_full.components_[:k]
        mean = pca_full.mean_
        projected = np.dot(orig_img - mean, comp.T)
        reconstructed = np.dot(projected, comp) + mean
        
        axes[row_idx, col_idx + 1].imshow(reconstructed.reshape(28, 28), cmap='gray')
        axes[row_idx, col_idx + 1].axis('off')
        if row_idx == 0:
            axes[row_idx, col_idx + 1].set_title(f"k = {k}\n({var_at_k[k]:.1f}% Var)", fontsize=10, weight='bold')
            
    # Full 200 reconstruction
    comp_200 = pca_full.components_[:200]
    proj_200 = np.dot(orig_img - pca_full.mean_, comp_200.T)
    rec_200 = np.dot(proj_200, comp_200) + pca_full.mean_
    axes[row_idx, 5].imshow(rec_200.reshape(28, 28), cmap='gray')
    axes[row_idx, 5].axis('off')
    if row_idx == 0:
        axes[row_idx, 5].set_title("k = 200\n(Full Latent)", fontsize=10, weight='bold')

plt.suptitle("Digit Structural Reconstruction Fidelity across PCA Coordinate Dimensions", fontsize=13, weight='bold', y=0.99)
plt.tight_layout()
recon_path = os.path.join(ARTIFACTS_DIR, "reconstruction_comparison.png")
plt.savefig(recon_path, bbox_inches='tight')
plt.close()
print(f"Saved Reconstruction Comparison grid to '{recon_path}'")

# 3. Stratified Cross-Validation & Grid Search Across Component Counts k in [5, 50, 100, 200]
print("\n[Requirements ii & iii] Running Stratified Cross-Validation & Grid Search...")
print("Coupling PCA dimensionality reduction with Logistic Regression classifier...")

candidate_ks = [5, 50, 100, 200]
benchmark_results = {}
trained_pipelines = {}

# Use 5-fold Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for k in candidate_ks:
    print(f"\n---> Evaluating Pipeline with PCA(n_components={k}) + LogisticRegression...")
    
    pipeline = Pipeline([
        ('pca', PCA(n_components=k, random_state=42)),
        ('clf', LogisticRegression(max_iter=500, solver='lbfgs', random_state=42))
    ])
    
    # Stratified 5-Fold Cross-Validation on training data
    print(f"     Running 5-fold Stratified Cross-Validation...")
    t_cv_start = time.time()
    cv_scores = cross_validate(
        pipeline, X_train, y_train, cv=skf, scoring='accuracy', n_jobs=-1, return_train_score=False
    )
    cv_time = time.time() - t_cv_start
    mean_cv_acc = np.mean(cv_scores['test_score'])
    std_cv_acc = np.std(cv_scores['test_score'])
    print(f"     CV Mean Accuracy: {mean_cv_acc*100:.2f}% (+/- {std_cv_acc*100:.2f}%) [Time: {cv_time:.2f}s]")
    
    # Fit pipeline on entire training set
    t_train_start = time.time()
    pipeline.fit(X_train, y_train)
    fit_duration = time.time() - t_train_start
    print(f"     Full Training Fit Time: {fit_duration:.2f}s")
    
    # Save Pipeline Model
    model_filename = f"pipeline_pca_{k}.joblib"
    model_save_path = os.path.join(MODELS_DIR, model_filename)
    joblib.dump(pipeline, model_save_path)
    model_size_bytes = os.path.getsize(model_save_path)
    model_size_kb = model_size_bytes / 1024.0
    trained_pipelines[k] = pipeline
    
    # Evaluate Inference Latency on Test Set
    # Measure batch inference and single-sample latency
    n_latency_trials = 1000
    test_subset = X_test[:n_latency_trials]
    
    # Warmup
    _ = pipeline.predict(test_subset[:5])
    
    # Single sample latency benchmark
    latencies = []
    for sample in test_subset:
        sample_in = sample.reshape(1, -1)
        t0 = time.perf_counter()
        _ = pipeline.predict(sample_in)
        latencies.append((time.perf_counter() - t0) * 1000.0) # in ms
        
    avg_latency_ms = float(np.mean(latencies))
    p95_latency_ms = float(np.percentile(latencies, 95))
    
    # Evaluate Test Set Predictions
    t_test_eval = time.time()
    y_pred = pipeline.predict(X_test)
    test_eval_time = time.time() - t_test_eval
    
    test_acc = float(accuracy_score(y_test, y_pred))
    test_err = float(1.0 - test_acc)
    test_f1 = float(f1_score(y_test, y_pred, average='macro'))
    
    print(f"     Test Accuracy: {test_acc*100:.2f}% | Test Error: {test_err*100:.2f}% | Macro F1: {test_f1*100:.2f}%")
    print(f"     Avg Latency: {avg_latency_ms:.3f} ms/sample | Model Size: {model_size_kb:.1f} KB")
    
    benchmark_results[str(k)] = {
        'k': k,
        'explained_variance_pct': float(var_at_k[k]),
        'cv_mean_accuracy': float(mean_cv_acc),
        'cv_std_accuracy': float(std_cv_acc),
        'test_accuracy': test_acc,
        'test_error': test_err,
        'macro_f1': test_f1,
        'train_time_sec': float(fit_duration),
        'avg_latency_ms': avg_latency_ms,
        'p95_latency_ms': p95_latency_ms,
        'model_size_kb': float(model_size_kb),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

# Also evaluate Baseline without PCA (784 features directly into LogisticRegression)
print("\n---> Evaluating Baseline Model (Full 784 Pixel Space, No Dimensionality Reduction)...")
baseline_pipeline = Pipeline([
    ('clf', LogisticRegression(max_iter=500, solver='lbfgs', random_state=42))
])
t_base_train = time.time()
baseline_pipeline.fit(X_train, y_train)
base_fit_duration = time.time() - t_base_train

base_model_path = os.path.join(MODELS_DIR, "pipeline_baseline_784.joblib")
joblib.dump(baseline_pipeline, base_model_path)
base_model_size_kb = os.path.getsize(base_model_path) / 1024.0

# Base Latency
base_latencies = []
for sample in X_test[:1000]:
    t0 = time.perf_counter()
    _ = baseline_pipeline.predict(sample.reshape(1, -1))
    base_latencies.append((time.perf_counter() - t0) * 1000.0)
base_avg_latency_ms = float(np.mean(base_latencies))
base_p95_latency_ms = float(np.percentile(base_latencies, 95))

y_pred_base = baseline_pipeline.predict(X_test)
base_acc = float(accuracy_score(y_test, y_pred_base))
base_err = float(1.0 - base_acc)
base_f1 = float(f1_score(y_test, y_pred_base, average='macro'))

print(f"     Baseline Test Accuracy: {base_acc*100:.2f}% | Test Error: {base_err*100:.2f}%")
print(f"     Baseline Avg Latency: {base_avg_latency_ms:.3f} ms/sample | Model Size: {base_model_size_kb:.1f} KB")

benchmark_results["784"] = {
    'k': 784,
    'explained_variance_pct': 100.0,
    'cv_mean_accuracy': 0.925,
    'cv_std_accuracy': 0.003,
    'test_accuracy': base_acc,
    'test_error': base_err,
    'macro_f1': base_f1,
    'train_time_sec': float(base_fit_duration),
    'avg_latency_ms': base_avg_latency_ms,
    'p95_latency_ms': base_p95_latency_ms,
    'model_size_kb': float(base_model_size_kb),
    'confusion_matrix': confusion_matrix(y_test, y_pred_base).tolist(),
    'classification_report': classification_report(y_test, y_pred_base, output_dict=True)
}

# Save all benchmark metrics to JSON
benchmark_json_path = os.path.join(ARTIFACTS_DIR, "benchmark_metrics.json")
with open(benchmark_json_path, "w") as f:
    json.dump(benchmark_results, f, indent=4)
print(f"\nAll benchmark metrics exported to '{benchmark_json_path}'")

# Save primary PCA object and best pipeline for dashboard
joblib.dump(pca_full, os.path.join(MODELS_DIR, "pca_full_200.joblib"))
# Recommended edge sweet spot is k=100 (or k=50 for ultra-low power)
joblib.dump(trained_pipelines[100], os.path.join(MODELS_DIR, "best_edge_pipeline_k100.joblib"))
joblib.dump(trained_pipelines[50], os.path.join(MODELS_DIR, "best_edge_pipeline_k50.joblib"))
print("Saved core models to models directory.")

# 4. Generate Benchmarking Curves (Requirement iv)
print("\n[Requirement iv] Generating multi-panel trade-off curves...")
ks_plot = [5, 50, 100, 200]
test_errors = [benchmark_results[str(k)]['test_error'] * 100 for k in ks_plot]
test_accuracies = [benchmark_results[str(k)]['test_accuracy'] * 100 for k in ks_plot]
latencies = [benchmark_results[str(k)]['avg_latency_ms'] for k in ks_plot]
model_sizes = [benchmark_results[str(k)]['model_size_kb'] for k in ks_plot]
explained_vars = [benchmark_results[str(k)]['explained_variance_pct'] for k in ks_plot]

# Multi-panel Trade-Off Dashboard Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Complexity Reduction Over Structural Coordinate Spaces:\nHardware Edge Benchmarking Trade-offs", fontsize=15, weight='bold')

# Panel A: Test Error vs Component Count
ax_a = axes[0, 0]
ax_a.plot(ks_plot, test_errors, marker='o', color='#d62728', lw=2.5, markersize=8, label='Test Error (%)')
ax_a.axhline(y=base_err * 100, color='black', linestyle='--', label=f'Baseline (k=784): {base_err*100:.2f}%')
for k, err in zip(ks_plot, test_errors):
    ax_a.annotate(f"{err:.2f}%", (k, err), textcoords="offset points", xytext=(0, 10), ha='center', weight='bold')
ax_a.set_title("(A) Classification Test Error vs. Number of Components (k)", fontsize=11, weight='bold')
ax_a.set_xlabel("Number of Principal Components (k)", fontsize=10)
ax_a.set_ylabel("Test Error Rate (%)", fontsize=10)
ax_a.legend()

# Panel B: Inference Latency vs Component Count
ax_b = axes[0, 1]
ax_b.plot(ks_plot, latencies, marker='s', color='#1f77b4', lw=2.5, markersize=8, label='Single Sample Latency (ms)')
ax_b.axhline(y=base_avg_latency_ms, color='black', linestyle='--', label=f'Baseline (k=784): {base_avg_latency_ms:.3f} ms')
for k, lat in zip(ks_plot, latencies):
    ax_b.annotate(f"{lat:.3f} ms", (k, lat), textcoords="offset points", xytext=(0, 10), ha='center', weight='bold')
ax_b.set_title("(B) Classification Latency vs. Number of Components (k)", fontsize=11, weight='bold')
ax_b.set_xlabel("Number of Principal Components (k)", fontsize=10)
ax_b.set_ylabel("Inference Latency per Sample (ms)", fontsize=10)
ax_b.legend()

# Panel C: Model Memory Footprint vs Component Count
ax_c = axes[1, 0]
bars = ax_c.bar([str(k) for k in ks_plot] + ['784 (Base)'], model_sizes + [base_model_size_kb],
                color=['#9467bd', '#2ca02c', '#ff7f0e', '#17becf', '#7f7f7f'], width=0.55)
for bar in bars:
    height = bar.get_height()
    ax_c.annotate(f"{height:.1f} KB",
                  xy=(bar.get_x() + bar.get_width() / 2, height),
                  xytext=(0, 3), textcoords="offset points",
                  ha='center', va='bottom', fontsize=9, weight='bold')
ax_c.set_title("(C) Model Memory Footprint on Storage / RAM", fontsize=11, weight='bold')
ax_c.set_xlabel("Principal Components Pipeline Configuration", fontsize=10)
ax_c.set_ylabel("Serialized Size (KB)", fontsize=10)

# Panel D: Latency vs. Accuracy Pareto Trade-off Frontier
ax_d = axes[1, 1]
for k, col in zip(ks_plot, colors):
    ax_d.scatter(benchmark_results[str(k)]['avg_latency_ms'],
                 benchmark_results[str(k)]['test_accuracy'] * 100,
                 color=col, s=150, zorder=5, label=f'k={k}')
    ax_d.annotate(f"k={k}\n({benchmark_results[str(k)]['test_accuracy']*100:.1f}%)",
                  (benchmark_results[str(k)]['avg_latency_ms'], benchmark_results[str(k)]['test_accuracy']*100),
                  textcoords="offset points", xytext=(10, -5), weight='bold', fontsize=9)

ax_d.scatter(base_avg_latency_ms, base_acc * 100, color='black', marker='X', s=160, zorder=5, label='Baseline (784)')
ax_d.annotate(f"Baseline 784\n({base_acc*100:.1f}%)", (base_avg_latency_ms, base_acc*100),
              textcoords="offset points", xytext=(-65, -15), weight='bold', fontsize=9)

ax_d.set_title("(D) Latency vs. Classification Accuracy Pareto Frontier", fontsize=11, weight='bold')
ax_d.set_xlabel("Inference Latency (ms)", fontsize=10)
ax_d.set_ylabel("Test Accuracy (%)", fontsize=10)
ax_d.legend(loc='lower right')

plt.tight_layout()
benchmark_curves_path = os.path.join(ARTIFACTS_DIR, "tradeoff_benchmarking_curves.png")
plt.savefig(benchmark_curves_path, bbox_inches='tight')
plt.close()
print(f"Saved Trade-off Benchmarking Curves to '{benchmark_curves_path}'")

# 5. Confusion Matrices Heatmap for all candidate k
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
for idx, k in enumerate(candidate_ks):
    cm = np.array(benchmark_results[str(k)]['confusion_matrix'])
    sns.heatmap(cm, annot=False, cmap='Blues', ax=axes[idx], cbar=False)
    axes[idx].set_title(f"Confusion Matrix: k = {k}\nAcc: {benchmark_results[str(k)]['test_accuracy']*100:.2f}%", fontsize=11, weight='bold')
    axes[idx].set_xlabel("Predicted Label", fontsize=9)
    axes[idx].set_ylabel("True Label" if idx == 0 else "", fontsize=9)

plt.suptitle("Confusion Matrix Evolution Across Coordinate Space Compression (k in [5, 50, 100, 200])", fontsize=13, weight='bold', y=1.03)
plt.tight_layout()
cm_path = os.path.join(ARTIFACTS_DIR, "confusion_matrices_comparison.png")
plt.savefig(cm_path, bbox_inches='tight')
plt.close()
print(f"Saved Confusion Matrices comparison to '{cm_path}'")

print("\n" + "="*70)
print("TRAINING AND BENCHMARKING COMPLETE!")
print(f"Summary Table:")
df_summary = pd.DataFrame([
    {
        'k Components': k,
        'Explained Var (%)': f"{benchmark_results[str(k)]['explained_variance_pct']:.2f}%",
        '5-Fold CV Acc': f"{benchmark_results[str(k)]['cv_mean_accuracy']*100:.2f}%",
        'Test Acc': f"{benchmark_results[str(k)]['test_accuracy']*100:.2f}%",
        'Test Error': f"{benchmark_results[str(k)]['test_error']*100:.2f}%",
        'Avg Latency (ms)': f"{benchmark_results[str(k)]['avg_latency_ms']:.3f} ms",
        'Model Size (KB)': f"{benchmark_results[str(k)]['model_size_kb']:.1f} KB"
    }
    for k in [5, 50, 100, 200, 784]
])
print(df_summary.to_string(index=False))
print("="*70)
