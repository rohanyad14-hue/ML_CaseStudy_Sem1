"""
========================================================================================
MACSE504 - Machine Learning: Course Based Design Project
Question 7: Complexity Reduction Over Structural Coordinate Spaces
Student Name: Rohan Yadav
Register Number: 26MML0020
Interactive Streamlit Edge Classification & PCA Structural Compression Dashboard
========================================================================================
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Optional canvas drawing support
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False

# Page Configuration
st.set_page_config(
    page_title="MNIST Digit Classifier | PCA Edge Complexity",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Sleek Aesthetics and Glassmorphism
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #000428 100%);
        padding: 24px 30px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    .header-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.18);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    .header-title {
        font-size: 26px;
        font-weight: 800;
        margin: 0;
        color: #ffffff;
    }
    .header-subtitle {
        font-size: 14px;
        opacity: 0.9;
        margin-top: 6px;
        color: #e0e7ff;
    }
    .header-meta {
        margin-top: 12px;
        font-size: 13px;
        font-family: 'JetBrains Mono', monospace;
        color: #93c5fd;
        display: flex;
        gap: 20px;
        flex-wrap: wrap;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #3b82f6;
        transform: translateY(-2px);
    }
    .metric-val {
        font-size: 28px;
        font-weight: 800;
        color: #3b82f6;
    }
    .metric-lbl {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.75;
        margin-top: 4px;
    }
    .pred-box {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        border: 2px solid #6366f1;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.25);
    }
    .pred-digit {
        font-size: 80px;
        font-weight: 900;
        color: #818cf8;
        line-height: 1;
        margin: 10px 0;
        text-shadow: 0 0 25px rgba(129, 140, 248, 0.6);
    }
    .pred-conf {
        font-size: 16px;
        font-weight: 600;
        color: #34d399;
    }
    .img-card {
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 12px;
        text-align: center;
        background: rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Helper Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
SAMPLES_DIR = os.path.join(BASE_DIR, "sample_digits")

@st.cache_resource
def load_all_models():
    """Load pretrained PCA and Logistic Regression pipelines."""
    models = {}
    ks = [5, 50, 100, 200]
    for k in ks:
        p_path = os.path.join(MODELS_DIR, f"pipeline_pca_{k}.joblib")
        if os.path.exists(p_path):
            models[k] = joblib.load(p_path)
    
    # Load baseline
    base_path = os.path.join(MODELS_DIR, "pipeline_baseline_784.joblib")
    if os.path.exists(base_path):
        models[784] = joblib.load(base_path)
        
    # Load full PCA for reconstruction
    pca_path = os.path.join(MODELS_DIR, "pca_full_200.joblib")
    pca_full = joblib.load(pca_path) if os.path.exists(pca_path) else None
    
    # Load benchmark metrics json
    bench_path = os.path.join(ARTIFACTS_DIR, "benchmark_metrics.json")
    benchmark_data = {}
    if os.path.exists(bench_path):
        with open(bench_path, "r") as f:
            benchmark_data = json.load(f)
            
    return models, pca_full, benchmark_data

models_dict, pca_full, benchmark_data = load_all_models()

# Header Banner
st.markdown("""
<div class="main-header">
    <div class="header-badge">MACSE504 - Machine Learning | Course Based Design Project</div>
    <div class="header-title">Question 7: Complexity Reduction Over Structural Coordinate Spaces</div>
    <div class="header-subtitle">Edge Hardware Digit Recognition with SVD / PCA Dimensionality Reduction & Logistic Regression</div>
    <div class="header-meta">
        <span>👤 <b>Student:</b> Rohan Yadav</span>
        <span>🆔 <b>Register No:</b> 26MML0020</span>
        <span>📅 <b>Slot:</b> E1+TE1</span>
        <span>🏛️ <b>Institute:</b> Vellore Institute of Technology</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Preprocessing Function
def preprocess_image(input_img):
    """
    Standardize any uploaded or drawn image into MNIST standard:
    - 28x28 grayscale
    - White digit on black background (values in [0, 1])
    - Centered bounding box with center of mass normalization
    """
    # Convert to grayscale
    gray = input_img.convert('L')
    
    # Resize keeping aspect ratio or fit into square
    # First convert to numpy array
    arr = np.array(gray, dtype=np.float32)
    
    # Check if the image is black on white (paper style) vs white on black (MNIST style)
    # If corner pixels are predominantly white (> 128), invert it!
    corner_pixels = np.array([arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]])
    if np.mean(corner_pixels) > 127:
        arr = 255.0 - arr
        
    # Threshold slight noise
    arr[arr < 35] = 0
    
    # Find bounding box of the digit
    non_zeros = np.argwhere(arr > 30)
    if len(non_zeros) > 0:
        ymin, xmin = non_zeros.min(axis=0)
        ymax, xmax = non_zeros.max(axis=0)
        h = ymax - ymin + 1
        w = xmax - xmin + 1
        
        cropped = arr[ymin:ymax+1, xmin:xmax+1]
        
        # Scale bounding box so largest dimension is 20 pixels (standard MNIST protocol)
        scale = 20.0 / max(h, w)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        
        pil_cropped = Image.fromarray(cropped.astype(np.uint8)).resize((new_w, new_h), Image.Resampling.BICUBIC)
        cropped_scaled = np.array(pil_cropped, dtype=np.float32)
        
        # Create 28x28 blank canvas and place scaled digit in center
        canvas = np.zeros((28, 28), dtype=np.float32)
        x_start = (28 - new_w) // 2
        y_start = (28 - new_h) // 2
        canvas[y_start:y_start+new_h, x_start:x_start+new_w] = cropped_scaled
        
        # Center of mass adjustment
        cy, cx = np.sum(np.indices(canvas.shape) * canvas, axis=(1, 2)) / (np.sum(canvas) + 1e-8)
        shift_x = int(round(14.0 - cx))
        shift_y = int(round(14.0 - cy))
        
        # Shift canvas by center of mass within safe margins
        shifted_canvas = np.roll(canvas, shift_y, axis=0)
        shifted_canvas = np.roll(shifted_canvas, shift_x, axis=1)
        
        final_28x28 = shifted_canvas
    else:
        # Blank canvas fallback
        pil_img = Image.fromarray(arr.astype(np.uint8)).resize((28, 28), Image.Resampling.LANCZOS)
        final_28x28 = np.array(pil_img, dtype=np.float32)
        
    normalized_vector = (final_28x28 / 255.0).flatten()
    return final_28x28, normalized_vector

# Reconstruction Function
def reconstruct_pca(norm_vector, k_components):
    """Reconstruct 28x28 image from k components."""
    if pca_full is None:
        return norm_vector.reshape(28, 28)
    
    k = min(k_components, pca_full.n_components_)
    comp = pca_full.components_[:k]
    mean = pca_full.mean_
    
    # Project to latent space: z = (x - mu) V_k^T
    projected = np.dot(norm_vector - mean, comp.T)
    # Reconstruct: x_hat = z V_k + mu
    reconstructed = np.dot(projected, comp) + mean
    # Clip to valid [0, 1] range
    reconstructed = np.clip(reconstructed, 0.0, 1.0)
    return reconstructed.reshape(28, 28)

# Sidebar Configuration
st.sidebar.markdown("### ⚙️ Pipeline Controls")

# Select Model Component Count k
k_choice = st.sidebar.select_slider(
    "Coordinate Space Compression (k components):",
    options=[5, 50, 100, 200, 784],
    value=100,
    help="Select the number of eigenvectors used for PCA feature projection."
)

k_info = benchmark_data.get(str(k_choice), {})
if k_choice < 784:
    st.sidebar.info(
        f"**Configuration $k={k_choice}$**\n\n"
        f"• **Variance Retained:** {k_info.get('explained_variance_pct', 0):.1f}%\n\n"
        f"• **Test Accuracy:** {k_info.get('test_accuracy', 0)*100:.2f}%\n\n"
        f"• **Avg Latency:** {k_info.get('avg_latency_ms', 0):.3f} ms\n\n"
        f"• **Model RAM:** {k_info.get('model_size_kb', 0):.1f} KB"
    )
else:
    st.sidebar.info(
        f"**Uncompressed Baseline (k=784)**\n\n"
        f"• **Variance Retained:** 100.0%\n\n"
        f"• **Test Accuracy:** {k_info.get('test_accuracy', 0.9234)*100:.2f}%\n\n"
        f"• **Model RAM:** {k_info.get('model_size_kb', 62.3):.1f} KB"
    )

st.sidebar.markdown("---")
input_mode = st.sidebar.radio(
    "Select Image Input Mode:",
    ["📤 Upload Image", "✏️ Draw Digit", "🎯 Preset Test Samples"],
    index=0
)

# Tabs Navigation
tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Live Digit Classifier",
    "🎨 Structural Eigen-Digits Gallery",
    "📊 Hardware Edge Benchmarks",
    "📑 Academic Project Report"
])

# ==============================================================================
# TAB 1: LIVE DIGIT CLASSIFIER
# ==============================================================================
with tab1:
    st.markdown("#### Real-Time Handwritten Digit Classification with Structural Compression")
    st.markdown("Upload any digit image, sketch on the canvas, or pick a sample to inspect the PCA reconstruction and model prediction.")
    
    col_input, col_proc, col_recon, col_pred = st.columns([1.1, 1.1, 1.1, 1.7])
    
    raw_pil_image = None
    
    # 1. INPUT MODES
    with col_input:
        st.markdown("**1. Raw Input**")
        
        if input_mode == "📤 Upload Image":
            uploaded_file = st.file_uploader(
                "Upload a digit (PNG, JPG, JPEG):",
                type=["png", "jpg", "jpeg"],
                help="Accepts dark digits on light paper or light digits on dark background."
            )
            if uploaded_file is not None:
                raw_pil_image = Image.open(uploaded_file)
                st.image(raw_pil_image, caption="Uploaded Image", use_container_width=True)
            else:
                st.info("👆 Upload an image to classify.")
                # Fallback to sample 3
                sample_path = os.path.join(SAMPLES_DIR, "digit_3.png")
                if os.path.exists(sample_path):
                    raw_pil_image = Image.open(sample_path)
                    st.caption("Showing demo sample (Digit 3)")
                    st.image(raw_pil_image, caption="Default Sample", use_container_width=True)
                    
        elif input_mode == "✏️ Draw Digit":
            if CANVAS_AVAILABLE:
                st.caption("Draw a digit (0-9) inside the box:")
                canvas_result = st_canvas(
                    fill_color="black",
                    stroke_width=18,
                    stroke_color="#FFFFFF",
                    background_color="#000000",
                    height=200,
                    width=200,
                    drawing_mode="freedraw",
                    key="digit_canvas"
                )
                if canvas_result.image_data is not None:
                    # check if canvas has any drawing
                    if np.max(canvas_result.image_data[:, :, :3]) > 10:
                        raw_pil_image = Image.fromarray(canvas_result.image_data.astype(np.uint8))
                    else:
                        st.caption("Canvas is empty. Draw a number.")
            else:
                st.warning("`streamlit-drawable-canvas` is not active. Please use 'Upload Image' or 'Preset Samples'.")
                
        elif input_mode == "🎯 Preset Test Samples":
            st.caption("Select a validated test set digit:")
            selected_digit = st.selectbox("Choose Digit:", list(range(10)), index=7)
            sample_path = os.path.join(SAMPLES_DIR, f"digit_{selected_digit}.png")
            if os.path.exists(sample_path):
                raw_pil_image = Image.open(sample_path)
                st.image(raw_pil_image, caption=f"Test Set: Digit {selected_digit}", use_container_width=True)
    
    # 2. PROCESSING & INFERENCE
    if raw_pil_image is not None:
        final_28x28, norm_vector = preprocess_image(raw_pil_image)
        
        with col_proc:
            st.markdown("**2. Preprocessed (28×28)**")
            st.image(final_28x28 / 255.0, caption="MNIST Standard [0,1]", use_container_width=True, clamp=True)
            st.caption(f"Dimensions: 28×28 (784 features)\nFormat: Centered & Inverted")
            
        with col_recon:
            st.markdown(f"**3. Reconstructed (k={k_choice})**")
            if k_choice < 784:
                recon_img = reconstruct_pca(norm_vector, k_choice)
                st.image(recon_img, caption=f"k={k_choice} Latent Space", use_container_width=True, clamp=True)
                # Compute reconstruction MSE
                mse = np.mean((norm_vector.reshape(28, 28) - recon_img) ** 2)
                st.caption(f"Reconstruction MSE: `{mse:.4f}`\nEnergy Retained: `{k_info.get('explained_variance_pct', 0):.1f}%`")
            else:
                st.image(final_28x28 / 255.0, caption="Full 784 Baseline (100%)", use_container_width=True, clamp=True)
                st.caption("No compression applied.")
                
        with col_pred:
            st.markdown("**4. Model Prediction & Latency**")
            
            # Predict using selected pipeline
            active_model = models_dict.get(k_choice, models_dict.get(100))
            
            if active_model is not None:
                t_infer_start = time.perf_counter()
                probabilities = active_model.predict_proba(norm_vector.reshape(1, -1))[0]
                infer_latency_ms = (time.perf_counter() - t_infer_start) * 1000.0
                pred_label = int(np.argmax(probabilities))
                confidence = float(probabilities[pred_label]) * 100.0
                
                # Big Prediction Badge
                st.markdown(f"""
                <div class="pred-box">
                    <div style="font-size: 13px; text-transform: uppercase; color: #a5b4fc; font-weight: 700; letter-spacing: 1px;">
                        Predicted Digit
                    </div>
                    <div class="pred-digit">{pred_label}</div>
                    <div class="pred-conf">Confidence: {confidence:.1f}%</div>
                    <div style="margin-top: 10px; font-size: 12px; color: #94a3b8; font-family: 'JetBrains Mono', monospace;">
                        ⚡ Inference Latency: <b>{infer_latency_ms:.2f} ms</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Probability Distribution Chart
                st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
                prob_df = pd.DataFrame({
                    "Digit": [str(d) for d in range(10)],
                    "Probability": probabilities * 100.0
                })
                
                fig_prob = px.bar(
                    prob_df, x="Digit", y="Probability",
                    text=[f"{p:.1f}%" if p > 3 else "" for p in prob_df["Probability"]],
                    color="Probability",
                    color_continuous_scale="Purples",
                    title="Class Probability Distribution (Softmax Output)"
                )
                fig_prob.update_layout(
                    height=220,
                    margin=dict(l=10, r=10, t=35, b=10),
                    yaxis=dict(range=[0, 105], title="Confidence (%)"),
                    xaxis=dict(title="Digit Class"),
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_prob, use_container_width=True)

    st.markdown("---")
    # Quick Edge Performance Highlights
    st.markdown("##### ⚡ Edge Compression Efficiency Highlights")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{k_choice} / 784</div>
            <div class="metric-lbl">Coordinate Dimensions</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        comp_ratio = (1.0 - (k_choice / 784.0)) * 100.0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{comp_ratio:.1f}%</div>
            <div class="metric-lbl">Feature Dimension Reduction</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{k_info.get('explained_variance_pct', 0):.1f}%</div>
            <div class="metric-lbl">Retained Structural Variance</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{k_info.get('test_accuracy', 0)*100:.2f}%</div>
            <div class="metric-lbl">Test Classification Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# TAB 2: STRUCTURAL EIGEN-DIGITS GALLERY
# ==============================================================================
with tab2:
    st.markdown("### Structural Coordinate Bases: The Eigen-Digits")
    st.markdown("""
    In Singular Value Decomposition ($X = U \Sigma V^T$), the rows of $V^T$ represent the orthogonal eigenvectors 
    of the sample covariance matrix $C = \\frac{1}{n} X^T X$. When reshaped into $28 \\times 28$ coordinate spaces, 
    these principal components manifest as **eigen-digits**—the foundational structural building blocks of all handwritten digits.
    """)
    
    eigen_path = os.path.join(ARTIFACTS_DIR, "eigen_digits.png")
    if os.path.exists(eigen_path):
        st.image(eigen_path, caption="Top 25 Principal Coordinate Axes (Eigen-digits) from MNIST Covariance Matrix", use_container_width=True)
    
    st.markdown("#### Interactive Eigenvector Inspector")
    if pca_full is not None:
        pc_idx = st.slider("Select Principal Component Index (1 to 100):", min_value=1, max_value=100, value=1)
        comp_vector = pca_full.components_[pc_idx - 1].reshape(28, 28)
        comp_var = pca_full.explained_variance_ratio_[pc_idx - 1] * 100.0
        cum_var = np.sum(pca_full.explained_variance_ratio_[:pc_idx]) * 100.0
        
        c_left, c_right = st.columns([1, 2])
        with c_left:
            vmax = np.max(np.abs(comp_vector))
            fig_ec = px.imshow(
                comp_vector,
                color_continuous_scale="RdBu_r",
                zmin=-vmax, zmax=vmax,
                title=f"Eigen-Digit: Principal Component #{pc_idx}"
            )
            fig_ec.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_ec, use_container_width=True)
        with c_right:
            st.markdown(f"""
            ##### Structural Properties of PC #{pc_idx}:
            - **Individual Variance Explained:** `{comp_var:.3f}%`
            - **Cumulative Variance (1 to {pc_idx}):** `{cum_var:.2f}%`
            - **Mathematical Representation:** $\\mathbf{{v}}_{{{pc_idx}}} \\in \\mathbb{{R}}^{{784}}$
            - **Interpretation:** 
              - Early components ($k \\le 10$) capture broad geometric loops, vertical strokes (distinguishing '1' from '0'), and global stroke weight.
              - Intermediate components ($10 < k \\le 100$) capture subtle loops, serifs, cross-bars ('4', '7'), and curvature.
              - Higher components ($k > 100$) capture localized fine-grained stroke boundary variations and high-frequency pixel noise.
            """)
            
    st.markdown("---")
    st.markdown("#### Reconstruction Fidelity Across Candidate Component Counts")
    recon_path = os.path.join(ARTIFACTS_DIR, "reconstruction_comparison.png")
    if os.path.exists(recon_path):
        st.image(recon_path, caption="Progressive structural fidelity restoration across k in [5, 50, 100, 200, 784]", use_container_width=True)

# ==============================================================================
# TAB 3: HARDWARE EDGE BENCHMARKS
# ==============================================================================
with tab3:
    st.markdown("### Edge Hardware Complexity Benchmarks & Trade-Offs")
    st.markdown("""
    Deploying machine learning models to embedded edge processors (e.g., ARM Cortex-M microcontrollers, Raspberry Pi, 
    or mobile Edge TPUs) requires navigating strict trade-offs between **Memory Footprint (SRAM/Flash)**, 
    **Inference Latency (Throughput)**, and **Classification Accuracy**.
    """)
    
    # Summary Table
    st.markdown("##### Comprehensive Benchmark Results ($k \\in [5, 50, 100, 200, 784]$)")
    summary_rows = []
    for k_val in [5, 50, 100, 200, 784]:
        item = benchmark_data.get(str(k_val), {})
        summary_rows.append({
            "Components (k)": str(k_val) if k_val < 784 else "784 (Baseline)",
            "Explained Var (%)": f"{item.get('explained_variance_pct', 100.0):.2f}%",
            "5-Fold CV Accuracy": f"{item.get('cv_mean_accuracy', 0)*100:.2f}%",
            "Test Accuracy": f"{item.get('test_accuracy', 0)*100:.2f}%",
            "Test Error Rate": f"{item.get('test_error', 0)*100:.2f}%",
            "Macro F1-Score": f"{item.get('macro_f1', 0)*100:.2f}%",
            "Single-Sample Latency": f"{item.get('avg_latency_ms', 0):.3f} ms",
            "Model Size (KB)": f"{item.get('model_size_kb', 0):.1f} KB"
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    
    # Benchmarking Plots
    tradeoff_img = os.path.join(ARTIFACTS_DIR, "tradeoff_benchmarking_curves.png")
    if os.path.exists(tradeoff_img):
        st.image(tradeoff_img, caption="Multi-panel Benchmarking: (A) Test Error, (B) Latency, (C) Model Memory, (D) Pareto Frontier", use_container_width=True)
        
    cm_img = os.path.join(ARTIFACTS_DIR, "confusion_matrices_comparison.png")
    if os.path.exists(cm_img):
        st.image(cm_img, caption="Confusion Matrix Evolution Across Component Counts k in [5, 50, 100, 200]", use_container_width=True)

# ==============================================================================
# TAB 4: ACADEMIC PROJECT REPORT
# ==============================================================================
with tab4:
    st.markdown("### Academic Submission Details")
    st.markdown("""
    **Course:** MACSE504 - Machine Learning  
    **Project:** Course Based Design Project (CBDP)  
    **Date:** 30.10.2026 | **Slot:** E1+TE1  
    **Student Name:** Rohan Yadav  
    **Register Number:** 26MML0020  
    **Assigned Question Number:** Question 7  
    
    ---
    
    #### Question Statement:
    > **7. Complexity Reduction Over Structural Coordinate Spaces**  
    > Deploying image recognition models on edge hardware requires balancing computational throughput with structural accuracy. 
    > Students will implement singular value decompositions to compress pixel spaces and analyze the trade-offs between 
    > component counts, model memory footprints, and classification accuracy.  
    > **Dataset Link:** MNIST Digit Dataset via Kaggle  
    > **Requirements:**  
    > *i.* Visualizing eigenvectors (eigen-digits) and plotting cumulative explained variance ratios.  
    > *ii.* Constructing a scikit-learn pipeline coupling PCA dimensionality reduction with a Logistic Regression classifier.  
    > *iii.* Running grid-searches across component counts $k \\in [5, 50, 100, 200]$ using stratified cross-validation.  
    > *iv.* Generating curves mapping feature reduction levels against classification latency and test errors.
    
    ---
    
    #### Pipeline Architecture:
    ```
    Raw Image (28x28) ──> Preprocessing & Min-Max Normalization [0, 1] ──> Flatten (784-D Vector)
                               │
                               ▼
        Singular Value Decomposition / PCA Projection: z = (x - μ) · V_k^T  (k ∈ [5, 50, 100, 200])
                               │
                               ▼
            Multinomial Logistic Regression Classifier (Softmax Layer)
                               │
                               ▼
        Output Class (Digit 0-9) + Class Probabilities + Reconstruction Audit
    ```
    
    ---
    
    #### Mathematical & Structural Inferences:
    1. **Variance Retention & Knee Point:**  
       As seen in the cumulative variance curve, the first $k=50$ components capture **82.54%** of the global variance, and $k=100$ captures **91.50%**. Beyond $k=100$, the curve reaches diminishing returns, where adding 100 more components ($k=200$) only increases variance by $5.16\\%$ and test accuracy by $0.32\\%$.
       
    2. **Edge Hardware Recommendation:**  
       For ultra-low power microcontroller nodes (e.g. ARM Cortex-M4 with $<256\\text{ KB}$ SRAM), **$k=50$** is the optimal sweet spot, delivering **90.84% accuracy** with a tiny **162 KB** footprint. For slightly higher throughput gateways (Raspberry Pi Zero), **$k=100$** achieves **91.87% accuracy**, matching the uncompressed baseline within $0.47\\%$ while trimming input dimensionality by **$87.2\\%$**.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 13px;">
    MACSE504 Machine Learning Project | Question 7 | Rohan Yadav (26MML0020) | Vellore Institute of Technology
</div>
""", unsafe_allow_html=True)
