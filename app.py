import os
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from modules.ai_insights import (
    generate_ai_insights,
    get_gemini_api_key,
    get_openai_api_key,
)
from modules.column_detection import detect_column_types
from modules.correlation import analyze_correlations
from modules.data_loader import load_dataset, prepare_data
from modules.data_quality import calculate_data_quality_score
from modules.ml_task import suggest_ml_task
from modules.ml_trainer import create_model_download_bytes, train_baseline_model
from modules.prediction import create_prediction_input, make_single_prediction
from modules.report_export import create_docx_report_bytes, create_pdf_report_bytes
from modules.report_generator import generate_markdown_report
from modules.workflow import get_agent_workflow_steps

# Set page configuration
st.set_page_config(
    page_title="AI Data Analysis & ML Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Matplotlib Dark Styling (Compact)
def setup_dark_matplotlib():
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': '#111827',
        'axes.facecolor': '#1F2937',
        'axes.edgecolor': '#374151',
        'grid.color': '#374151',
        'text.color': '#F9FAFB',
        'axes.labelcolor': '#E5E7EB',
        'xtick.color': '#9CA3AF',
        'ytick.color': '#9CA3AF',
        'font.sans-serif': 'Plus Jakarta Sans, Inter, sans-serif'
    })

setup_dark_matplotlib()

# Inject Modern Glassmorphism & Cyber Dark Custom CSS with Scaled-Down Typography
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 13.5px;
    }

    /* Main background */
    .stApp {
        background: #0B0F19;
        color: #E5E7EB;
    }

    /* Global Heading & Typography Scaling */
    h1 { font-size: 1.5rem !important; font-weight: 700 !important; color: #F9FAFB !important; margin-bottom: 8px !important; }
    h2 { font-size: 1.2rem !important; font-weight: 700 !important; color: #F3F4F6 !important; margin-bottom: 6px !important; }
    h3 { font-size: 1.02rem !important; font-weight: 700 !important; color: #E5E7EB !important; margin-top: 10px !important; margin-bottom: 6px !important; }
    h4 { font-size: 0.9rem !important; font-weight: 600 !important; color: #D1D5DB !important; margin-top: 6px !important; margin-bottom: 4px !important; }
    p, li, label, div { font-size: 0.85rem !important; line-height: 1.45 !important; }
    
    .stMarkdown, .stMarkdown p {
        font-size: 0.85rem !important;
        line-height: 1.45 !important;
    }

    /* Compact Hero Banner */
    .hero-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(6, 182, 212, 0.12) 50%, rgba(16, 185, 129, 0.08) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 18px;
        backdrop-filter: blur(16px);
    }

    .hero-title {
        font-size: 1.7rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6366F1 0%, #06B6D4 50%, #10B981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }

    .hero-subtitle {
        font-size: 0.85rem;
        color: #9CA3AF;
        margin-bottom: 8px;
        line-height: 1.4;
    }

    /* Custom Badges */
    .badge-chip {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 9999px;
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        background: rgba(99, 102, 241, 0.18);
        color: #818CF8;
        border: 1px solid rgba(99, 102, 241, 0.35);
        margin-right: 6px;
    }

    .badge-cyan {
        background: rgba(6, 182, 212, 0.18);
        color: #22D3EE;
        border: 1px solid rgba(6, 182, 212, 0.35);
    }

    .badge-emerald {
        background: rgba(16, 185, 129, 0.18);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }

    /* Metric Card Component */
    .stat-card {
        background: rgba(17, 24, 39, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 12px;
        text-align: center;
        backdrop-filter: blur(12px);
    }

    .stat-number {
        font-size: 1.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #06B6D4 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .stat-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 2px;
    }

    /* Custom Glass Box Container */
    .glass-box {
        background: rgba(17, 24, 39, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 16px;
        backdrop-filter: blur(12px);
        font-size: 0.85rem !important;
    }

    /* Scaled-Down Typography inside Report & AI Insight Markdown */
    .glass-box h1 { font-size: 1.25rem !important; margin-top: 12px !important; margin-bottom: 6px !important; }
    .glass-box h2 { font-size: 1.05rem !important; margin-top: 10px !important; margin-bottom: 4px !important; }
    .glass-box h3 { font-size: 0.92rem !important; margin-top: 8px !important; margin-bottom: 4px !important; }
    .glass-box h4 { font-size: 0.85rem !important; margin-top: 6px !important; margin-bottom: 2px !important; }
    .glass-box p, .glass-box li { font-size: 0.82rem !important; line-height: 1.4 !important; }

    /* Streamlit Sidebar Custom Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* Styled Streamlit Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(17, 24, 39, 0.8);
        padding: 6px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 16px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 36px;
        border-radius: 8px;
        color: #9CA3AF;
        font-weight: 600;
        font-size: 0.8rem !important;
        padding: 0px 14px;
        border: none;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px 0 rgba(99, 102, 241, 0.35);
    }

    /* Primary Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 6px 16px;
        font-size: 0.82rem !important;
        font-weight: 600;
        box-shadow: 0 3px 10px rgba(99, 102, 241, 0.3);
    }

    /* Download Buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #06B6D4 0%, #0891B2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 6px 16px;
        font-size: 0.82rem !important;
        font-weight: 600;
        box-shadow: 0 3px 10px rgba(6, 182, 212, 0.3);
    }

    /* Dataframe styling */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        font-size: 0.8rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Hero Header Component
st.markdown("""
<div class="hero-card">
    <div style="margin-bottom: 8px;">
        <span class="badge-chip">⚡ Autonomous AI Agent</span>
        <span class="badge-chip badge-cyan">📊 ML Pipeline Engine</span>
        <span class="badge-chip badge-emerald">📄 Exec Report Generator</span>
    </div>
    <div class="hero-title">AI Data Analyst & ML Agent</div>
    <div class="hero-subtitle">
        An end-to-end intelligent agent for dataset profiling, data quality auditing, correlation analysis, 
        automated machine learning training, model export, and instant report generation.
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Design
with st.sidebar:
    st.markdown("### ⚙️ Dataset & Settings")
    
    uploaded_file = st.file_uploader(
        "Upload dataset (.csv, .xlsx)",
        type=["csv", "xlsx"],
        help="Upload any structured tabular dataset to trigger automated AI analysis."
    )

    # Option to load sample dataset
    sample_file_path = os.path.join(os.path.dirname(__file__), "examples", "sample_churn.csv")
    use_sample = False
    if uploaded_file is None and os.path.exists(sample_file_path):
        st.markdown("---")
        st.markdown("**💡 Or try a demo dataset:**")
        if st.button("🧪 Load Sample Customer Churn Dataset"):
            use_sample = True

    # Check API keys
    openai_key = get_openai_api_key()
    gemini_key = get_gemini_api_key()
    
    st.markdown("---")
    st.markdown("### 🔑 AI API Status")
    col_api1, col_api2 = st.columns(2)
    with col_api1:
        if openai_key:
            st.markdown("🟢 **OpenAI**: Ready")
        else:
            st.markdown("🔴 **OpenAI**: Off")
    with col_api2:
        if gemini_key:
            st.markdown("🟢 **Gemini**: Ready")
        else:
            st.markdown("🔴 **Gemini**: Off")
    
    if not (openai_key or gemini_key):
        st.caption("AI insights are optional. Rule-based analysis, ML baseline training, and reports work 100% offline.")

    st.markdown("---")
    st.markdown("### 🤖 Agent Architecture")
    with st.expander("Explore Tool-Based Agent Steps"):
        for step_info in get_agent_workflow_steps():
            st.markdown(f"**{step_info['step']}**")
            st.caption(f"🔧 **Tool:** {step_info['tool']}")
            st.caption(f"📌 **Output:** {step_info['output']}")
            st.markdown("---")

# Determine active file source
active_file = uploaded_file
if active_file is None and use_sample:
    active_file = sample_file_path

if active_file is not None:
    try:
        df, file_type, sheet_name = load_dataset(active_file)
        df = prepare_data(df)

        file_name = active_file.name if hasattr(active_file, 'name') else "sample_churn.csv"
        file_signature = f"{file_name}-{df.shape}-{list(df.columns)}"
        if st.session_state.get("ai_file_signature") != file_signature:
            st.session_state["ai_file_signature"] = file_signature
            st.session_state["ai_insights"] = None
            st.session_state["model_results"] = None

        # Data Profiling & Column Type Detection
        numeric_cols, categorical_cols, id_cols = detect_column_types(df)
        duplicate_count = df.duplicated().sum()
        data_quality_score, data_quality_suggestions = calculate_data_quality_score(df)

        # Tabs Navigation UI
        tab_overview, tab_eda, tab_ml, tab_predict, tab_ai, tab_report = st.tabs([
            "📋 Overview & Quality",
            "📊 EDA & Visuals",
            "🤖 ML Engine",
            "🎯 Prediction Simulator",
            "🧠 AI Insights",
            "📄 Report & Downloads"
        ])

        # ==========================================
        # TAB 1: OVERVIEW & DATA QUALITY
        # ==========================================
        with tab_overview:
            st.markdown("### 📊 Dataset High-Level Metrics")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{df.shape[0]:,}</div>
                    <div class="stat-label">Total Rows</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{df.shape[1]}</div>
                    <div class="stat-label">Total Columns</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number" style="background: linear-gradient(135deg, #F59E0B 0%, #EF4444 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{duplicate_count}</div>
                    <div class="stat-label">Duplicates</div>
                </div>
                """, unsafe_allow_html=True)
            with m4:
                quality_color = "linear-gradient(135deg, #10B981 0%, #059669 100%)" if data_quality_score >= 80 else "linear-gradient(135deg, #F59E0B 0%, #D97706 100%)"
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number" style="background: {quality_color}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{data_quality_score}/100</div>
                    <div class="stat-label">Data Quality Score</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            col_left, col_right = st.columns([3, 2])
            with col_left:
                st.markdown("#### 🔍 Dataset Preview")
                st.dataframe(df.head(10), use_container_width=True)

            with col_right:
                st.markdown("#### 🛡️ Data Quality Recommendations")
                for suggestion in data_quality_suggestions:
                    if data_quality_score >= 80:
                        st.success(f"💡 {suggestion}")
                    else:
                        st.warning(f"⚠️ {suggestion}")

            st.markdown("---")
            st.markdown("#### 📋 Detailed Column Schema & Types")
            column_info = pd.DataFrame({
                "Column Name": df.columns,
                "Data Type": df.dtypes.astype(str),
                "Detected Role": [
                    "Numerical" if c in numeric_cols else ("Categorical" if c in categorical_cols else "ID / Index")
                    for c in df.columns
                ],
                "Missing Values": df.isnull().sum().values,
                "Missing (%)": (df.isnull().sum().values / len(df) * 100).round(2),
                "Unique Values": df.nunique().values
            })
            st.dataframe(column_info, use_container_width=True)

        # ==========================================
        # TAB 2: EDA & VISUALIZATIONS (COMPACT CHARTS)
        # ==========================================
        with tab_eda:
            st.markdown("### 📈 Exploratory Data Analysis")
            
            if len(numeric_cols) > 0:
                st.markdown("#### 🔢 Numerical Feature Descriptive Statistics")
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)

            st.markdown("---")
            col_corr, col_dist = st.columns([1, 1])

            with col_corr:
                st.markdown("#### 🔗 Correlation Matrix & Heatmap")
                correlation_summary = analyze_correlations(df, numeric_cols)
                if correlation_summary is None:
                    st.info("Correlation analysis requires at least two suitable numerical columns.")
                else:
                    corr_matrix = correlation_summary["corr_matrix"]
                    # Compact Heatmap Figure Size
                    fig, ax = plt.subplots(figsize=(4.0, 2.8))
                    heatmap = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
                    ax.set_xticks(range(len(corr_matrix.columns)))
                    ax.set_yticks(range(len(corr_matrix.columns)))
                    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right", fontsize=7.5)
                    ax.set_yticklabels(corr_matrix.columns, fontsize=7.5)
                    ax.set_title("Correlation Heatmap", fontsize=9, fontweight="bold", pad=8)
                    cbar = fig.colorbar(heatmap, ax=ax, shrink=0.75)
                    cbar.ax.tick_params(labelsize=7)
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=False)

                    if len(correlation_summary["top_correlations"]) > 0:
                        st.markdown("**Top Strongest Correlations:**")
                        for item in correlation_summary["top_correlations"]:
                            st.caption(f"• **{item['column_a']}** ↔ **{item['column_b']}**: `{item['correlation']:.3f}`")

            with col_dist:
                st.markdown("#### 📊 Interactive Feature Distribution")
                if len(numeric_cols) > 0:
                    selected_num_col = st.selectbox("Select Numerical Feature", numeric_cols, key="eda_num")
                    # Compact Histogram Figure Size
                    fig, ax = plt.subplots(figsize=(3.8, 2.4))
                    ax.hist(df[selected_num_col].dropna(), bins=20, color="#06B6D4", edgecolor="#0891B2", alpha=0.85)
                    ax.set_title(f"Distribution of {selected_num_col}", fontsize=9, fontweight="bold", pad=8)
                    ax.set_xlabel(selected_num_col, fontsize=7.5)
                    ax.set_ylabel("Frequency", fontsize=7.5)
                    ax.tick_params(labelsize=7)
                    ax.grid(True, linestyle="--", alpha=0.2)
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=False)
                
                if len(categorical_cols) > 0:
                    st.markdown("---")
                    selected_cat_col = st.selectbox("Select Categorical Feature", categorical_cols, key="eda_cat")
                    val_counts = df[selected_cat_col].value_counts().head(8)
                    # Compact Bar Chart Figure Size
                    fig, ax = plt.subplots(figsize=(3.8, 2.4))
                    ax.bar(val_counts.index.astype(str), val_counts.values, color="#6366F1", alpha=0.85, edgecolor="#4F46E5")
                    ax.set_title(f"Top Categories in {selected_cat_col}", fontsize=9, fontweight="bold", pad=8)
                    ax.set_xlabel(selected_cat_col, fontsize=7.5)
                    ax.set_ylabel("Count", fontsize=7.5)
                    plt.xticks(rotation=35, ha="right", fontsize=7)
                    ax.tick_params(labelsize=7)
                    ax.grid(True, linestyle="--", alpha=0.2)
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=False)

        # ==========================================
        # TAB 3: ML ENGINE & BASELINE TRAINING
        # ==========================================
        with tab_ml:
            st.markdown("### 🤖 Automated Baseline Machine Learning Pipeline")
            
            target_options = [col for col in df.columns if col not in id_cols]
            selected_target = None
            task_info = None

            if len(target_options) == 0:
                st.info("No suitable target column found in this dataset.")
            else:
                default_index = target_options.index("Churn") if "Churn" in target_options else 0
                
                c_target, c_space = st.columns([2, 1])
                with c_target:
                    selected_target = st.selectbox(
                        "🎯 Target Column for Prediction",
                        target_options,
                        index=default_index,
                        help="Select the column you want the machine learning model to predict."
                    )

                task_info = suggest_ml_task(df, selected_target, numeric_cols, categorical_cols, id_cols)

                st.markdown(f"""
                <div class="glass-box" style="border-left: 4px solid #6366F1;">
                    <div style="font-size: 0.98rem; font-weight: 700; color: #818CF8; margin-bottom: 4px;">
                        Recommended Task: {task_info['task_type']}
                    </div>
                    <div style="color: #D1D5DB; font-size: 0.85rem; margin-bottom: 8px;">{task_info['reason']}</div>
                    <div style="font-size: 0.78rem; color: #9CA3AF;">
                        <b>Target Unique Values:</b> {task_info['target_unique']} | 
                        <b>Numerical Features:</b> {len(task_info['numeric_features'])} | 
                        <b>Categorical Features:</b> {len(task_info['categorical_features'])}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### ⚡ Train & Benchmark Baseline Models")
                if st.button("🚀 Train Baseline ML Pipeline", key="train_btn"):
                    with st.spinner("Training preprocessing & machine learning baseline pipelines..."):
                        try:
                            model_results = train_baseline_model(
                                df, selected_target, task_info, numeric_cols, categorical_cols, id_cols
                            )
                            st.session_state["model_results"] = model_results
                            st.success("✨ Baseline ML Training Completed Successfully!")
                        except Exception as e:
                            st.session_state["model_results"] = None
                            st.error(f"Training failed: {e}")

                model_results = st.session_state.get("model_results")
                current_model_results = None

                if model_results and model_results["target_column"] == selected_target:
                    current_model_results = model_results
                    st.markdown("---")
                    st.markdown("#### 🏆 Model Performance Leaderboard")
                    st.dataframe(model_results["comparison_table"], use_container_width=True)

                    st.info(f"🏆 **Best Performing Model:** `{model_results['best_model_name']}`\n\n{model_results['interpretation']}")

                    if model_results["task_type"] in ["Binary Classification", "Multi-class Classification"]:
                        confusion_info = model_results["confusion_matrix"]
                        confusion_df = pd.DataFrame(
                            confusion_info["matrix"],
                            index=confusion_info["labels"],
                            columns=confusion_info["labels"]
                        )
                        st.markdown("**Confusion Matrix (Best Model):**")
                        st.dataframe(confusion_df, use_container_width=True)

                    if "best_pipeline" in current_model_results:
                        st.markdown("#### 📦 Export Scikit-Learn Pipeline Package")
                        st.caption("Includes full preprocessing transformers (imputers, encoders, scalers) and trained model.")
                        st.download_button(
                            label="💾 Download Trained Model Pipeline (.pkl)",
                            data=create_model_download_bytes(current_model_results),
                            file_name=current_model_results["model_file_name"],
                            mime="application/octet-stream"
                        )
                elif model_results:
                    st.warning("Previous training results belong to a different target column. Re-run training above.")

        # ==========================================
        # TAB 4: PREDICTION SIMULATOR
        # ==========================================
        with tab_predict:
            st.markdown("### 🎯 Single-Sample Live Prediction Simulator")
            
            model_results = st.session_state.get("model_results")
            if model_results and "best_pipeline" in model_results and (selected_target is None or model_results["target_column"] == selected_target):
                st.caption(f"Using trained best baseline pipeline: **{model_results['best_model_name']}**")
                
                prediction_input_df = create_prediction_input(df, model_results)
                
                if st.button("🔮 Run Prediction Simulation", key="predict_btn"):
                    try:
                        prediction, probabilities = make_single_prediction(model_results, prediction_input_df)
                        
                        st.markdown(f"""
                        <div class="glass-box" style="text-align: center; border-color: rgba(6, 182, 212, 0.4); max-width: 400px; margin: 0 auto 16px auto;">
                            <div class="stat-label">Predicted Target Value</div>
                            <div class="stat-number" style="font-size: 1.8rem; margin-top: 4px;">{prediction}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        if probabilities is not None:
                            st.markdown("#### 📊 Prediction Probabilities")
                            st.dataframe(probabilities, use_container_width=True)
                    except Exception as pred_err:
                        st.error(f"Prediction error: {pred_err}")
            else:
                st.info("💡 Train a baseline model in the **🤖 ML Engine** tab first to unlock the interactive prediction simulator.")

        # ==========================================
        # TAB 5: AI AGENT INSIGHTS
        # ==========================================
        with tab_ai:
            st.markdown("### 🧠 AI-Assisted Insights & Strategy")
            
            data_quality_summary = {
                "score": data_quality_score,
                "suggestions": data_quality_suggestions,
                "duplicate_rows": int(duplicate_count),
                "missing_values_by_column": {
                    col: int(val) for col, val in df.isnull().sum().items() if val > 0
                }
            }

            ml_task_summary = None
            if task_info is not None and selected_target is not None:
                ml_task_summary = {
                    "selected_target": selected_target,
                    "task_type": task_info["task_type"],
                    "reason": task_info["reason"],
                    "suggested_models": task_info["suggested_models"],
                    "suggested_metrics": task_info["suggested_metrics"]
                }

            api_key_available = bool(openai_key or gemini_key)

            if not api_key_available:
                st.warning("⚠️ No OpenAI or Gemini API keys found. Add `OPENAI_API_KEY` or `GEMINI_API_KEY` to your environment to generate AI insights.")
            
            if st.button("✨ Generate AI Insights with Agent", disabled=not api_key_available, key="ai_btn"):
                with st.spinner("Analyzing dataset summary and prompting LLM agent..."):
                    ai_insights, ai_error, provider_name = generate_ai_insights(
                        df, column_info, data_quality_summary, ml_task_summary
                    )
                
                if ai_error:
                    st.error(ai_error)
                else:
                    st.session_state["ai_insights"] = ai_insights
                    st.success(f"AI insights generated via {provider_name}!")

            if st.session_state.get("ai_insights"):
                st.markdown("""<div class="glass-box">""", unsafe_allow_html=True)
                st.markdown(st.session_state["ai_insights"])
                st.markdown("""</div>""", unsafe_allow_html=True)

        # ==========================================
        # TAB 6: REPORT & DOWNLOADS
        # ==========================================
        with tab_report:
            st.markdown("### 📄 Executive Automated Report Generator")
            
            current_model_results = st.session_state.get("model_results")
            if current_model_results and current_model_results.get("target_column") != selected_target:
                current_model_results = None

            report = generate_markdown_report(
                df,
                data_quality_score=data_quality_score,
                data_quality_suggestions=data_quality_suggestions,
                selected_target=selected_target,
                task_info=task_info,
                correlation_summary=correlation_summary if 'correlation_summary' in locals() else None,
                ai_insights=st.session_state.get("ai_insights"),
                model_results=current_model_results
            )

            st.markdown("#### 📥 Export Formatted Executive Reports")
            col_dl1, col_dl2, col_dl3 = st.columns(3)
            with col_dl1:
                st.download_button(
                    label="📄 Download Markdown (.md)",
                    data=report,
                    file_name="ai_data_analysis_report.md",
                    mime="text/markdown",
                    key="dl_md"
                )
            with col_dl2:
                st.download_button(
                    label="📝 Download Word (.docx)",
                    data=create_docx_report_bytes(report),
                    file_name="ai_data_analysis_report.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key="dl_docx"
                )
            with col_dl3:
                st.download_button(
                    label="📕 Download PDF Report",
                    data=create_pdf_report_bytes(report),
                    file_name="ai_data_analysis_report.pdf",
                    mime="application/pdf",
                    key="dl_pdf"
                )

            st.markdown("---")
            st.markdown("#### 👁️ Report Live Preview")
            st.markdown(f"""<div class="glass-box">{report}</div>""", unsafe_allow_html=True)

    except Exception as e:
        st.error("⚠️ An error occurred while processing the dataset.")
        st.exception(e)
else:
    st.markdown("""
    <div class="glass-box" style="text-align: center; padding: 36px 20px;">
        <div style="font-size: 2.5rem; margin-bottom: 10px;">📁</div>
        <div style="font-size: 1.2rem; font-weight: 700; color: #F3F4F6; margin-bottom: 6px;">
            Upload your CSV or Excel dataset to start
        </div>
        <div style="color: #9CA3AF; max-width: 480px; margin: 0 auto 20px auto; line-height: 1.5; font-size: 0.85rem;">
            Select a file using the sidebar on the left, or test the application instantly with our pre-loaded Customer Churn demo dataset.
        </div>
    </div>
    """, unsafe_allow_html=True)
