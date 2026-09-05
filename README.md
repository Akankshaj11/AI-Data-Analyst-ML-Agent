# AI Data Analysis & ML Agent

An end-to-end Streamlit application for CSV and Excel dataset analysis, data quality auditing, AI-assisted insight generation, baseline machine learning pipeline training, model download, prediction simulation, and Markdown/Word/PDF executive report generation.

---

## 🌐 Live Demo

🔗 **[Open Live Application on Streamlit Cloud](https://ai-data-analysis-ml-agent.streamlit.app)**

---

## Key Features

### 📊 Data Analysis & Quality Auditing
- Upload CSV or Excel `.xlsx` datasets (with custom sheet selection).
- Automatic column role detection (Numerical, Categorical, ID/Index).
- Missing value analysis, duplicate row detection, and data quality scoring (0–100).
- Actionable data cleaning recommendations.
- Exploratory Data Analysis: Descriptive statistics, numerical distribution histograms, categorical bar charts, and correlation heatmap with top correlation highlights.

### 🤖 Automated Baseline Machine Learning Pipeline
- Target column selection with automatic ML task type recommendation (Binary Classification, Multi-class Classification, Regression).
- Automated baseline model training and evaluation metric benchmarking.
- Model performance leaderboard comparison.
- Full Scikit-Learn Pipeline export (`.pkl`) including preprocessing (imputation, encoding, scaling) and trained model.

### 🎯 Single-Sample Live Prediction Simulator
- Interactive feature input generator powered by the trained best baseline pipeline.
- Real-time prediction generation with target values and class probability distribution metrics.

### 🧠 AI-Assisted Insights & Strategy
- Optional AI insight generator supporting OpenAI (`OPENAI_API_KEY`) and Google Gemini (`GEMINI_API_KEY`) with automatic provider fallback.
- Rule-based analysis works 100% offline without API keys.

### 📄 Executive Automated Report Export
- Multi-format report export: Markdown (`.md`), Microsoft Word (`.docx`), and Adobe PDF (`.pdf`).
- Formatted executive report includes dataset overview, missing value analysis, data quality audit, ML benchmarking results, AI strategic insights, and recommended next steps.

---

## How to Run Guidelines

Follow these step-by-step instructions to run the application locally on your machine.

### Prerequisites
- Python 3.9 or higher installed on your system.

### 1. Open Terminal / PowerShell & Navigate to Project Directory
```bash
cd "d:\AI RESUME PROJECTS\AI DATA ANALYST\AI_Data_Analysis_ML_Agent-main\AI_Data_Analysis_ML_Agent-main"
```

### 2. Create and Activate a Virtual Environment

- **On Windows (PowerShell)**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
  *(If PowerShell execution policy error occurs, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first)*

- **On Windows (Command Prompt - cmd)**:
  ```cmd
  python -m venv venv
  .\venv\Scripts\activate.bat
  ```

- **On macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Configure AI API Keys
AI features are optional. If you want to enable OpenAI or Gemini AI insights, create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```
*Note: If no API keys are provided, all data profiling, quality checks, ML model training, prediction demo, and report generation features continue to work 100% offline.*

### 5. Launch the Web Application
```bash
streamlit run app.py
```

The application will start the local server and automatically open `http://localhost:8501` in your web browser.

---

## Quick Demo Testing

You can instantly test the full workflow using the included sample customer churn dataset:
- File path: `examples/sample_churn.csv` (or click the **"🧪 Load Sample Customer Churn Dataset"** button in the app sidebar).
