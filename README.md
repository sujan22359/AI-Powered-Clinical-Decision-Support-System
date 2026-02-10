# 🏥 AI-Powered Multi-Modal Clinical Decision Support System

**Intelligent Medical Report Analysis with Threshold-Based Risk Assessment**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> AI-powered clinical report analysis system for informational and educational purposes.

**⚠️ DISCLAIMER:** This application is for informational purposes only and does not provide medical diagnosis or treatment recommendations. Always consult with qualified healthcare professionals for medical decisions.

---

## 📖 Table of Contents

- [Features](#-features)
- [Demo](#-demo)
- [Quick Start](#-quick-start)
- [Architecture](#-architecture)
- [Threshold-Based Risk Assessment](#-threshold-based-risk-assessment)
- [Project Structure](#-project-structure)
- [API Documentation](#-api-documentation)
- [Testing](#-testing)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Future Roadmap](#-future-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

---

## 🎬 Demo

### Screenshots

**Main Interface:**
![Main Interface](docs/screenshots/main-interface.png)

**Analysis Results:**
![Analysis Results](docs/screenshots/analysis-results.png)

**Risk Assessment:**
![Risk Assessment](docs/screenshots/risk-assessment.png)

> Note: Add screenshots to `docs/screenshots/` directory

### Sample Analysis

Upload a clinical lab report and get:
- ✅ Patient-friendly summary in plain language
- ✅ Key medical findings extracted automatically
- ✅ Threshold-based risk assessment with specific values
- ✅ Risk indicators categorized by severity (LOW/MEDIUM/HIGH)
- ✅ Follow-up suggestions and recommendations

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your Google Gemini API key (already configured)
```

### 3. Start the Application

**Option A: Use startup scripts (Recommended)**
```bash
# Terminal 1 - Start backend
python start_backend.py

# Terminal 2 - Start frontend  
python start_frontend.py
```

**Option B: Manual startup**
```bash
# Terminal 1 - Start backend
uvicorn backend.main:app --reload --host localhost --port 8000

# Terminal 2 - Start frontend
streamlit run frontend/app.py --server.port 8501
```

### 4. Access the Application
- **Frontend UI**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

## 📋 Features

- **Document Upload**: Support for PDF and DOCX clinical reports
- **AI Analysis**: Google Gemini-powered medical text analysis
- **Threshold-Based Risk Assessment**: Automated evaluation of clinical parameters against reference ranges
  - Extracts numerical values from lab reports (glucose, cholesterol, blood pressure, etc.)
  - Compares against evidence-based reference ranges
  - Categorizes risk levels (LOW, MEDIUM, HIGH) based on deviation from normal
  - Supports international unit conversion (mg/dL ↔ mmol/L, g/dL ↔ g/L, etc.)
  - Generates patient-friendly explanations with specific values and percentages
- **Hybrid Risk Detection**: Combines AI-detected risks with threshold-based analysis
- **Structured Output**: Patient-friendly summaries, key findings, risk indicators
- **Medical Safety**: Built-in disclaimers and safety filters
- **Error Handling**: Comprehensive error messages and suggestions
- **Real-time Processing**: Live analysis with progress indicators

## 🏗️ Architecture

```
Frontend (Streamlit) → API (FastAPI) → Analysis Engine → Google Gemini API
                                    ↓
                              Document Parser
                                    ↓
                         Threshold-Based Pipeline:
                         - Parameter Extractor
                         - Reference Range Database
                         - Threshold Evaluator
                         - Risk Explainer
```

### Threshold-Based Risk Assessment Pipeline

The system uses a sophisticated multi-stage pipeline for automated clinical parameter evaluation:

1. **Parameter Extraction**: Identifies clinical parameters (glucose, cholesterol, etc.) and their values from text
2. **Reference Range Lookup**: Retrieves evidence-based normal ranges from built-in database
3. **Unit Conversion**: Automatically converts between measurement systems (US ↔ SI units)
4. **Threshold Evaluation**: Compares values against ranges and categorizes risk levels
5. **Risk Explanation**: Generates patient-friendly explanations with specific details
6. **Risk Merging**: Combines threshold-based and AI-detected risks, removing duplicates

## 📁 Project Structure

```
clinical-report-analyzer/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── config.py              # Configuration management
│   ├── models/                # Data models
│   │   └── clinical_parameters.py  # Parameter and risk models
│   ├── services/              # Core services
│   │   ├── document_parser.py      # PDF/DOCX parsing
│   │   ├── llm_service.py          # Google Gemini integration
│   │   ├── analysis_engine.py      # Analysis orchestration
│   │   ├── parameter_extractor.py  # Clinical parameter extraction
│   │   ├── reference_range_db.py   # Reference range database
│   │   ├── threshold_evaluator.py  # Threshold-based evaluation
│   │   └── risk_explainer.py       # Patient-friendly explanations
│   ├── utils/                 # Utilities
│   │   ├── logger.py          # Logging setup
│   │   └── error_handlers.py  # Error handling
│   └── tests/                 # Test suite
│       ├── fixtures/          # Test clinical documents
│       ├── test_parameter_extractor.py
│       ├── test_threshold_evaluator.py
│       ├── test_risk_explainer.py
│       └── ...
├── frontend/
│   └── app.py                 # Streamlit application
├── requirements.txt           # Python dependencies
├── .env                       # Environment configuration
├── start_backend.py          # Backend startup script
└── start_frontend.py         # Frontend startup script
```

## 🔧 API Endpoints

- `GET /health` - Health check
- `GET /status` - Detailed system status
- `POST /analyze` - Analyze clinical document
- `GET /error-codes` - Error code documentation

## 🎯 Threshold-Based Risk Assessment

### Overview

The Clinical Report Analyzer includes an advanced threshold-based risk assessment system that automatically evaluates clinical parameters against evidence-based reference ranges. This provides objective, quantitative risk assessment alongside AI-powered analysis.

### Supported Parameters

The system includes reference ranges for 20+ common clinical parameters:

**Glucose Metabolism:**
- Blood Glucose (Fasting)
- Hemoglobin A1c

**Lipid Panel:**
- Total Cholesterol
- LDL Cholesterol
- HDL Cholesterol
- Triglycerides

**Kidney Function:**
- Creatinine
- Blood Urea Nitrogen (BUN)
- Estimated GFR (eGFR)

**Complete Blood Count:**
- Hemoglobin
- Hematocrit
- White Blood Cell Count
- Platelet Count

**Vital Signs:**
- Blood Pressure (Systolic/Diastolic)
- Heart Rate

**Liver Function:**
- ALT, AST, Bilirubin

**Thyroid Function:**
- TSH

**And more...**

### Unit Conversion Support

The system automatically handles unit conversions between US and international (SI) units:

- **Glucose:** mg/dL ↔ mmol/L
- **Cholesterol:** mg/dL ↔ mmol/L
- **Creatinine:** mg/dL ↔ μmol/L
- **Hemoglobin:** g/dL ↔ g/L

### Risk Categorization

Parameters are categorized into three risk levels:

- **LOW (Green):** Value within normal range
- **MEDIUM (Yellow):** Value outside normal range but not critical (requires attention)
- **HIGH (Red):** Value critically abnormal (requires immediate attention)

**Critical Thresholds:**
- HIGH risk: >50% deviation from normal range boundaries
- MEDIUM risk: Outside normal range but <50% deviation

### Example Output

For a patient with elevated glucose:

```
🟡 MEDIUM RISK (Metabolic) 📊 Threshold-Based
Blood Glucose: 110.0 mg/dL (Normal: 70.0-100.0 mg/dL)

Parameter: Blood Glucose
Your Value: 110.0 mg/dL
Normal Range: 70.0-100.0 mg/dL
Deviation: 10.0% above normal

Details: Your Blood Glucose level is 110.0 mg/dL, which is above the 
normal range. Your value is approximately 10.0% above the normal threshold. 
While not critical, this should be discussed with your healthcare provider 
to determine if any action is needed.
```

### Benefits

1. **Objective Assessment:** Quantitative evaluation based on established medical standards
2. **Detailed Information:** Specific values, ranges, and deviation percentages
3. **Patient-Friendly:** Clear explanations without medical jargon
4. **Comprehensive:** Covers multiple body systems and parameter types
5. **Accurate:** Handles unit conversions and edge cases properly
6. **Complementary:** Works alongside AI analysis for comprehensive insights

## ⚠️ Important Notice

This application is for **informational purposes only** and does not provide medical diagnosis or treatment recommendations. Always consult with qualified healthcare professionals for medical decisions.

## 🧪 Testing

```bash
# Run all tests
python -m pytest backend/tests/ -v

# Run specific test categories
python -m pytest backend/tests/test_document_parser.py -v
python -m pytest backend/tests/test_llm_service.py -v
python -m pytest backend/tests/test_analysis_engine.py -v
```

## 🔍 Demo Scripts

```bash
# Test document parsing
python backend/demo_document_parser.py

# Test LLM service
python backend/demo_llm_service.py

# Test analysis engine
python backend/demo_analysis_engine.py
```

## 🛠️ Configuration

Key environment variables in `.env`:
- `GEMINI_API_KEY`: Your Google Gemini API key (required)
- `API_HOST`: API host (default: localhost)
- `API_PORT`: API port (default: 8000)
- `LOG_LEVEL`: Logging level (default: INFO)

## 📊 Supported File Formats

- **PDF**: Clinical reports, lab results, medical records
- **DOCX**: Word documents with clinical content
- **File Size Limit**: 10MB maximum

## 🔒 Security & Privacy

- No data is stored permanently
- Files are processed in memory only
- API keys are managed through environment variables
- Medical content is filtered for safety compliance

## 🐛 Troubleshooting

**Backend won't start:**
- Check if GEMINI_API_KEY is set in .env
- Verify port 8000 is available
- Check Python dependencies are installed

**Frontend can't connect:**
- Ensure backend is running on localhost:8000
- Check firewall settings
- Verify CORS configuration

**Analysis fails:**
- Check Google Gemini API key validity
- Verify document format (PDF/DOCX only)
- Check file size (max 10MB)
- Review error messages for specific guidance

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Google Gemini API for AI-powered analysis
- FastAPI for robust backend framework
- Streamlit for intuitive frontend
- Medical community for reference ranges and standards

---

## 📞 Support

For issues or questions:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Review the [API Documentation](#-api-documentation)
3. Open an issue on GitHub
4. Check existing issues for solutions

---



---

**⚠️ Medical Disclaimer:** This software is provided for informational and educational purposes only. It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
