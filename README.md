# ⚕️ MediVision AI - Clinical Decision Support System

**AI-Powered Multi-Modal Medical Analysis Platform**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red.svg)](https://streamlit.io/)
[![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-orange.svg)](https://ai.google.dev/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Overview

MediVision AI is an advanced AI-powered clinical decision support system designed to assist healthcare professionals and patients in analyzing medical reports, imaging studies, and laboratory results. Built with Google Gemini 2.5 Flash and PyTorch, it combines state-of-the-art artificial intelligence with medical domain expertise to provide accurate, actionable insights from clinical data.

### Key Features

- 📄 **Lab Report Analysis** - Automated analysis of clinical lab reports (PDF/DOCX) with AI-powered insights
- 🔬 **Medical Image Analysis** - Specialized medical imaging interpretation from X-rays, CT scans, MRI, and ultrasound
- 🔗 **Multi-Modal Analysis** - Integrated analysis combining lab reports and medical images with correlation detection
- 🩸 **Blood Group Prediction** - Custom PyTorch CNN model for blood group prediction from fingerprints
- ⚠️ **Risk Assessment** - Dual analysis pipeline (AI + threshold-based) with color-coded severity indicators
- 🎯 **Medical Specialization** - Purpose-built for clinical analysis with 15-year expert persona prompts
- 📊 **Structured Output** - Standardized JSON format with organized findings and recommendations

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Google Gemini API key ([Get one here](https://ai.google.dev/))

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd HealthCare_analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**
   
   Create `.env` file from example:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` file and add your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   GEMINI_MODEL=gemini-2.5-flash
   GEMINI_VISION_MODEL=gemini-2.5-flash
   API_HOST=localhost
   API_PORT=8000
   LOG_LEVEL=INFO
   MAX_FILE_SIZE_MB=10
   ```

4. **Start Application** (Single Command!)
   ```bash
   python run.py
   ```
   
   This will:
   - ✅ Start backend API on http://localhost:8000
   - ✅ Start frontend UI on http://localhost:8501
   - ✅ Validate configuration
   
   Access the application at: **http://localhost:8501**

### Alternative: Manual Start

**Backend:**
```bash
uvicorn backend.main:app --host localhost --port 8000 --reload
```

**Frontend:**
```bash
streamlit run frontend/app.py
```

---

## 📁 Project Structure

```
HealthCare_analysis/
├── backend/                    # Backend API (FastAPI)
│   ├── config.py              # Configuration settings
│   ├── main.py                # Main API application
│   ├── models/                # Data models & ML models
│   │   ├── clinical_parameters.py
│   │   └── ml_models/         # PyTorch CNN models
│   │       └── best_basic_cnn_with_residual.pth
│   ├── services/              # Business logic services
│   │   ├── analysis_engine.py      # Main analysis orchestrator
│   │   ├── blood_group_predictor.py # Blood group CNN
│   │   ├── document_parser.py      # PDF/DOCX parsing
│   │   ├── gemini_service.py       # Gemini text AI
│   │   ├── gemini_vision.py        # Gemini vision AI
│   │   ├── measurement_extractor.py
│   │   ├── parameter_extractor.py  # Clinical parameter extraction
│   │   ├── reference_range_db.py   # Reference ranges
│   │   ├── risk_explainer.py       # Risk assessment
│   │   └── threshold_evaluator.py  # Threshold validation
│   ├── tests/                 # Unit & integration tests
│   │   ├── conftest.py
│   │   ├── fixtures/          # Test data
│   │   └── test_*.py          # Test files
│   └── utils/                 # Utility functions
│       ├── error_handlers.py
│       └── logger.py
│
├── frontend/                   # Frontend UI (Streamlit)
│   ├── app.py                 # Main application
│   ├── frontend_components.py # Reusable UI components
│   └── handlers.py            # Request handlers
│
├── .env                       # Environment variables (create from .env.example)
├── .env.example               # Example environment file
├── .gitignore                 # Git ignore rules
├── requirements.txt           # Python dependencies
├── run.py                     # Application startup script
├── documentation.html         # HTML documentation
├── DOCUMENTATION.md           # Comprehensive documentation
├── README.md                  # This file
└── LICENSE                    # MIT License
```

---

## 🎯 Features in Detail

### 1. Lab Report Analysis
- Upload PDF or DOCX clinical reports
- Automatic parameter extraction with AI
- Dual analysis pipeline:
  - **AI Analysis:** Gemini 2.5 Flash for intelligent interpretation
  - **Threshold Analysis:** Objective parameter checking against reference ranges
- Risk indicator detection with severity classification (High/Medium/Low)
- Threshold-based parameter evaluation with deviation percentages
- Patient-friendly summaries with medical terminology
- Actionable follow-up recommendations

### 2. Medical Image Analysis
- **Supported Types:**
  - Chest X-rays (9-point systematic review)
  - Brain CT scans (10-point systematic review)
  - Bone X-rays (fractures, arthritis)
  - MRI scans (soft tissue, tumors)
  - Ultrasound (organs, pregnancy)
  - Auto-detection mode
- **Features:**
  - Specialized medical imaging interpretation with Gemini Vision
  - Medical expert persona (15-year experience)
  - Structured findings report
  - Abnormality detection with confidence scoring
  - Urgency assessment (Critical/High/Medium/Low)
  - Color-coded risk indicators

### 3. Blood Group Prediction
- Custom PyTorch CNN model trained on fingerprint patterns
- Supports all 8 blood groups (A+, A-, B+, B-, AB+, AB-, O+, O-)
- GradCAM visualization for explainability
- Top-3 predictions with confidence scores
- Fast prediction (<1 second)
- High accuracy specialized model

### 4. Multi-Modal Analysis
- Combines lab reports and medical images simultaneously
- Cross-modal correlation detection
- Identifies patterns across diagnostic modalities
- Integrated diagnostic insights with confidence scoring
- Comprehensive recommendations
- **Use Cases:**
  - Diabetic retinopathy (glucose + retinal imaging)
  - Pneumonia (WBC + chest X-ray)
  - Cardiovascular risk (cholesterol + cardiac imaging)

---

## 🛠️ Technology Stack

### Backend
- **Framework:** FastAPI 0.104.1 (Modern, high-performance Python web framework)
- **Server:** Uvicorn 0.24.0 (Lightning-fast ASGI server)
- **AI/ML:** 
  - **Google Generative AI** 0.8.3 (Gemini 2.5 Flash for text and vision analysis)
  - **PyTorch** 2.1.0 (Deep learning framework for blood group prediction)
  - **TorchVision** 0.16.0 (Computer vision utilities)
- **Document Processing:** 
  - PDFPlumber 0.10.3 (Advanced PDF text extraction)
  - PyPDF2 3.0.1 (PDF manipulation)
  - python-docx 1.1.0 (Word document processing)
- **Image Processing:** Pillow 10.1.0
- **Testing:** pytest 7.4.3, hypothesis 6.88.1 (Property-based testing)
- **Utilities:** python-dotenv, requests, python-multipart

### Frontend
- **Framework:** Streamlit 1.28.1 (Interactive web application framework)
- **UI Design:** Custom CSS with professional medical-grade styling
- **Responsive:** Mobile and desktop compatible
- **Color Scheme:** Medical blue gradient (#1e3c72 to #2a5298)

### AI Models
- **Text Analysis:** Gemini 2.5 Flash - Medical text interpretation with expert prompts
- **Vision Analysis:** Gemini 2.5 Flash - Medical image analysis with systematic review protocols
- **Blood Group CNN:** Custom PyTorch ResNet-based model with residual connections

### Architecture Patterns
- **RESTful API:** Standard HTTP methods and status codes
- **Microservices:** Modular service architecture
- **MVC Pattern:** Separation of concerns
- **Error Handling:** Comprehensive error tracking and logging
- **Async Processing:** Non-blocking I/O operations

## 🌟 What Makes MediVision AI Unique

### Comparison with General AI Tools (ChatGPT, Gemini, Claude)

| Feature | MediVision AI | ChatGPT/Claude/Gemini |
|---------|---------------|----------------------|
| **Medical Specialization** | ✅ Purpose-built for clinical analysis | ❌ General-purpose conversational AI |
| **Structured Output** | ✅ Standardized JSON format with sections | ❌ Unstructured text responses |
| **Risk Classification** | ✅ Automatic severity categorization | ❌ No systematic risk assessment |
| **Threshold Validation** | ✅ Objective parameter evaluation | ❌ No clinical threshold checking |
| **Multi-Modal Analysis** | ✅ Integrated lab + imaging correlation | ❌ Limited multi-modal capabilities |
| **Medical Image Analysis** | ✅ Specialized medical imaging interpretation | ⚠️ Basic image description only |
| **Blood Group Prediction** | ✅ Custom CNN model | ❌ Not available |
| **Safety Filters** | ✅ Medical compliance enforcement | ❌ No medical-specific safety |
| **API Integration** | ✅ RESTful API for healthcare systems | ❌ Limited API access |

### Key Differentiators

1. **Medical Domain Expertise**
   - 15-year expert persona in AI prompts
   - Systematic review protocols (9-point chest X-ray, 10-point brain CT)
   - Clinical terminology with patient-friendly explanations

2. **Dual Analysis Pipeline**
   - AI-powered analysis (Gemini 2.5 Flash)
   - Threshold-based validation (objective parameter checking)
   - Hybrid risk detection combining both approaches

3. **Structured Clinical Output**
   - Professional clinical summaries
   - Organized key findings with values
   - Risk indicators with severity, category, and deviation percentages
   - Actionable follow-up suggestions
   - Automatic medical disclaimers

4. **Medical Safety Features**
   - Built-in safety filters
   - Clear distinction between informational analysis and medical advice
   - Urgency level assessment
   - Professional consultation recommendations

---

## 🧪 Testing

Run all tests:
```bash
pytest
```

Run specific test file:
```bash
pytest backend/tests/test_analysis_engine.py
```

Run with coverage:
```bash
pytest --cov=backend
```

Run property-based tests:
```bash
pytest backend/tests/test_threshold_evaluator.py -v
```

### Test Coverage
- Unit tests for all services
- Integration tests for API endpoints
- Property-based tests for threshold evaluation
- Fixture-based testing with sample medical data

---

## 📖 Documentation

- **[DOCUMENTATION.md](DOCUMENTATION.md)** - Comprehensive project documentation
- **[documentation.html](documentation.html)** - Professional HTML documentation (open in browser)
- **API Documentation** - Interactive API docs at http://localhost:8000/docs (when backend is running)

### Documentation Includes:
- Project overview and capabilities
- Problem statement and healthcare challenges
- Detailed comparison with ChatGPT/Gemini/Claude
- Technology stack breakdown
- System architecture diagrams
- Installation and setup guide
- Complete usage guide for all features
- API documentation with examples
- Technical implementation details
- Future enhancements roadmap

---

## 🎨 UI Design

Professional medical-grade website design featuring:
- **Color Scheme:** Medical blue gradient (#1e3c72 to #2a5298)
- **Typography:** Clean, professional fonts (Segoe UI, Tahoma, Geneva, Verdana)
- **Layout:** Modern card-based design with proper spacing
- **Risk Indicators:** Color-coded severity badges (High/Medium/Low)
- **Responsive Design:** Works seamlessly on desktop, tablet, and mobile
- **Professional Branding:** Clean interface without AI provider branding
- **Smooth Animations:** Subtle transitions for better user experience
- **Accessibility:** High contrast and readable text
- **Navigation:** Intuitive tab-based navigation for different analysis types

---

## ⚠️ Medical Disclaimer

**IMPORTANT:** MediVision AI is for **informational and educational purposes only**. It is NOT intended as medical advice, diagnosis, or treatment. 

- This system provides analysis and insights based on AI interpretation
- Always consult qualified healthcare professionals for medical decisions
- Do not use this system as a substitute for professional medical consultation
- Results should be reviewed by licensed medical practitioners
- The system is designed to assist, not replace, healthcare professionals

---

## 🔒 Security & Privacy

### Data Handling
- **Stateless Operation:** No data persistence by default
- **In-Memory Processing:** Files processed in memory only
- **No Database Storage:** Medical data not stored (unless explicitly configured)
- **Temporary File Cleanup:** Automatic cleanup after processing
- **HIPAA Considerations:** Designed with healthcare privacy in mind

### API Security
- **CORS Configuration:** Controlled cross-origin access
- **File Size Limits:** Maximum 10MB per file
- **File Type Validation:** Only allowed file types accepted
- **Input Sanitization:** Protection against malicious inputs
- **Error Handling:** No sensitive data leakage in error messages
- **Logging:** Comprehensive logging without exposing sensitive data

### Medical Safety
- **Automatic Disclaimers:** All outputs include medical disclaimers
- **Safety Filters:** Prevents diagnostic language misuse
- **Urgency Indicators:** Clear urgency level assessment
- **Professional Consultation:** Always recommends professional review

### API Key Security
- **Environment Variables:** API keys stored in .env file (not in code)
- **Git Ignore:** .env file excluded from version control
- **Example File:** .env.example provided without sensitive data

---

## 🚀 Deployment

### Local Development
Follow the Quick Start guide above for local development setup.

### Production Considerations
For production deployment:
- Use environment-specific configuration
- Implement proper authentication and authorization
- Set up HTTPS/SSL certificates
- Configure proper CORS policies
- Implement rate limiting
- Set up monitoring and logging
- Consider HIPAA compliance requirements
- Use secure API key management (e.g., AWS Secrets Manager, Azure Key Vault)

### Docker Deployment (Future)
Docker support is planned for easier deployment and scaling.

---

## 📝 API Documentation

### Interactive Documentation

Once the backend is running, access interactive API documentation:

**Swagger UI:**
```
http://localhost:8000/docs
```

**ReDoc:**
```
http://localhost:8000/redoc
```

### Main Endpoints

1. **Health Check**
   - `GET /health` - Check API health status

2. **Lab Report Analysis**
   - `POST /analyze` - Analyze clinical lab reports (PDF/DOCX)

3. **Medical Image Analysis**
   - `POST /analyze-image` - Analyze medical images with AI

4. **Multi-Modal Analysis**
   - `POST /analyze-multimodal` - Combined lab report and image analysis

5. **Blood Group Prediction**
   - `POST /predict-blood-group` - Predict blood group from fingerprint

6. **System Status**
   - `GET /status` - Get system status and configuration

### Features
- Complete request/response schemas
- Try-it-out functionality
- Authentication details
- Error response examples
- File upload specifications

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
7. Push to the branch (`git push origin feature/AmazingFeature`)
8. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide for Python code
- Write comprehensive tests for new features
- Update documentation for any changes
- Add docstrings to all functions and classes
- Keep commits atomic and well-described

### Areas for Contribution
- Additional medical image types support
- Enhanced AI prompts for better accuracy
- UI/UX improvements
- Performance optimizations
- Documentation improvements
- Bug fixes and issue resolution

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### MIT License Summary
- ✅ Commercial use allowed
- ✅ Modification allowed
- ✅ Distribution allowed
- ✅ Private use allowed
- ⚠️ Liability and warranty limitations apply

---

## 👥 Authors & Acknowledgments

### Development Team
MediVision AI - AI-Powered Clinical Decision Support System

### Acknowledgments
- **Google Gemini AI** - For powerful vision and language models
- **PyTorch Team** - For the deep learning framework
- **FastAPI Community** - For the excellent web framework
- **Streamlit Team** - For the intuitive UI framework
- **Open-source Medical Datasets** - For training and validation data
- **Healthcare Professionals** - For domain expertise and feedback

---

## 📞 Support & Contact

### Getting Help
- **Documentation:** Check [DOCUMENTATION.md](DOCUMENTATION.md) for comprehensive guides
- **API Docs:** Visit http://localhost:8000/docs when backend is running
- **Issues:** Open an issue on GitHub for bugs or feature requests
- **Questions:** Use GitHub Discussions for questions and community support

### Troubleshooting
Common issues and solutions:

1. **API Key Error:** Ensure your Gemini API key is correctly set in `.env` file
2. **Port Already in Use:** Change ports in `.env` file or stop conflicting services
3. **Module Not Found:** Run `pip install -r requirements.txt` to install dependencies
4. **File Upload Error:** Check file size (max 10MB) and format (PDF/DOCX for reports, JPEG/PNG for images)

---

## 🔄 Version History

### Version 2.1.0 (Current - February 2026)
- ✅ Migrated to Google Gemini 2.5 Flash for text and vision analysis
- ✅ Removed Ollama dependency for simplified setup
- ✅ Enhanced medical expert prompts (15-year experience persona)
- ✅ Dual analysis pipeline (AI + threshold-based validation)
- ✅ Professional UI redesign (removed emojis, clean medical-grade interface)
- ✅ Comprehensive documentation (HTML + Markdown)
- ✅ Improved risk classification with deviation percentages
- ✅ Structured JSON output format
- ✅ Enhanced medical safety features

### Version 2.0.0
- ✅ Multi-modal analysis capability
- ✅ Blood group prediction from fingerprints
- ✅ Professional UI design
- ✅ Threshold-based parameter evaluation
- ✅ Reference range database

### Version 1.0.0
- ✅ Lab report analysis (PDF/DOCX)
- ✅ Medical image analysis
- ✅ Basic risk assessment
- ✅ Streamlit frontend
- ✅ FastAPI backend

---

## 🎯 Roadmap & Future Enhancements

### Planned Features
- [ ] **DICOM Support** - Native medical imaging format (DICOM files)
- [ ] **Report History** - Optional user account system with analysis history
- [ ] **Batch Processing** - Analyze multiple files simultaneously
- [ ] **PDF Export** - Generate professional PDF reports
- [ ] **Telemedicine Integration** - Video consultation booking
- [ ] **Mobile Applications** - iOS and Android native apps
- [ ] **Multi-Language Support** - Interface and analysis in multiple languages
- [ ] **Voice Input** - Speech-to-text for clinical notes
- [ ] **EHR/EMR Integration** - Connect with existing healthcare systems
- [ ] **Temporal Analysis** - Compare scans over time to track progression
- [ ] **Docker Support** - Containerized deployment
- [ ] **Cloud Deployment** - AWS/Azure/GCP deployment guides

### Research Directions
- [ ] **Federated Learning** - Privacy-preserving model training across institutions
- [ ] **Enhanced Explainability** - More detailed AI decision explanations
- [ ] **Predictive Analytics** - Disease progression modeling
- [ ] **Drug Interaction Checking** - Medication safety analysis
- [ ] **Clinical Trial Matching** - Patient-to-trial matching algorithms
- [ ] **Real-time Monitoring** - Integration with medical devices
- [ ] **3D Medical Imaging** - Support for 3D CT/MRI reconstructions

---

## 🏆 Project Highlights

- **🎯 Medical Specialization:** Purpose-built for clinical analysis, not general-purpose AI
- **🔬 Dual Analysis:** Combines AI intelligence with objective threshold validation
- **📊 Structured Output:** Standardized JSON format for easy integration
- **🛡️ Safety First:** Built-in medical disclaimers and safety filters
- **⚡ Fast & Efficient:** Powered by Gemini 2.5 Flash for quick analysis
- **🎨 Professional UI:** Clean, medical-grade interface design
- **🔒 Privacy-Focused:** Stateless operation with no data persistence
- **📚 Well-Documented:** Comprehensive documentation and API specs
- **🧪 Thoroughly Tested:** Unit, integration, and property-based tests
- **🚀 Easy Setup:** Single command to start the entire application

---

**Built with ❤️ for better healthcare**

**Version:** 2.1.0  
**Last Updated:** February 24, 2026  
**Status:** Active Development

---

### ⭐ Star this repository if you find it helpful!

