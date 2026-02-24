"""
Frontend components for the Clinical Decision Support System
Contains reusable UI components, styling, and API interaction functions
"""

import streamlit as st
import requests
import json
import re
from typing import Optional, Dict, Any
from PIL import Image
import io

# API Configuration
API_BASE_URL = "http://localhost:8000"

def convert_markdown_to_html(text: str) -> str:
    """
    Convert markdown bold (**text**) to HTML bold (<strong>text</strong>)
    
    Args:
        text: Text with markdown formatting
        
    Returns:
        Text with HTML formatting
    """
    # Replace **text** with <strong>text</strong>
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    return text

def apply_custom_css():
    """Apply beautiful custom CSS styling - Professional Medical Website Design"""
    st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@400;500;600;700&display=swap');
        
        /* Global Styles */
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Main container */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }
        
        /* Main header styling - Modern Medical Design */
        .main-header {
            text-align: center;
            padding: 3rem 2rem;
            background: linear-gradient(135deg, #0066cc 0%, #004c99 50%, #003366 100%);
            color: white;
            border-radius: 20px;
            margin-bottom: 2.5rem;
            box-shadow: 0 15px 40px rgba(0, 102, 204, 0.3);
            position: relative;
            overflow: hidden;
        }
        
        .main-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg"><defs><pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="1"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
            opacity: 0.3;
        }
        
        .main-header h1 {
            font-family: 'Poppins', sans-serif;
            font-size: 2.8rem;
            font-weight: 700;
            margin-bottom: 0.8rem;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
            position: relative;
            z-index: 1;
            letter-spacing: -0.5px;
        }
        
        .main-header p {
            font-size: 1.15rem;
            opacity: 0.95;
            margin: 0.5rem 0;
            position: relative;
            z-index: 1;
            font-weight: 400;
        }
        
        .main-header .subtitle {
            font-size: 1rem;
            opacity: 0.85;
            margin-top: 1rem;
            padding: 0.5rem 1.5rem;
            background: rgba(255,255,255,0.1);
            border-radius: 25px;
            display: inline-block;
            backdrop-filter: blur(10px);
        }
        
        /* Risk indicator cards - Enhanced Medical Design */
        .risk-card {
            padding: 1.5rem;
            margin: 1rem 0;
            border-radius: 15px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.08);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            border: 2px solid transparent;
            position: relative;
            overflow: hidden;
        }
        
        .risk-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 6px;
            height: 100%;
            transition: width 0.3s ease;
        }
        
        .risk-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 30px rgba(0,0,0,0.12);
        }
        
        .risk-critical {
            background: linear-gradient(135deg, #ff4757 0%, #ff3838 100%);
            color: white;
            border-color: #ff1744;
            animation: pulse-critical 2s infinite;
        }
        
        .risk-critical::before {
            background: #c62828;
        }
        
        @keyframes pulse-critical {
            0%, 100% { 
                box-shadow: 0 6px 20px rgba(255, 71, 87, 0.3);
            }
            50% { 
                box-shadow: 0 6px 30px rgba(255, 71, 87, 0.5);
            }
        }
        
        .risk-high {
            background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
            border-color: #ef5350;
            color: #b71c1c;
        }
        
        .risk-high::before {
            background: #ef5350;
        }
        
        .risk-medium {
            background: linear-gradient(135deg, #fff8e1 0%, #ffe082 100%);
            border-color: #ffa726;
            color: #e65100;
        }
        
        .risk-medium::before {
            background: #ffa726;
        }
        
        .risk-low {
            background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
            border-color: #66bb6a;
            color: #1b5e20;
        }
        
        .risk-low::before {
            background: #66bb6a;
        }
        
        /* Finding and suggestion cards - Modern Design */
        .finding-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            padding: 1.2rem;
            margin: 0.8rem 0;
            border-radius: 12px;
            border-left: 5px solid #0066cc;
            box-shadow: 0 4px 12px rgba(0,0,0,0.06);
            transition: all 0.3s ease;
        }
        
        .finding-card:hover {
            transform: translateX(5px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.1);
        }
        
        .suggestion-card {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            padding: 1.2rem;
            margin: 0.8rem 0;
            border-radius: 12px;
            border-left: 5px solid #1976d2;
            box-shadow: 0 4px 12px rgba(25, 118, 210, 0.1);
            transition: all 0.3s ease;
        }
        
        .suggestion-card:hover {
            transform: translateX(5px);
            box-shadow: 0 6px 16px rgba(25, 118, 210, 0.15);
        }
        
        /* Diagnosis card - Premium Design */
        .diagnosis-card {
            background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
            padding: 2rem;
            margin: 1.5rem 0;
            border-radius: 16px;
            border: 2px solid #9c27b0;
            box-shadow: 0 8px 24px rgba(156, 39, 176, 0.15);
            position: relative;
            overflow: hidden;
        }
        
        .diagnosis-card::before {
            content: '🔍';
            position: absolute;
            top: -20px;
            right: -20px;
            font-size: 120px;
            opacity: 0.05;
        }
        
        .diagnosis-card h3 {
            color: #6a1b9a;
            margin-bottom: 1rem;
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            font-size: 1.4rem;
        }
        
        /* Confidence badge - Modern Pills */
        .confidence-badge {
            display: inline-block;
            padding: 0.5rem 1.2rem;
            border-radius: 25px;
            font-weight: 600;
            font-size: 0.95rem;
            margin: 0.5rem 0.5rem 0.5rem 0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        
        .confidence-badge:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.15);
        }
        
        .confidence-high {
            background: linear-gradient(135deg, #4caf50 0%, #66bb6a 100%);
            color: white;
        }
        
        .confidence-medium {
            background: linear-gradient(135deg, #ff9800 0%, #ffa726 100%);
            color: white;
        }
        
        .confidence-low {
            background: linear-gradient(135deg, #f44336 0%, #ef5350 100%);
            color: white;
        }
        
        /* Urgency badge - Alert Design */
        .urgency-badge {
            display: inline-block;
            padding: 0.5rem 1.2rem;
            border-radius: 25px;
            font-weight: 600;
            font-size: 0.95rem;
            margin: 0.5rem 0.5rem 0.5rem 0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        
        .urgency-critical {
            background: linear-gradient(135deg, #d32f2f 0%, #f44336 100%);
            color: white;
            animation: pulse-urgency 2s infinite;
            box-shadow: 0 4px 20px rgba(211, 47, 47, 0.4);
        }
        
        @keyframes pulse-urgency {
            0%, 100% { 
                transform: scale(1);
                box-shadow: 0 4px 20px rgba(211, 47, 47, 0.4);
            }
            50% { 
                transform: scale(1.05);
                box-shadow: 0 6px 25px rgba(211, 47, 47, 0.6);
            }
        }
        
        .urgency-high {
            background: linear-gradient(135deg, #ff6b6b 0%, #ff8787 100%);
            color: white;
        }
        
        .urgency-medium {
            background: linear-gradient(135deg, #ffa726 0%, #ffb74d 100%);
            color: white;
        }
        
        .urgency-low {
            background: linear-gradient(135deg, #66bb6a 0%, #81c784 100%);
            color: white;
        }
        
        /* Tab styling - Modern Medical Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 1rem;
            border-radius: 15px;
            box-shadow: inset 0 2px 8px rgba(0,0,0,0.05);
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 12px;
            padding: 14px 28px;
            font-weight: 600;
            font-size: 1rem;
            background: white;
            border: 2px solid transparent;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            border-color: #0066cc;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 102, 204, 0.2);
        }
        
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, #0066cc 0%, #004c99 100%);
            color: white;
            border-color: #003366;
            box-shadow: 0 6px 16px rgba(0, 102, 204, 0.3);
        }
        
        /* Button styling - Professional Medical Buttons */
        .stButton>button {
            border-radius: 12px;
            font-weight: 600;
            padding: 0.75rem 2.5rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            border: 2px solid transparent;
            font-size: 1rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        .stButton>button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        }
        
        .stButton>button:active {
            transform: translateY(-1px);
        }
        
        .stButton>button[kind="primary"] {
            background: linear-gradient(135deg, #0066cc 0%, #004c99 100%);
            color: white;
        }
        
        .stButton>button[kind="primary"]:hover {
            background: linear-gradient(135deg, #0052a3 0%, #003d7a 100%);
        }
        
        /* File uploader - Modern Design */
        .uploadedFile {
            border-radius: 12px;
            border: 2px dashed #0066cc;
            background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
            padding: 1rem;
            transition: all 0.3s ease;
        }
        
        .uploadedFile:hover {
            border-color: #004c99;
            background: linear-gradient(135deg, #e3f2fd 0%, #ffffff 100%);
            box-shadow: 0 4px 12px rgba(0, 102, 204, 0.1);
        }
        
        /* Success/Error messages - Enhanced Alerts */
        .stSuccess {
            background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
            border-left: 5px solid #4caf50;
            border-radius: 12px;
            padding: 1.2rem;
            box-shadow: 0 4px 12px rgba(76, 175, 80, 0.15);
        }
        
        .stError {
            background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
            border-left: 5px solid #f44336;
            border-radius: 12px;
            padding: 1.2rem;
            box-shadow: 0 4px 12px rgba(244, 67, 54, 0.15);
        }
        
        .stWarning {
            background: linear-gradient(135deg, #fff8e1 0%, #ffe082 100%);
            border-left: 5px solid #ffa726;
            border-radius: 12px;
            padding: 1.2rem;
            box-shadow: 0 4px 12px rgba(255, 167, 38, 0.15);
        }
        
        .stInfo {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            border-left: 5px solid #2196f3;
            border-radius: 12px;
            padding: 1.2rem;
            box-shadow: 0 4px 12px rgba(33, 150, 243, 0.15);
        }
        
        /* Metric cards - Dashboard Style */
        .metric-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.08);
            text-align: center;
            border: 2px solid #e9ecef;
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 30px rgba(0,0,0,0.12);
            border-color: #0066cc;
        }
        
        /* Input fields - Modern Form Design */
        .stTextInput>div>div>input,
        .stTextArea>div>div>textarea,
        .stSelectbox>div>div>select {
            border-radius: 10px;
            border: 2px solid #e0e0e0;
            padding: 0.75rem 1rem;
            font-size: 1rem;
            transition: all 0.3s ease;
        }
        
        .stTextInput>div>div>input:focus,
        .stTextArea>div>div>textarea:focus,
        .stSelectbox>div>div>select:focus {
            border-color: #0066cc;
            box-shadow: 0 0 0 3px rgba(0, 102, 204, 0.1);
        }
        
        /* Sidebar - Professional Medical Sidebar */
        .css-1d391kg {
            background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
            border-right: 2px solid #e9ecef;
        }
        
        /* Expander - Collapsible Sections */
        .streamlit-expanderHeader {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 10px;
            padding: 1rem;
            font-weight: 600;
            border: 2px solid #e0e0e0;
            transition: all 0.3s ease;
        }
        
        .streamlit-expanderHeader:hover {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            border-color: #0066cc;
        }
        
        /* Loading spinner */
        .stSpinner > div {
            border-top-color: #0066cc !important;
        }
        
        /* Dataframe styling */
        .dataframe {
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #0066cc 0%, #004c99 100%);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #0052a3 0%, #003d7a 100%);
        }
        
        /* Animations */
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .main .block-container {
            animation: fadeIn 0.5s ease-out;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .main-header h1 {
                font-size: 2rem;
            }
            
            .main-header p {
                font-size: 1rem;
            }
            
            .stTabs [data-baseweb="tab"] {
                padding: 10px 16px;
                font-size: 0.9rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)


def display_header():
    """Display the main header - Professional Medical Website Design"""
    st.markdown("""
    <div class="main-header">
        <h1 style="font-size: 2.5em; margin: 0; font-weight: 700;">MediVision AI</h1>
        <p style="font-size: 1.3em; margin-top: 0.8rem; font-weight: 500;">
            AI-Powered Multi-Modal Clinical Decision Support System
        </p>
        <p style="font-size: 1.05em; opacity: 0.9; margin-top: 0.8rem;">
            Advanced Medical Report & Image Analysis with AI Technology
        </p>
        <div class="subtitle">
            Lab Reports | Medical Imaging | Blood Group Prediction
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_sidebar():
    """Display sidebar with information - Professional Medical Design"""
    with st.sidebar:
        # Logo/Brand section
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0; border-bottom: 2px solid #e9ecef; margin-bottom: 1.5rem;">
            <h2 style="color: #0066cc; margin: 0; font-family: 'Poppins', sans-serif; font-weight: 700;">
                MediVision AI
            </h2>
            <p style="color: #6c757d; font-size: 0.85rem; margin-top: 0.5rem;">
                Clinical Decision Support
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # System status
        st.markdown("### System Status")
        st.success("All Systems Operational")
        st.metric("Uptime", "99.9%", delta="0.1%")
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("### Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Patients", "0", delta="0")
        with col2:
            st.metric("Analyses", "0", delta="0")
        
        st.markdown("---")
        
        # Supported formats
        st.markdown("### Supported Formats")
        
        with st.expander("Documents", expanded=False):
            st.markdown("""
            - **PDF** (.pdf)
            - **Word** (.docx)
            
            *Max size: 10MB*
            """)
        
        with st.expander("Images", expanded=False):
            st.markdown("""
            - **JPEG** (.jpg, .jpeg)
            - **PNG** (.png)
            
            *Max size: 10MB*
            """)
        
        st.markdown("---")
        
        # Analysis types
        st.markdown("### Analysis Types")
        
        analysis_types = [
            ("Chest X-ray", "Pneumonia, lung cancer, heart issues"),
            ("Brain CT", "Stroke, bleeding, tumors"),
            ("Bone X-ray", "Fractures, arthritis"),
            ("MRI Scan", "Soft tissue, tumors"),
            ("Ultrasound", "Organs, pregnancy"),
            ("Auto-detect", "Automatic identification")
        ]
        
        for name, desc in analysis_types:
            st.markdown(f"""
            <div style="padding: 0.5rem; margin: 0.3rem 0; background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%); 
                        border-radius: 8px; border-left: 3px solid #0066cc;">
                <strong>{name}</strong><br>
                <small style="color: #6c757d;">{desc}</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick guide
        st.markdown("### 📖 Quick Guide")
        
        with st.expander("🔬 How to Use", expanded=False):
            st.markdown("""
            **1. Lab Report Analysis:**
            - Upload PDF/DOCX report
            - Click "Analyze Document"
            - View comprehensive results
            
            **2. Image Analysis:**
            - Upload medical image
            - Select image type
            - Add clinical context (optional)
            - Click "Analyze Image"
            
            **3. Multi-Modal:**
            - Upload both report & image
            - Get correlated findings
            - View integrated diagnosis
            
            **4. Blood Group:**
            - Upload fingerprint image
            - Get AI prediction
            - View confidence score
            
            **Note:** This is a standalone analysis tool. Upload any medical document or image for instant AI-powered analysis.
            - View analytics dashboard
            """)
        
        st.markdown("---")
        
        # Medical disclaimer
        st.markdown("### ⚠️ Important Notice")
        st.warning("""
        **Medical Disclaimer**
        
        This tool is for **informational and educational purposes only**. 
        
        Always consult qualified healthcare professionals for medical decisions.
        
        Not a substitute for professional medical advice, diagnosis, or treatment.
        """)
        
        st.markdown("---")
        
        # Footer
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; color: #6c757d; font-size: 0.85rem;">
            <p style="margin: 0.3rem 0;">© 2026 MediVision AI</p>
            <p style="margin: 0.3rem 0;">Version 2.1.0</p>
            <p style="margin: 0.3rem 0; font-size: 0.75rem;">For informational purposes only. Not a substitute for professional medical advice.</p>
        </div>
        """, unsafe_allow_html=True)


def check_api_health(api_url: str) -> bool:
    """Check if the FastAPI backend is available"""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False



def upload_and_analyze_document(file, api_url: str) -> Optional[Dict[str, Any]]:
    """Upload document to API and get analysis results"""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        
        with st.spinner("🔍 Analyzing document... This may take a few moments."):
            response = requests.post(
                f"{api_url}/analyze",
                files=files,
                timeout=120
            )
        
        if response.status_code == 200:
            return response.json()
        else:
            try:
                error_data = response.json()
                st.error(f"❌ Analysis failed: {error_data.get('message', 'Unknown error')}")
            except json.JSONDecodeError:
                st.error(f"❌ Analysis failed with status code: {response.status_code}")
            return None
        
    except requests.exceptions.Timeout:
        st.error("⏱️ Analysis timed out. Please try again.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("🔌 Could not connect to the analysis service.")
        return None
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {str(e)}")
        return None


def upload_and_analyze_image(file, image_type: str, clinical_context: str, api_url: str) -> Optional[Dict[str, Any]]:
    """Upload medical image to API and get analysis results"""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        data = {
            "image_type": image_type,
            "clinical_context": clinical_context if clinical_context else ""
        }
        
        with st.spinner("🔬 Analyzing medical image... This may take 10-30 seconds."):
            response = requests.post(
                f"{api_url}/analyze-image",
                files=files,
                data=data,
                timeout=120
            )
        
        if response.status_code == 200:
            return response.json()
        else:
            try:
                error_data = response.json()
                st.error(f"❌ Image analysis failed: {error_data.get('message', 'Unknown error')}")
            except json.JSONDecodeError:
                st.error(f"❌ Analysis failed with status code: {response.status_code}")
            return None
        
    except requests.exceptions.Timeout:
        st.error("⏱️ Image analysis timed out. Please try again.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("🔌 Could not connect to the analysis service.")
        return None
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {str(e)}")
        return None


def upload_and_analyze_multimodal(report_file, image_file, clinical_context: str, api_url: str) -> Optional[Dict[str, Any]]:
    """Upload both report and image for multi-modal analysis"""
    try:
        files = {}
        if report_file:
            files["report"] = (report_file.name, report_file.getvalue(), report_file.type)
        if image_file:
            files["image"] = (image_file.name, image_file.getvalue(), image_file.type)
        
        data = {"clinical_context": clinical_context if clinical_context else ""}
        
        with st.spinner("🔗 Performing multi-modal analysis... This may take 30-60 seconds."):
            response = requests.post(
                f"{api_url}/analyze-multimodal",
                files=files,
                data=data,
                timeout=180
            )
        
        if response.status_code == 200:
            return response.json()
        else:
            try:
                error_data = response.json()
                st.error(f"❌ Multi-modal analysis failed: {error_data.get('message', 'Unknown error')}")
            except json.JSONDecodeError:
                st.error(f"❌ Analysis failed with status code: {response.status_code}")
            return None
        
    except requests.exceptions.Timeout:
        st.error("⏱️ Multi-modal analysis timed out. Please try again.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("🔌 Could not connect to the analysis service.")
        return None
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {str(e)}")
        return None



def display_document_results(results: Dict[str, Any]):
    """Display document analysis results with beautiful formatting"""
    if not results.get("success", False):
        st.error("❌ Analysis was not successful")
        return
    
    analysis = results.get("analysis", {})
    
    # Success message
    st.success(f"✅ Successfully analyzed: **{results.get('filename', 'Unknown file')}**")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 File Size", f"{results.get('file_size_mb', 0):.2f} MB")
    with col2:
        if analysis.get('analysis_timestamp'):
            st.metric("⏰ Processed At", analysis['analysis_timestamp'])
    with col3:
        risk_count = len(analysis.get('risk_indicators', []))
        st.metric("⚠️ Risk Indicators", risk_count)
    
    st.markdown("---")
    
    # Medical Disclaimer
    st.warning("⚠️ **MEDICAL DISCLAIMER**: " + analysis.get('medical_disclaimer', 
        'This analysis is for informational purposes only.'))
    
    # Summary
    if analysis.get('summary'):
        st.markdown("### 📋 Summary")
        st.info(analysis['summary'])
    
    # Key Findings
    if analysis.get('key_findings'):
        st.markdown("### 🔍 Key Medical Findings")
        findings = analysis['key_findings']
        if isinstance(findings, list):
            for i, finding in enumerate(findings, 1):
                # Convert markdown bold to HTML
                finding_html = convert_markdown_to_html(finding)
                st.markdown(f"""
                <div class="finding-card">
                    <strong>{i}.</strong> {finding_html}
                </div>
                """, unsafe_allow_html=True)
    
    # Risk Indicators
    if analysis.get('risk_indicators'):
        st.markdown("### ⚠️ Risk Indicators")
        indicators = analysis['risk_indicators']
        if isinstance(indicators, list):
            for i, indicator in enumerate(indicators, 1):
                if isinstance(indicator, dict):
                    display_risk_indicator(indicator, i)
    else:
        st.success("🟢 No significant risk indicators identified.")
    
    # Follow-up Suggestions
    if analysis.get('follow_up_suggestions'):
        st.markdown("### 💡 Follow-up Suggestions")
        suggestions = analysis['follow_up_suggestions']
        if isinstance(suggestions, list):
            for i, suggestion in enumerate(suggestions, 1):
                # Convert markdown bold to HTML
                suggestion_html = convert_markdown_to_html(suggestion)
                st.markdown(f"""
                <div class="suggestion-card">
                    <strong>{i}.</strong> {suggestion_html}
                </div>
                """, unsafe_allow_html=True)


def display_risk_indicator(indicator: Dict, index: int):
    """Display a single risk indicator with beautiful formatting"""
    severity = indicator.get('severity', 'medium')
    finding = indicator.get('finding', 'Unknown risk')
    category = indicator.get('category', 'general')
    threshold_based = indicator.get('threshold_based', False)
    
    # Severity emoji
    severity_emojis = {
        'critical': '🔴',
        'high': '🔴',
        'medium': '🟡',
        'low': '🟢'
    }
    
    emoji = severity_emojis.get(severity, '⚪')
    severity_text = severity.upper()
    
    # Build display content
    display_content = f"{emoji} <strong>{index}. {severity_text} RISK</strong> ({category.title()})"
    
    if threshold_based:
        display_content += " <span style='background-color: #e3f2fd; padding: 2px 8px; border-radius: 12px; font-size: 0.85em;'>📊 Threshold-Based</span>"
    
    display_content += f"<br><span style='margin-left: 20px; font-size: 1.05em;'>{finding}</span>"
    
    # Add detailed threshold information
    if threshold_based:
        param_name = indicator.get('parameter_name')
        actual_value = indicator.get('actual_value')
        unit = indicator.get('unit')
        reference_range = indicator.get('reference_range')
        deviation_percent = indicator.get('deviation_percent')
        description = indicator.get('description', '')
        
        if param_name and actual_value is not None:
            display_content += "<br><div style='margin-left: 20px; margin-top: 10px; font-size: 0.95em; background: rgba(255,255,255,0.5); padding: 10px; border-radius: 8px;'>"
            display_content += f"<strong>📊 Parameter:</strong> {param_name}<br>"
            display_content += f"<strong>📈 Your Value:</strong> {actual_value} {unit}"
            
            if reference_range:
                display_content += f"<br><strong>✅ Normal Range:</strong> {reference_range}"
            
            if deviation_percent is not None and deviation_percent != 0:
                display_content += f"<br><strong>📉 Deviation:</strong> {abs(deviation_percent):.1f}% {'above' if deviation_percent > 0 else 'below'} normal"
            
            if description:
                display_content += f"<br><strong>ℹ️ Details:</strong> {description}"
            
            display_content += "</div>"
    
    # Display with appropriate styling
    risk_class = f"risk-{severity}"
    st.markdown(f"""
    <div class="risk-card {risk_class}">
        {display_content}
    </div>
    """, unsafe_allow_html=True)



def display_image_results(results: Dict[str, Any]):
    """Display medical image analysis results with beautiful formatting"""
    if not results.get("success", False):
        st.error("❌ Image analysis was not successful")
        return
    
    # Success message
    st.success(f"✅ Successfully analyzed: **{results.get('filename', 'Unknown file')}**")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📄 File Size", f"{results.get('file_size_mb', 0):.2f} MB")
    with col2:
        st.metric("🔬 Image Type", results.get('image_type', 'Unknown').replace('_', ' ').title())
    with col3:
        confidence = results.get('confidence', 0)
        confidence_class = "high" if confidence >= 75 else "medium" if confidence >= 50 else "low"
        st.markdown(f"""
        <div class="confidence-badge confidence-{confidence_class}">
            📊 Confidence: {confidence}%
        </div>
        """, unsafe_allow_html=True)
    with col4:
        urgency = results.get('urgency', 'medium').lower()
        st.markdown(f"""
        <div class="urgency-badge urgency-{urgency}">
            🚨 Urgency: {urgency.upper()}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Diagnosis Card
    diagnosis = results.get('diagnosis', 'No diagnosis available')
    # Convert markdown bold to HTML
    diagnosis_html = convert_markdown_to_html(diagnosis)
    st.markdown(f"""
    <div class="diagnosis-card">
        <h3>🔍 DIAGNOSIS</h3>
        <p style="font-size: 1.2em; margin: 0;">{diagnosis_html}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Issues Identified with Risk-Based Color Coding
    issues = results.get('issues', [])
    urgency = results.get('urgency', 'medium').lower()
    
    if issues:
        st.markdown("### ⚠️ Issues Identified")
        
        # Determine risk level styling based on urgency
        if urgency == 'critical':
            risk_class = 'risk-critical'
            risk_emoji = '🔴'
            risk_label = 'CRITICAL'
        elif urgency == 'high':
            risk_class = 'risk-high'
            risk_emoji = '🔴'
            risk_label = 'HIGH RISK'
        elif urgency == 'medium':
            risk_class = 'risk-medium'
            risk_emoji = '🟡'
            risk_label = 'MODERATE'
        else:
            risk_class = 'risk-low'
            risk_emoji = '🟢'
            risk_label = 'LOW RISK'
        
        # Display risk level banner
        st.markdown(f"""
        <div class="risk-card {risk_class}" style="margin-bottom: 1rem;">
            <h4 style="margin: 0;">{risk_emoji} {risk_label} - {len(issues)} Issue(s) Found</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Display each issue with appropriate styling
        for i, issue in enumerate(issues, 1):
            # Check if issue contains critical keywords
            issue_lower = issue.lower()
            is_critical_issue = any(keyword in issue_lower for keyword in [
                'fracture', 'hemorrhage', 'bleeding', 'tumor', 'mass', 'cancer',
                'stroke', 'infarction', 'pneumothorax', 'emergency', 'urgent',
                'severe', 'critical', 'acute', 'rupture', 'obstruction'
            ])
            
            # Use red styling for critical issues or high/critical urgency
            if is_critical_issue or urgency in ['critical', 'high']:
                issue_class = 'risk-high'
                issue_emoji = '🔴'
            else:
                issue_class = 'finding-card'
                issue_emoji = '⚠️'
            
            st.markdown(f"""
            <div class="risk-card {issue_class}">
                {issue_emoji} <strong>{i}.</strong> {convert_markdown_to_html(issue)}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("🟢 No significant issues identified.")
    
    # Follow-up Suggestions
    suggestions = results.get('suggestions', [])
    if suggestions:
        st.markdown("### 💡 Follow-up Suggestions")
        for i, suggestion in enumerate(suggestions, 1):
            # Convert markdown bold to HTML
            suggestion_html = convert_markdown_to_html(suggestion)
            st.markdown(f"""
            <div class="suggestion-card">
                <strong>{i}.</strong> {suggestion_html}
            </div>
            """, unsafe_allow_html=True)
    
    # Detailed Findings (if available)
    findings = results.get('findings', {})
    if findings and findings.get('detailed'):
        with st.expander("📋 View Detailed Findings"):
            st.write(findings['detailed'])


def display_multimodal_results(results: Dict[str, Any]):
    """Display multi-modal analysis results with beautiful formatting"""
    if not results.get("success", False):
        st.error("❌ Multi-modal analysis was not successful")
        return
    
    st.success("✅ Multi-modal analysis completed successfully!")
    
    st.markdown("---")
    
    # Create tabs for different analysis types
    tab1, tab2, tab3 = st.tabs(["📄 Report Analysis", "🔬 Image Analysis", "🔗 Correlation"])
    
    # Tab 1: Report Analysis
    with tab1:
        report_analysis = results.get('report_analysis')
        if report_analysis:
            if isinstance(report_analysis, dict) and not report_analysis.get('error'):
                display_document_results({"success": True, "analysis": report_analysis, "filename": "Lab Report"})
            else:
                st.warning("⚠️ Report analysis not available or failed")
        else:
            st.info("ℹ️ No report was provided for analysis")
    
    # Tab 2: Image Analysis
    with tab2:
        image_analysis = results.get('image_analysis')
        if image_analysis:
            if isinstance(image_analysis, dict) and not image_analysis.get('error'):
                display_image_results({**image_analysis, "success": True, "filename": "Medical Image"})
            else:
                st.warning("⚠️ Image analysis not available or failed")
        else:
            st.info("ℹ️ No image was provided for analysis")
    
    # Tab 3: Correlation
    with tab3:
        correlation = results.get('correlation')
        if correlation:
            st.markdown("### 🔗 Integrated Analysis")
            
            # Integrated Diagnosis
            integrated_diagnosis = correlation.get('integrated_diagnosis', 'No correlation found')
            # Convert markdown bold to HTML
            integrated_diagnosis_html = convert_markdown_to_html(integrated_diagnosis)
            confidence = correlation.get('confidence', 0)
            
            st.markdown(f"""
            <div class="diagnosis-card">
                <h3>🎯 Integrated Diagnosis</h3>
                <p style="font-size: 1.2em; margin: 0.5rem 0;">{integrated_diagnosis_html}</p>
                <div class="confidence-badge confidence-{'high' if confidence >= 75 else 'medium' if confidence >= 50 else 'low'}">
                    📊 Confidence: {confidence}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Correlations Found
            correlations_found = correlation.get('correlations', [])
            if correlations_found:
                st.markdown("### 🔍 Correlations Found")
                for i, corr in enumerate(correlations_found, 1):
                    # Convert markdown bold to HTML
                    corr_diagnosis = convert_markdown_to_html(corr.get('diagnosis', 'Unknown correlation'))
                    st.markdown(f"""
                    <div class="finding-card">
                        <strong>{i}.</strong> {corr_diagnosis}
                        <br><small>Confidence: {corr.get('confidence', 0)}%</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Recommendations
            recommendations = correlation.get('recommendations', [])
            if recommendations:
                st.markdown("### 💡 Integrated Recommendations")
                for i, rec in enumerate(recommendations, 1):
                    # Convert markdown bold to HTML
                    rec_html = convert_markdown_to_html(rec)
                    st.markdown(f"""
                    <div class="suggestion-card">
                        <strong>{i}.</strong> {rec_html}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("ℹ️ No correlation analysis available")
