"""
Frontend components for the Clinical Decision Support System
Contains reusable UI components, styling, and API interaction functions
"""

import streamlit as st
import requests
import json
from typing import Optional, Dict, Any
from PIL import Image
import io

# API Configuration
API_BASE_URL = "http://localhost:8000"

def apply_custom_css():
    """Apply beautiful custom CSS styling"""
    st.markdown("""
    <style>
        /* Main header styling */
        .main-header {
            text-align: center;
            padding: 2rem 1rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .main-header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        .main-header p {
            font-size: 1.1rem;
            opacity: 0.95;
            margin: 0.3rem 0;
        }
        
        /* Risk indicator cards */
        .risk-card {
            padding: 1.2rem;
            margin: 0.8rem 0;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        
        .risk-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        
        .risk-critical {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
            color: white;
            border-left: 6px solid #c92a2a;
        }
        
        .risk-high {
            background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
            border-left: 6px solid #f44336;
            color: #b71c1c;
        }
        
        .risk-medium {
            background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
            border-left: 6px solid #ff9800;
            color: #e65100;
        }
        
        .risk-low {
            background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
            border-left: 6px solid #4caf50;
            color: #1b5e20;
        }
        
        /* Finding and suggestion cards */
        .finding-card {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 8px;
            border-left: 4px solid #007bff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .suggestion-card {
            background: linear-gradient(135deg, #e7f3ff 0%, #cfe2ff 100%);
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 8px;
            border-left: 4px solid #17a2b8;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        /* Diagnosis card */
        .diagnosis-card {
            background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
            padding: 1.5rem;
            margin: 1rem 0;
            border-radius: 12px;
            border-left: 6px solid #9c27b0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        .diagnosis-card h3 {
            color: #6a1b9a;
            margin-bottom: 0.5rem;
        }
        
        /* Confidence badge */
        .confidence-badge {
            display: inline-block;
            padding: 0.4rem 1rem;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.9rem;
            margin: 0.5rem 0;
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
        
        /* Urgency badge */
        .urgency-badge {
            display: inline-block;
            padding: 0.4rem 1rem;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.9rem;
            margin: 0.5rem 0.5rem 0.5rem 0;
        }
        
        .urgency-critical {
            background: linear-gradient(135deg, #d32f2f 0%, #f44336 100%);
            color: white;
            animation: pulse 2s infinite;
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
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        /* Image preview */
        .image-preview {
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        
        /* Metric cards */
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 12px 24px;
            font-weight: 600;
        }
        
        /* Button styling */
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            padding: 0.6rem 2rem;
            transition: all 0.3s;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        /* File uploader */
        .uploadedFile {
            border-radius: 8px;
            border: 2px dashed #667eea;
        }
        
        /* Success/Error messages */
        .stSuccess, .stError, .stWarning, .stInfo {
            border-radius: 8px;
            padding: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)


def display_header():
    """Display the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🏥 AI-Powered Multi-Modal Clinical Decision Support System</h1>
        <p style="font-size: 1.2em; margin-top: 0.5rem;">Intelligent Medical Report & Image Analysis</p>
        <p style="font-size: 0.95em; opacity: 0.9; margin-top: 0.5rem;">
            Powered by Google Gemini Vision AI | For Educational & Informational Purposes Only
        </p>
    </div>
    """, unsafe_allow_html=True)


def display_sidebar():
    """Display sidebar with information"""
    with st.sidebar:
        st.markdown("### 📊 System Information")
        
        # System status
        st.success("✅ System Online")
        
        st.markdown("---")
        
        # Supported formats
        st.markdown("### 📄 Supported Formats")
        st.markdown("**Documents:**")
        st.write("• PDF (.pdf)")
        st.write("• Word (.docx)")
        
        st.markdown("**Images:**")
        st.write("• JPEG (.jpg, .jpeg)")
        st.write("• PNG (.png)")
        
        st.markdown("---")
        
        # Image types
        st.markdown("### 🔬 Image Types")
        st.write("• Chest X-ray")
        st.write("• Brain CT Scan")
        st.write("• Bone X-ray")
        st.write("• MRI Scan")
        st.write("• Ultrasound")
        st.write("• Auto-detect")
        
        st.markdown("---")
        
        # Quick guide
        st.markdown("### 📖 Quick Guide")
        st.markdown("""
        **Lab Report Analysis:**
        1. Upload PDF/DOCX report
        2. Click Analyze
        3. View results
        
        **Image Analysis:**
        1. Upload medical image
        2. Select image type
        3. Add context (optional)
        4. Click Analyze
        
        **Multi-Modal:**
        1. Upload both files
        2. Click Analyze
        3. View correlations
        """)
        
        st.markdown("---")
        
        # Medical disclaimer
        st.warning("⚠️ **Medical Disclaimer**: This tool is for informational purposes only. Always consult healthcare professionals for medical decisions.")


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
                st.markdown(f"""
                <div class="finding-card">
                    <strong>{i}.</strong> {finding}
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
                st.markdown(f"""
                <div class="suggestion-card">
                    <strong>{i}.</strong> {suggestion}
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
    st.markdown(f"""
    <div class="diagnosis-card">
        <h3>🔍 DIAGNOSIS</h3>
        <p style="font-size: 1.2em; margin: 0;">{diagnosis}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Issues Identified
    issues = results.get('issues', [])
    if issues:
        st.markdown("### ⚠️ Issues Identified")
        for i, issue in enumerate(issues, 1):
            st.markdown(f"""
            <div class="finding-card">
                <strong>{i}.</strong> {issue}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("🟢 No significant issues identified.")
    
    # Follow-up Suggestions
    suggestions = results.get('suggestions', [])
    if suggestions:
        st.markdown("### 💡 Follow-up Suggestions")
        for i, suggestion in enumerate(suggestions, 1):
            st.markdown(f"""
            <div class="suggestion-card">
                <strong>{i}.</strong> {suggestion}
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
            confidence = correlation.get('confidence', 0)
            
            st.markdown(f"""
            <div class="diagnosis-card">
                <h3>🎯 Integrated Diagnosis</h3>
                <p style="font-size: 1.2em; margin: 0.5rem 0;">{integrated_diagnosis}</p>
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
                    st.markdown(f"""
                    <div class="finding-card">
                        <strong>{i}.</strong> {corr.get('diagnosis', 'Unknown correlation')}
                        <br><small>Confidence: {corr.get('confidence', 0)}%</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Recommendations
            recommendations = correlation.get('recommendations', [])
            if recommendations:
                st.markdown("### 💡 Integrated Recommendations")
                for i, rec in enumerate(recommendations, 1):
                    st.markdown(f"""
                    <div class="suggestion-card">
                        <strong>{i}.</strong> {rec}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("ℹ️ No correlation analysis available")
