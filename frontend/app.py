# Streamlit frontend application
import streamlit as st
import requests
import json
import time
from typing import Optional, Dict, Any

# Configure Streamlit page
st.set_page_config(
    page_title="AI-Powered Clinical Decision Support System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .risk-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    
    .risk-medium {
        background-color: #fff8e1;
        border-left: 5px solid #ff9800;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    
    .risk-low {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    
    .finding-item {
        background-color: #f8f9fa;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 5px;
        border-left: 3px solid #007bff;
    }
    
    .suggestion-item {
        background-color: #e7f3ff;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 5px;
        border-left: 3px solid #17a2b8;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "http://localhost:8000"
SUPPORTED_FORMATS = [".pdf", ".docx"]
MAX_FILE_SIZE_MB = 10

def check_api_health() -> bool:
    """Check if the FastAPI backend is available"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def upload_and_analyze_document(file) -> Optional[Dict[str, Any]]:
    """Upload document to API and get analysis results"""
    # First, make the request
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        
        with st.spinner("Analyzing document... This may take a few moments."):
            response = requests.post(
                f"{API_BASE_URL}/analyze",
                files=files,
                timeout=120  # 2 minute timeout for analysis
            )
        
        # Process the response (only if request succeeded)
        if response.status_code == 200:
            return response.json()
        else:
            # Handle error responses (non-200 status codes)
            try:
                error_data = response.json()
                st.error(f"Analysis failed: {error_data.get('message', 'Unknown error')}")
                
                # Show additional error details if available
                if 'details' in error_data:
                    st.error(f"Details: {error_data['details']}")
                
                # Show suggestions if available
                if 'suggestions' in error_data and error_data['suggestions']:
                    st.info("Suggestions:")
                    for suggestion in error_data['suggestions']:
                        st.write(f"• {suggestion}")
                        
            except json.JSONDecodeError:
                st.error(f"Analysis failed with status code: {response.status_code}")
            
            return None
        
    except requests.exceptions.Timeout:
        st.error("Analysis timed out. Please try again with a smaller document or check your connection.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the analysis service. Please ensure the backend is running.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        return None

def display_analysis_results(results: Dict[str, Any]):
    """Display the analysis results in a structured format"""
    if not results.get("success", False):
        st.error("Analysis was not successful")
        return
    
    analysis = results.get("analysis", {})
    
    # Display file information
    st.success(f"✅ Successfully analyzed: **{results.get('filename', 'Unknown file')}**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("File Size", f"{results.get('file_size_mb', 0):.2f} MB")
    with col2:
        if analysis.get('analysis_timestamp'):
            st.metric("Processed At", analysis['analysis_timestamp'])
    
    # Medical Disclaimer - Prominent display
    st.warning("⚠️ **MEDICAL DISCLAIMER**: " + analysis.get('medical_disclaimer', 
        'This analysis is for informational purposes only and should not be used for medical diagnosis or treatment decisions.'))
    
    # Display analysis sections
    if analysis.get('summary'):
        st.subheader("📋 Summary")
        st.info(analysis['summary'])
    
    if analysis.get('key_findings'):
        st.subheader("🔍 Key Medical Findings")
        findings = analysis['key_findings']
        if isinstance(findings, list):
            for i, finding in enumerate(findings, 1):
                st.markdown(f"""
                <div class="finding-item">
                    <strong>{i}.</strong> {finding}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.write(findings)
    
    if analysis.get('risk_indicators'):
        st.subheader("⚠️ Risk Indicators")
        indicators = analysis['risk_indicators']
        if isinstance(indicators, list):
            for i, indicator in enumerate(indicators, 1):
                if isinstance(indicator, dict):
                    # Handle structured risk indicator
                    severity = indicator.get('severity', 'medium')
                    finding = indicator.get('finding', 'Unknown risk')
                    category = indicator.get('category', 'general')
                    threshold_based = indicator.get('threshold_based', False)
                    
                    # Create clean display
                    severity_colors = {
                        'high': '🔴',
                        'medium': '🟡', 
                        'low': '🟢'
                    }
                    
                    emoji = severity_colors.get(severity, '⚪')
                    severity_text = severity.upper()
                    
                    # Build the display content
                    display_content = f"{emoji} <strong>{i}. {severity_text} RISK</strong> ({category.title()})"
                    
                    # Add threshold-based badge if applicable
                    if threshold_based:
                        display_content += " <span style='background-color: #e3f2fd; padding: 2px 6px; border-radius: 3px; font-size: 0.85em;'>📊 Threshold-Based</span>"
                    
                    display_content += f"<br><span style='margin-left: 20px;'>{finding}</span>"
                    
                    # Add detailed threshold information if available
                    if threshold_based:
                        param_name = indicator.get('parameter_name')
                        actual_value = indicator.get('actual_value')
                        unit = indicator.get('unit')
                        reference_range = indicator.get('reference_range')
                        deviation_percent = indicator.get('deviation_percent')
                        description = indicator.get('description', '')
                        
                        if param_name and actual_value is not None:
                            display_content += "<br><div style='margin-left: 20px; margin-top: 8px; font-size: 0.9em;'>"
                            display_content += f"<strong>Parameter:</strong> {param_name}<br>"
                            display_content += f"<strong>Your Value:</strong> {actual_value} {unit}"
                            
                            if reference_range:
                                display_content += f"<br><strong>Normal Range:</strong> {reference_range}"
                            
                            if deviation_percent is not None and deviation_percent != 0:
                                display_content += f"<br><strong>Deviation:</strong> {abs(deviation_percent):.1f}% {'above' if deviation_percent > 0 else 'below'} normal"
                            
                            if description:
                                display_content += f"<br><strong>Details:</strong> {description}"
                            
                            display_content += "</div>"
                    
                    # Display with appropriate styling
                    if severity == 'high':
                        st.markdown(f"""
                        <div class="risk-high">
                            {display_content}
                        </div>
                        """, unsafe_allow_html=True)
                    elif severity == 'medium':
                        st.markdown(f"""
                        <div class="risk-medium">
                            {display_content}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="risk-low">
                            {display_content}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    # Handle simple string format
                    st.warning(f"⚠️ **{i}.** {indicator}")
        else:
            st.warning(indicators)
    else:
        st.success("🟢 No significant risk indicators identified.")
    
    if analysis.get('follow_up_suggestions'):
        st.subheader("💡 Follow-up Suggestions")
        suggestions = analysis['follow_up_suggestions']
        if isinstance(suggestions, list):
            for i, suggestion in enumerate(suggestions, 1):
                st.markdown(f"""
                <div class="suggestion-item">
                    <strong>{i}.</strong> {suggestion}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info(suggestions)

def main():
    """Main Streamlit application"""
    
    # Header with styling
    st.markdown("""
    <div class="main-header">
        <h1>🏥 AI-Powered Multi-Modal Clinical Decision Support System</h1>
        <p style="font-size: 1.1em; margin-top: 0.5rem;">Intelligent Medical Report Analysis with Threshold-Based Risk Assessment</p>
        <p style="font-size: 0.9em; opacity: 0.9; margin-top: 0.3rem;">For informational and educational purposes only</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ **Backend Service Unavailable**")
        st.error("The analysis service is not currently available. Please ensure the FastAPI backend is running on localhost:8000")
        st.info("To start the backend service, run: `uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000`")
        return
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ Information")
        st.write("**Supported Formats:**")
        for fmt in SUPPORTED_FORMATS:
            st.write(f"• {fmt.upper()}")
        
        st.write(f"**Maximum File Size:** {MAX_FILE_SIZE_MB} MB")
        
        st.markdown("---")
        st.write("**How it works:**")
        st.write("1. Upload a clinical document")
        st.write("2. Click 'Analyze Document'")
        st.write("3. Review the AI-generated insights")
        
        st.markdown("---")
        st.warning("**Important:** This tool is for informational purposes only and should not replace professional medical advice.")
    
    # Main content area
    st.markdown("---")
    
    # File upload section
    st.subheader("📁 Upload Clinical Document")
    
    uploaded_file = st.file_uploader(
        "Choose a clinical document to analyze",
        type=['pdf', 'docx'],
        help=f"Upload a PDF or DOCX file (max {MAX_FILE_SIZE_MB}MB)"
    )
    
    if uploaded_file is not None:
        # Display file information
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Filename:** {uploaded_file.name}")
        with col2:
            st.write(f"**Size:** {file_size_mb:.2f} MB")
        with col3:
            st.write(f"**Type:** {uploaded_file.type}")
        
        # Validate file size
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error(f"File size ({file_size_mb:.2f} MB) exceeds the maximum limit of {MAX_FILE_SIZE_MB} MB. Please upload a smaller file.")
            return
        
        # Analysis trigger button
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button(
                "🔍 Analyze Document",
                type="primary",
                use_container_width=True,
                help="Click to start AI analysis of your clinical document"
            )
        
        # Process analysis when button is clicked
        if analyze_button:
            # Reset file pointer
            uploaded_file.seek(0)
            
            # Perform analysis
            results = upload_and_analyze_document(uploaded_file)
            
            if results:
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                display_analysis_results(results)
                
                # Option to download results (future enhancement)
                st.markdown("---")
                st.info("💡 **Tip:** You can take a screenshot or copy the results for your records. Remember that this analysis is for informational purposes only.")
    
    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload a clinical document (PDF or DOCX) to begin analysis.")
        
        # Example of what the tool can analyze
        with st.expander("📖 What can this tool analyze?"):
            st.write("""
            This tool can analyze various types of clinical documents including:
            
            • **Lab Reports** - Blood tests, urine tests, imaging results
            • **Medical Records** - Doctor's notes, patient summaries
            • **Test Results** - Diagnostic test outcomes and measurements
            • **Clinical Notes** - Healthcare provider observations and assessments
            
            The AI will provide:
            - A patient-friendly summary in plain language
            - Key medical findings and observations
            - Risk indicators that may need attention
            - General follow-up suggestions (not medical advice)
            """)
        
        with st.expander("⚠️ Important Disclaimers"):
            st.warning("""
            **This tool is for informational and educational purposes only:**
            
            • Results are NOT medical diagnoses
            • Do NOT use for treatment decisions
            • Always consult healthcare professionals
            • AI analysis may contain errors or omissions
            • Not a substitute for professional medical advice
            """)

if __name__ == "__main__":
    main()