# Streamlit frontend application - Enhanced with Medical Image Analysis
import streamlit as st
import requests
import json
import time
from typing import Optional, Dict, Any
from PIL import Image
import io

# Configure Streamlit page
st.set_page_config(
    page_title="MediVision AI - Clinical Decision Support",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
API_BASE_URL = "http://localhost:8000"
SUPPORTED_DOC_FORMATS = [".pdf", ".docx"]
SUPPORTED_IMG_FORMATS = [".jpg", ".jpeg", ".png"]
MAX_FILE_SIZE_MB = 10

# Import custom CSS and functions
from frontend_components import (
    apply_custom_css,
    check_api_health,
    display_header,
    display_sidebar
)

from handlers import (
    handle_document_analysis,
    handle_image_analysis,
    handle_multimodal_analysis,
    handle_blood_group_prediction
)

def main():
    """Main Streamlit application"""
    
    # Apply custom CSS
    apply_custom_css()
    
    # Display header
    display_header()
    
    # Check API health
    if not check_api_health(API_BASE_URL):
        st.error("⚠️ **Backend Service Unavailable**")
        st.error("Please ensure the FastAPI backend is running on localhost:8000")
        st.code("python run.py", language="bash")
        return
    
    # Display sidebar
    display_sidebar()
    
    # Main content - Tabs for different analysis types
    tab1, tab2, tab3, tab4 = st.tabs([
        "Lab Report Analysis",
        "Medical Image Analysis", 
        "Multi-Modal Analysis",
        "Blood Group Prediction"
    ])
    
    # Tab 1: Document Analysis
    with tab1:
        handle_document_analysis()
    
    # Tab 2: Image Analysis
    with tab2:
        handle_image_analysis()
    
    # Tab 3: Multi-Modal Analysis
    with tab3:
        handle_multimodal_analysis()
    
    # Tab 4: Blood Group Prediction
    with tab4:
        handle_blood_group_prediction()

if __name__ == "__main__":
    main()
