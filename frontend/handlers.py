"""
Handler functions for different analysis types
"""

import streamlit as st
from PIL import Image
import io
from frontend_components import (
    upload_and_analyze_document,
    upload_and_analyze_image,
    upload_and_analyze_multimodal,
    display_document_results,
    display_image_results,
    display_multimodal_results
)

API_BASE_URL = "http://localhost:8000"
MAX_FILE_SIZE_MB = 10


def handle_document_analysis():
    """Handle lab report analysis"""
    st.markdown("## 📄 Lab Report Analysis")
    st.markdown("Upload a clinical lab report (PDF or DOCX) for AI-powered analysis")
    
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a clinical document",
        type=['pdf', 'docx'],
        help=f"Upload a PDF or DOCX file (max {MAX_FILE_SIZE_MB}MB)",
        key="doc_uploader"
    )
    
    if uploaded_file is not None:
        # Display file info
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**📄 Filename:** {uploaded_file.name}")
        with col2:
            st.info(f"**📊 Size:** {file_size_mb:.2f} MB")
        with col3:
            st.info(f"**📋 Type:** {uploaded_file.type}")
        
        # Validate file size
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error(f"❌ File size ({file_size_mb:.2f} MB) exceeds maximum limit of {MAX_FILE_SIZE_MB} MB")
            return
        
        st.markdown("---")
        
        # Analyze button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔍 Analyze Document", type="primary", use_container_width=True, key="analyze_doc"):
                uploaded_file.seek(0)
                results = upload_and_analyze_document(uploaded_file, API_BASE_URL)
                
                if results:
                    st.markdown("---")
                    st.markdown("## 📊 Analysis Results")
                    display_document_results(results)
    else:
        st.info("👆 Please upload a clinical document (PDF or DOCX) to begin analysis")
        
        with st.expander("📖 What can this analyze?"):
            st.markdown("""
            **This tool can analyze:**
            - Blood test results
            - Urine test results
            - Lab reports
            - Medical records
            - Test outcomes
            
            **The AI will provide:**
            - Patient-friendly summary
            - Key medical findings
            - Risk indicators
            - Follow-up suggestions
            """)


def handle_image_analysis():
    """Handle medical image analysis"""
    st.markdown("## 🔬 Medical Image Analysis")
    st.markdown("Upload a medical image (X-ray, CT, MRI) for AI-powered diagnosis")
    
    st.markdown("---")
    
    # Image uploader
    uploaded_image = st.file_uploader(
        "Choose a medical image",
        type=['jpg', 'jpeg', 'png'],
        help=f"Upload a JPEG or PNG file (max {MAX_FILE_SIZE_MB}MB)",
        key="img_uploader"
    )
    
    if uploaded_image is not None:
        # Display image preview
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 🖼️ Image Preview")
            image = Image.open(uploaded_image)
            st.image(image, caption=uploaded_image.name)
        
        with col2:
            st.markdown("### ⚙️ Analysis Settings")
            
            # Image type selector
            image_type = st.selectbox(
                "Select Image Type",
                options=[
                    "auto",
                    "chest_xray",
                    "ct_brain",
                    "bone_xray",
                    "mri",
                    "ultrasound"
                ],
                format_func=lambda x: {
                    "auto": "🔍 Auto-detect",
                    "chest_xray": "🫁 Chest X-ray",
                    "ct_brain": "🧠 Brain CT Scan",
                    "bone_xray": "🦴 Bone X-ray",
                    "mri": "🔬 MRI Scan",
                    "ultrasound": "📡 Ultrasound"
                }[x],
                key="image_type"
            )
            
            # Clinical context
            clinical_context = st.text_area(
                "Clinical Context (Optional)",
                placeholder="Enter patient symptoms, history, or reason for imaging...",
                help="Providing clinical context helps improve analysis accuracy",
                key="clinical_context"
            )
            
            # File info
            file_size_mb = len(uploaded_image.getvalue()) / (1024 * 1024)
            st.info(f"**📊 File Size:** {file_size_mb:.2f} MB")
            
            if file_size_mb > MAX_FILE_SIZE_MB:
                st.error(f"❌ File size exceeds {MAX_FILE_SIZE_MB} MB limit")
                return
        
        st.markdown("---")
        
        # Analyze button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔬 Analyze Image", type="primary", use_container_width=True, key="analyze_img"):
                uploaded_image.seek(0)
                results = upload_and_analyze_image(
                    uploaded_image,
                    image_type,
                    clinical_context,
                    API_BASE_URL
                )
                
                if results:
                    st.markdown("---")
                    st.markdown("## 📊 Analysis Results")
                    display_image_results(results)
    else:
        st.info("👆 Please upload a medical image (JPEG or PNG) to begin analysis")
        
        with st.expander("🔬 Supported Image Types"):
            st.markdown("""
            **This tool can analyze:**
            - **Chest X-rays:** Pneumonia, lung cancer, heart problems
            - **Brain CT:** Stroke, bleeding, tumors
            - **Bone X-rays:** Fractures, arthritis, dislocations
            - **MRI Scans:** Tumors, disc herniation, MS
            - **Ultrasound:** Gallstones, pregnancy, organ issues
            
            **The AI will provide:**
            - Primary diagnosis
            - Specific issues identified
            - Follow-up suggestions
            - Confidence level
            - Urgency assessment
            """)


def handle_multimodal_analysis():
    """Handle multi-modal analysis (report + image)"""
    st.markdown("## 🔗 Multi-Modal Analysis")
    st.markdown("Upload both lab report and medical image for comprehensive integrated analysis")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    # Report uploader
    with col1:
        st.markdown("### 📄 Lab Report")
        report_file = st.file_uploader(
            "Upload Lab Report (Optional)",
            type=['pdf', 'docx'],
            help="Upload a PDF or DOCX lab report",
            key="multimodal_report"
        )
        
        if report_file:
            file_size_mb = len(report_file.getvalue()) / (1024 * 1024)
            st.success(f"✅ {report_file.name} ({file_size_mb:.2f} MB)")
    
    # Image uploader
    with col2:
        st.markdown("### 🔬 Medical Image")
        image_file = st.file_uploader(
            "Upload Medical Image (Optional)",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a JPEG or PNG medical image",
            key="multimodal_image"
        )
        
        if image_file:
            file_size_mb = len(image_file.getvalue()) / (1024 * 1024)
            st.success(f"✅ {image_file.name} ({file_size_mb:.2f} MB)")
            
            # Show image preview
            image = Image.open(image_file)
            st.image(image, caption="Image Preview")
    
    # Clinical context
    st.markdown("### 📝 Clinical Context")
    clinical_context = st.text_area(
        "Enter clinical information (Optional)",
        placeholder="Patient symptoms, medical history, reason for testing...",
        help="Providing context helps correlate findings between report and image",
        key="multimodal_context"
    )
    
    st.markdown("---")
    
    # Validate at least one file is uploaded
    if not report_file and not image_file:
        st.warning("⚠️ Please upload at least one file (report or image) for analysis")
        
        with st.expander("💡 How Multi-Modal Analysis Works"):
            st.markdown("""
            **Multi-modal analysis combines:**
            1. Lab report findings (blood tests, biomarkers)
            2. Medical image findings (X-rays, scans)
            3. Clinical context (symptoms, history)
            
            **Benefits:**
            - More accurate diagnosis
            - Identifies correlations
            - Comprehensive assessment
            - Integrated recommendations
            
            **Example:**
            - High WBC count (from report) + Lung infiltrate (from X-ray) = Pneumonia diagnosis
            """)
    else:
        # Analyze button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔗 Analyze Multi-Modal", type="primary", use_container_width=True, key="analyze_multi"):
                if report_file:
                    report_file.seek(0)
                if image_file:
                    image_file.seek(0)
                
                results = upload_and_analyze_multimodal(
                    report_file,
                    image_file,
                    clinical_context,
                    API_BASE_URL
                )
                
                if results:
                    st.markdown("---")
                    st.markdown("## 📊 Multi-Modal Analysis Results")
                    display_multimodal_results(results)



def handle_blood_group_prediction():
    """Handle blood group prediction from fingerprint"""
    st.markdown("## 🩸 Blood Group Prediction from Fingerprint")
    st.markdown("Upload a fingerprint image to predict blood group using AI")
    
    st.markdown("---")
    
    # File uploader
    uploaded_fingerprint = st.file_uploader(
        "Upload Fingerprint Image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear fingerprint image (JPEG or PNG)",
        key="fingerprint_uploader"
    )
    
    if uploaded_fingerprint is not None:
        # Display file info and image preview
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 📸 Uploaded Image")
            # Display the uploaded image
            image = Image.open(uploaded_fingerprint)
            st.image(image, caption="Fingerprint Image", width=300)
            
            # File info
            file_size_mb = len(uploaded_fingerprint.getvalue()) / (1024 * 1024)
            st.info(f"**Filename:** {uploaded_fingerprint.name}")
            st.info(f"**Size:** {file_size_mb:.2f} MB")
            st.info(f"**Dimensions:** {image.size[0]} x {image.size[1]}")
        
        with col2:
            st.markdown("### 🔬 Prediction Results")
            
            # Analyze button
            if st.button("🩸 Predict Blood Group", type="primary", use_container_width=True):
                with st.spinner("🔄 Analyzing fingerprint pattern..."):
                    try:
                        import requests
                        import time
                        
                        start_time = time.time()
                        
                        # Reset file pointer
                        uploaded_fingerprint.seek(0)
                        
                        # Prepare the file for upload
                        files = {
                            'fingerprint': (
                                uploaded_fingerprint.name,
                                uploaded_fingerprint.getvalue(),
                                uploaded_fingerprint.type
                            )
                        }
                        
                        # Make API request
                        response = requests.post(
                            f"{API_BASE_URL}/predict-blood-group",
                            files=files,
                            timeout=60
                        )
                        
                        elapsed_time = time.time() - start_time
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Display prediction
                            st.success("✅ Prediction Complete!")
                            
                            # Main prediction
                            st.markdown("---")
                            st.markdown("### 🎯 Predicted Blood Group")
                            
                            # Large display of predicted blood group
                            st.markdown(
                                f"<div style='text-align: center; padding: 30px; "
                                f"background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); "
                                f"border-radius: 15px; margin: 20px 0;'>"
                                f"<h1 style='color: white; font-size: 72px; margin: 0;'>"
                                f"{result['predicted_blood_group']}</h1>"
                                f"<p style='color: white; font-size: 24px; margin: 10px 0;'>"
                                f"Confidence: {result['confidence']}</p>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                            
                            # Disclaimer
                            st.markdown("---")
                            st.warning(
                                f"⚠️ **Medical Disclaimer:** {result['disclaimer']}"
                            )
                            
                        else:
                            st.error(f"❌ Prediction failed: {response.text}")
                            
                    except requests.exceptions.Timeout:
                        st.error("⏱️ Request timed out. Please try again.")
                    except requests.exceptions.ConnectionError:
                        st.error("🔌 Cannot connect to backend. Please ensure the API is running.")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    else:
        # Simple message when no file is uploaded
        st.info("👆 Please upload a fingerprint image to begin prediction")
