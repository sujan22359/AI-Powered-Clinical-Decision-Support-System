# FastAPI main application entry point
import io
import logging
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, File, UploadFile, HTTPException, status, Request, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.config import Config
from backend.services.analysis_engine import AnalysisEngine, AnalysisEngineError
from backend.services.document_parser import DocumentParsingError
from backend.services.gemini_vision import GeminiMedicalImageAnalyzer
from backend.utils.logger import setup_logger
from backend.utils.error_handlers import error_handler, ErrorCategory, ErrorResponse

# Initialize logger
logger = setup_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Clinical Report Analyzer API",
    description="AI-powered clinical report analysis for informational purposes only",
    version="1.0.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analysis engine (will be initialized on startup)
analysis_engine = None
vision_analyzer = None
blood_group_predictor = None  # Cache blood group predictor

@app.get("/health")
async def health_check():
    """Health check endpoint with detailed status information"""
    try:
        # Check if analysis engine is properly initialized
        engine_status = "healthy" if analysis_engine is not None else "not_initialized"
        
        # Get additional system information
        health_info = {
            "status": "healthy",
            "service": "clinical-report-analyzer",
            "version": "1.0.0",
            "analysis_engine": engine_status,
            "supported_formats": list(Config.ALLOWED_EXTENSIONS),
            "max_file_size_mb": Config.MAX_FILE_SIZE_MB
        }
        
        # Add engine info if available
        if analysis_engine:
            try:
                engine_info = analysis_engine.get_engine_info()
                health_info["engine_info"] = engine_info
            except Exception as e:
                logger.warning(f"Could not get engine info: {str(e)}")
                health_info["engine_warning"] = "Engine info unavailable"
        
        return health_info
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "service": "clinical-report-analyzer",
                "error": "Health check failed"
            }
        )

@app.post("/analyze")
async def analyze_document(request: Request, file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Analyze uploaded clinical document and return structured insights.
    
    Args:
        request: FastAPI request object for error tracking
        file: Uploaded clinical document (PDF or DOCX)
        
    Returns:
        Structured JSON response with analysis results
        
    Raises:
        HTTPException: For various error conditions (handled by error handler)
    """
    context = {"endpoint": "analyze_document", "filename": file.filename}
    
    try:
        logger.info(f"Received document upload: {file.filename}")
        
        # Check if analysis engine is initialized
        if analysis_engine is None:
            error_response = ErrorResponse(
                error_code="SERVICE_NOT_CONFIGURED",
                message="Analysis service is not properly configured",
                category=ErrorCategory.SYSTEM,
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                details="The analysis engine failed to initialize. Please check API configuration.",
                suggestions=[
                    "Verify that the GEMINI_API_KEY environment variable is set",
                    "Check system logs for configuration errors",
                    "Contact support if the issue persists"
                ]
            )
            return JSONResponse(
                status_code=error_response.status_code,
                content=error_response.to_dict()
            )
        
        # Input validation with detailed error handling
        try:
            _validate_upload_file(file)
        except HTTPException as e:
            # Let the error handler process validation errors
            raise e
        
        # Read file content with size validation
        try:
            file_content = await file.read()
        except Exception as e:
            logger.error(f"Failed to read uploaded file {file.filename}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to read uploaded file: {str(e)}"
            )
        
        # Validate file size
        if len(file_content) > Config.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size ({len(file_content) / (1024*1024):.1f}MB) exceeds maximum limit of {Config.MAX_FILE_SIZE_MB}MB"
            )
        
        # Validate file content is not empty
        if len(file_content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty. Please upload a file with content."
            )
        
        # Add file info to context for error tracking
        context.update({
            "file_size_bytes": len(file_content),
            "file_size_mb": round(len(file_content) / (1024*1024), 2)
        })
        
        # Analyze the document
        logger.debug(f"Starting analysis for {file.filename}")
        analysis_result = analysis_engine.analyze_document(file_content, file.filename)
        
        # Convert to dictionary format for JSON response
        result_dict = analysis_engine.to_dict(analysis_result)
        
        logger.info(f"Successfully analyzed document: {file.filename}")
        
        return {
            "success": True,
            "filename": file.filename,
            "file_size_mb": context["file_size_mb"],
            "analysis": result_dict,
            "message": "Document analyzed successfully",
            "processing_time": result_dict.get("analysis_timestamp")
        }
        
    except Exception as e:
        # Use comprehensive error handler for all exceptions
        return error_handler.handle_error(e, request, context)

def _validate_upload_file(file: UploadFile) -> None:
    """
    Validate uploaded file meets requirements with detailed error messages.
    
    Args:
        file: Uploaded file to validate
        
    Raises:
        HTTPException: If validation fails with specific error details
    """
    # Check if filename is provided
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required. Please ensure your file has a valid name."
        )
    
    # Check file extension
    file_extension = file.filename.lower().split('.')[-1] if '.' in file.filename else ''
    if f".{file_extension}" not in Config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file format '.{file_extension}'. Supported formats: {', '.join(Config.ALLOWED_EXTENSIONS)}. Please convert your document to PDF or DOCX format."
        )
    
    # Check content type if provided (with more detailed logging)
    if file.content_type:
        allowed_content_types = {
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword"
        }
        
        if file.content_type not in allowed_content_types:
            logger.warning(
                f"Content type mismatch for {file.filename}: "
                f"received '{file.content_type}', expected one of {allowed_content_types}"
            )
            # Don't raise error for content type mismatch, just log warning
            # Some browsers may send incorrect content types
    
    # Additional filename validation
    if len(file.filename) > 255:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is too long. Please use a shorter filename (maximum 255 characters)."
        )
    
    # Check for potentially problematic characters in filename
    import re
    if re.search(r'[<>:"/\\|?*]', file.filename):
        logger.warning(f"Filename contains special characters: {file.filename}")
        # Don't block, but log for monitoring

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors with comprehensive logging"""
    logger.error(f"Unhandled exception in global handler: {str(exc)}")
    
    # Use comprehensive error handler
    return error_handler.handle_error(exc, request, {"handler": "global_exception_handler"})


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP exception handler with enhanced error details"""
    # Use comprehensive error handler for HTTP exceptions too
    return error_handler.handle_error(exc, request, {"handler": "http_exception_handler"})


@app.post("/analyze-image")
async def analyze_medical_image(
    request: Request,
    file: UploadFile = File(...),
    image_type: str = Form("auto"),
    clinical_context: Optional[str] = Form(None)
) -> Dict[str, Any]:
    """
    Analyze uploaded medical image and provide diagnosis with suggestions
    
    Args:
        request: FastAPI request object
        file: Uploaded medical image (JPEG, PNG, DICOM)
        image_type: Type of medical image (chest_xray, ct_brain, bone_xray, mri, ultrasound, auto)
        clinical_context: Optional clinical information (symptoms, patient history)
    
    Returns:
        Structured JSON response with:
        - diagnosis: Main diagnosis
        - issues: List of identified issues
        - suggestions: Follow-up recommendations
        - confidence: Confidence level
        - urgency: Urgency level
    """
    context = {"endpoint": "analyze_image", "filename": file.filename, "image_type": image_type}
    
    try:
        logger.info(f"Received medical image upload: {file.filename}, type: {image_type}")
        
        # Check if vision analyzer is initialized
        if vision_analyzer is None:
            error_response = ErrorResponse(
                error_code="SERVICE_NOT_CONFIGURED",
                message="Vision analysis service is not properly configured",
                category=ErrorCategory.SYSTEM,
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                details="The vision analyzer failed to initialize. Please check API configuration.",
                suggestions=[
                    "Verify that the GEMINI_API_KEY environment variable is set",
                    "Check system logs for configuration errors",
                    "Contact support if the issue persists"
                ]
            )
            return JSONResponse(
                status_code=error_response.status_code,
                content=error_response.to_dict()
            )
        
        # Validate image file
        _validate_image_file(file)
        
        # Read file content
        try:
            file_content = await file.read()
        except Exception as e:
            logger.error(f"Failed to read uploaded image {file.filename}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to read uploaded image: {str(e)}"
            )
        
        # Validate file size (max 10MB for images)
        max_image_size_mb = 10
        if len(file_content) > max_image_size_mb * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image size ({len(file_content) / (1024*1024):.1f}MB) exceeds maximum limit of {max_image_size_mb}MB"
            )
        
        # Validate file content is not empty
        if len(file_content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded image is empty. Please upload a valid image file."
            )
        
        # Add file info to context
        context.update({
            "file_size_bytes": len(file_content),
            "file_size_mb": round(len(file_content) / (1024*1024), 2)
        })
        
        # Analyze the medical image
        logger.debug(f"Starting image analysis for {file.filename}")
        analysis_result = vision_analyzer.analyze_medical_image(
            image_data=file_content,
            image_type=image_type,
            clinical_context=clinical_context
        )
        
        # Check if analysis was successful
        if not analysis_result.get("success", False):
            error_msg = analysis_result.get("error", "Unknown error during image analysis")
            logger.error(f"Image analysis failed: {error_msg}")
            raise AnalysisEngineError(f"Image analysis failed: {error_msg}")
        
        logger.info(f"Successfully analyzed image: {file.filename}")
        
        return {
            "success": True,
            "filename": file.filename,
            "file_size_mb": context["file_size_mb"],
            "image_type": image_type,
            "diagnosis": analysis_result["diagnosis"],
            "issues": analysis_result["issues"],
            "suggestions": analysis_result["suggestions"],
            "confidence": analysis_result["confidence"],
            "urgency": analysis_result["urgency"],
            "findings": analysis_result.get("findings", {}),
            "message": "Medical image analyzed successfully"
        }
        
    except Exception as e:
        return error_handler.handle_error(e, request, context)


@app.post("/analyze-multimodal")
async def analyze_multimodal(
    request: Request,
    report: UploadFile = File(None),
    image: UploadFile = File(None),
    clinical_context: Optional[str] = Form(None)
) -> Dict[str, Any]:
    """
    Analyze both lab report and medical image together for comprehensive diagnosis
    
    Args:
        request: FastAPI request object
        report: Optional lab report (PDF/DOCX)
        image: Optional medical image (JPEG/PNG)
        clinical_context: Optional clinical information
    
    Returns:
        Combined analysis with correlations between lab and imaging findings
    """
    context = {"endpoint": "analyze_multimodal"}
    
    try:
        logger.info("Received multi-modal analysis request")
        
        # Validate at least one input is provided
        if not report and not image:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Please provide at least one file (report or image) for analysis"
            )
        
        results = {
            "success": True,
            "analysis_type": "multimodal",
            "report_analysis": None,
            "image_analysis": None,
            "correlation": None
        }
        
        # Analyze report if provided
        if report:
            logger.info(f"Analyzing report: {report.filename}")
            report_content = await report.read()
            
            if analysis_engine:
                report_result = analysis_engine.analyze_document(report_content, report.filename)
                results["report_analysis"] = analysis_engine.to_dict(report_result)
            else:
                results["report_analysis"] = {"error": "Report analysis service not available"}
        
        # Analyze image if provided
        if image:
            logger.info(f"Analyzing image: {image.filename}")
            image_content = await image.read()
            
            if vision_analyzer:
                image_result = vision_analyzer.analyze_medical_image(
                    image_data=image_content,
                    image_type="auto",
                    clinical_context=clinical_context
                )
                results["image_analysis"] = image_result
            else:
                results["image_analysis"] = {"error": "Image analysis service not available"}
        
        # Correlate findings if both are available
        if results["report_analysis"] and results["image_analysis"]:
            results["correlation"] = _correlate_findings(
                results["report_analysis"],
                results["image_analysis"]
            )
        
        logger.info("Multi-modal analysis completed successfully")
        return results
        
    except Exception as e:
        return error_handler.handle_error(e, request, context)


@app.get("/supported-image-types")
async def get_supported_image_types():
    """Get list of supported medical image types"""
    if vision_analyzer:
        return {
            "success": True,
            "supported_types": vision_analyzer.get_supported_image_types()
        }
    else:
        return {
            "success": False,
            "error": "Vision analyzer not initialized",
            "supported_types": []
        }


def _validate_image_file(file: UploadFile) -> None:
    """
    Validate uploaded image file
    
    Args:
        file: Uploaded image file
        
    Raises:
        HTTPException: If validation fails
    """
    # Check if filename is provided
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required. Please ensure your image has a valid name."
        )
    
    # Check file extension
    allowed_image_extensions = [".jpg", ".jpeg", ".png", ".dcm", ".dicom"]
    file_extension = file.filename.lower().split('.')[-1] if '.' in file.filename else ''
    
    if f".{file_extension}" not in allowed_image_extensions:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image format '.{file_extension}'. Supported formats: {', '.join(allowed_image_extensions)}"
        )
    
    # Check content type if provided
    if file.content_type:
        allowed_content_types = {
            "image/jpeg",
            "image/jpg",
            "image/png",
            "application/dicom"
        }
        
        if file.content_type not in allowed_content_types:
            logger.warning(
                f"Content type mismatch for {file.filename}: "
                f"received '{file.content_type}'"
            )


def _correlate_findings(report_analysis: Dict, image_analysis: Dict) -> Dict:
    """
    Correlate findings from lab report and medical image
    
    Args:
        report_analysis: Lab report analysis results
        image_analysis: Medical image analysis results
    
    Returns:
        Dictionary with correlated findings and integrated diagnosis
    """
    correlation = {
        "integrated_diagnosis": "",
        "correlations": [],
        "confidence": 0,
        "recommendations": []
    }
    
    try:
        # Extract risk indicators from report
        report_risks = report_analysis.get("risk_indicators", [])
        
        # Extract issues from image
        image_issues = image_analysis.get("issues", [])
        
        # Simple correlation logic (can be enhanced with ML)
        correlations_found = []
        
        # Example correlations
        correlation_rules = {
            ("glucose", "retinal"): "Diabetic retinopathy - High glucose correlates with retinal changes",
            ("wbc", "infiltrate"): "Bacterial infection - Elevated WBC with lung infiltrate suggests pneumonia",
            ("cholesterol", "heart"): "Cardiovascular risk - High cholesterol with cardiac findings",
            ("hemoglobin", "spleen"): "Possible anemia - Low hemoglobin with splenomegaly"
        }
        
        # Check for correlations
        for (lab_key, image_key), diagnosis in correlation_rules.items():
            # Check if lab finding contains the key
            lab_match = any(lab_key.lower() in str(risk).lower() for risk in report_risks)
            # Check if image finding contains the key
            image_match = any(image_key.lower() in str(issue).lower() for issue in image_issues)
            
            if lab_match and image_match:
                correlations_found.append({
                    "type": "positive_correlation",
                    "diagnosis": diagnosis,
                    "confidence": 85
                })
        
        correlation["correlations"] = correlations_found
        
        # Generate integrated diagnosis
        if correlations_found:
            correlation["integrated_diagnosis"] = correlations_found[0]["diagnosis"]
            correlation["confidence"] = correlations_found[0]["confidence"]
            correlation["recommendations"] = [
                "Consult specialist for integrated treatment plan",
                "Monitor both lab values and imaging findings",
                "Follow-up testing recommended"
            ]
        else:
            correlation["integrated_diagnosis"] = "No significant correlations found between lab and imaging findings"
            correlation["confidence"] = 50
            correlation["recommendations"] = [
                "Review findings independently",
                "Consult healthcare provider for comprehensive assessment"
            ]
        
    except Exception as e:
        logger.error(f"Error correlating findings: {str(e)}")
        correlation["error"] = "Correlation analysis failed"
    
    return correlation


@app.post("/predict-blood-group")
async def predict_blood_group(
    request: Request,
    fingerprint: UploadFile = File(..., description="Fingerprint image (JPEG/PNG)")
) -> Dict[str, Any]:
    """
    Predict blood group from fingerprint image using trained ML model.
    
    Args:
        fingerprint: Fingerprint image file (JPEG or PNG)
    
    Returns:
        Dictionary containing prediction results with blood group and confidence
    
    Raises:
        HTTPException: If prediction fails or file is invalid
    """
    logger.info(f"Blood group prediction request from {request.client.host}")
    
    try:
        # Validate file type
        if not fingerprint.content_type or not fingerprint.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file type. Please upload an image file (JPEG or PNG)."
            )
        
        # Read file content
        file_content = await fingerprint.read()
        
        if len(file_content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file uploaded"
            )
        
        # Log file info
        logger.info(
            f"Processing fingerprint image: {fingerprint.filename}, "
            f"size: {len(file_content)} bytes, type: {fingerprint.content_type}"
        )
        
        # Use cached predictor (initialized at startup)
        global blood_group_predictor
        if blood_group_predictor is None:
            from backend.services.blood_group_predictor import BloodGroupPredictor
            blood_group_predictor = BloodGroupPredictor(model_type="pytorch_cnn", enable_gradcam=True)
        
        # Make prediction
        result = blood_group_predictor.predict(file_content)
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {result.get('error', 'Unknown error')}"
            )
        
        # Format response
        response = {
            "success": True,
            "filename": fingerprint.filename,
            "predicted_blood_group": result["predicted_blood_group"],
            "confidence": result["confidence_percent"],
            "confidence_score": result["confidence"],
            "all_probabilities": result["probabilities"],
            "top_3_predictions": [
                {
                    "blood_group": bg,
                    "probability": f"{prob * 100:.2f}%",
                    "probability_score": prob
                }
                for bg, prob in result["top_3_predictions"]
            ],
            "model_info": {
                "model_type": result["model_type"],
                "supported_blood_groups": blood_group_predictor.BLOOD_GROUPS
            },
            "gradcam": {
                "available": result.get("gradcam_available", False),
                "image": result.get("gradcam_image", None)  # Base64 encoded image
            },
            "disclaimer": (
                "IMPORTANT: This prediction is for informational purposes only and should not "
                "replace laboratory blood typing. Always confirm blood group through proper "
                "medical testing before any medical procedures."
            )
        }
        
        logger.info(
            f"Blood group prediction successful: {result['predicted_blood_group']} "
            f"(confidence: {result['confidence_percent']})"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Blood group prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Blood group prediction failed: {str(e)}"
        )


# Startup event to validate configuration
@app.on_event("startup")
async def startup_event():
    """Validate configuration on startup with detailed error reporting"""
    global analysis_engine, vision_analyzer, blood_group_predictor
    try:
        logger.info("Starting Clinical Report Analyzer API...")
        
        # Validate configuration
        Config.validate_config()
        logger.info("Configuration validation passed")
        
        # Initialize analysis engine after config validation
        analysis_engine = AnalysisEngine()
        logger.info("Analysis engine initialized successfully")
        
        # Initialize vision analyzer with Gemini
        try:
            vision_analyzer = GeminiMedicalImageAnalyzer()
            logger.info("✅ Vision analyzer initialized with Gemini AI")
        except Exception as e:
            logger.error(f"Failed to initialize vision analyzer: {str(e)}")
            vision_analyzer = None
        
        # Preload blood group predictor for faster first prediction
        try:
            from backend.services.blood_group_predictor import BloodGroupPredictor
            blood_group_predictor = BloodGroupPredictor(model_type="pytorch_cnn", enable_gradcam=False)
            logger.info("Blood group predictor preloaded successfully")
        except Exception as e:
            logger.warning(f"Failed to preload blood group predictor: {str(e)}")
            blood_group_predictor = None
        
        # Validate LLM service connection
        if hasattr(analysis_engine.llm_service, 'validate_api_connection'):
            try:
                if analysis_engine.llm_service.validate_api_connection():
                    logger.info("LLM service connection validated")
                else:
                    logger.warning("LLM service connection validation failed - service may be unavailable")
            except Exception as e:
                logger.warning(f"Could not validate LLM service connection: {str(e)}")
        
        logger.info("Clinical Report Analyzer API started successfully")
        
    except ValueError as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        # Don't raise here to allow the app to start but log the error
        # The analyze endpoint will handle the missing configuration gracefully
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        # Don't raise here to allow the app to start but log the error


@app.get("/status")
async def get_system_status():
    """Get detailed system status and configuration information"""
    try:
        status_info = {
            "service": "clinical-report-analyzer",
            "version": "1.0.0",
            "status": "operational",
            "components": {
                "analysis_engine": "healthy" if analysis_engine is not None else "not_initialized",
                "configuration": "valid" if Config.GEMINI_API_KEY else "invalid"
            },
            "capabilities": {
                "supported_formats": list(Config.ALLOWED_EXTENSIONS),
                "max_file_size_mb": Config.MAX_FILE_SIZE_MB,
                "api_host": Config.API_HOST,
                "api_port": Config.API_PORT,
                "log_level": Config.LOG_LEVEL
            }
        }
        
        # Add engine details if available
        if analysis_engine:
            try:
                engine_info = analysis_engine.get_engine_info()
                status_info["engine_details"] = engine_info
            except Exception as e:
                logger.warning(f"Could not get engine details: {str(e)}")
                status_info["components"]["analysis_engine"] = "partially_available"
        
        # Check overall system health
        if not Config.GEMINI_API_KEY:
            status_info["status"] = "degraded"
            status_info["warnings"] = ["GEMINI_API_KEY not configured"]
        elif analysis_engine is None:
            status_info["status"] = "degraded"
            status_info["warnings"] = ["Analysis engine not initialized"]
        
        return status_info
        
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "service": "clinical-report-analyzer",
                "status": "error",
                "error": "Status check failed"
            }
        )


@app.get("/error-codes")
async def get_error_codes():
    """Get information about API error codes and their meanings"""
    return {
        "error_codes": {
            "DOCUMENT_EMPTY": {
                "description": "The uploaded document appears to be empty or contains no readable text",
                "category": "parsing",
                "suggestions": ["Ensure the document contains readable text", "Try uploading a different document"]
            },
            "DOCUMENT_CORRUPTED": {
                "description": "The uploaded document appears to be corrupted or invalid",
                "category": "parsing", 
                "suggestions": ["Try uploading the document again", "Ensure the file is not corrupted"]
            },
            "UNSUPPORTED_FORMAT": {
                "description": "The uploaded file format is not supported",
                "category": "validation",
                "suggestions": ["Use supported formats: PDF (.pdf) or Word (.docx)"]
            },
            "FILE_TOO_LARGE": {
                "description": "The uploaded file exceeds the maximum size limit",
                "category": "validation",
                "suggestions": [f"Reduce file size to under {Config.MAX_FILE_SIZE_MB}MB"]
            },
            "AI_SERVICE_UNAVAILABLE": {
                "description": "The AI analysis service is temporarily unavailable",
                "category": "ai_service",
                "suggestions": ["Please try again in a few minutes"]
            },
            "AI_RATE_LIMIT_EXCEEDED": {
                "description": "Too many requests to the AI service",
                "category": "rate_limit",
                "suggestions": ["Please wait a moment and try again"]
            },
            "AI_CONTENT_BLOCKED": {
                "description": "The document content was flagged by safety filters",
                "category": "ai_service",
                "suggestions": ["Try uploading a different document with appropriate medical content"]
            },
            "VALIDATION_ERROR": {
                "description": "Request validation failed",
                "category": "validation",
                "suggestions": ["Check your input parameters and ensure all required fields are provided"]
            },
            "INTERNAL_SERVER_ERROR": {
                "description": "An unexpected error occurred on the server",
                "category": "system",
                "suggestions": ["Please try again later", "Contact support if the issue persists"]
            }
        },
        "categories": {
            "parsing": "Errors related to document parsing and text extraction",
            "validation": "Errors related to input validation and file requirements",
            "ai_service": "Errors related to AI processing and analysis",
            "rate_limit": "Errors related to request rate limiting",
            "system": "General system and server errors"
        }
    }


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Clinical Report Analyzer API shutting down...")
    # Add any cleanup logic here if needed



# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

# Add measurement extraction endpoint
@app.post("/extract-measurements")
async def extract_measurements(request: Request, text: str = Form(...), image_type: str = Form("auto")) -> Dict:
    """Extract measurements from text"""
    try:
        from backend.services.measurement_extractor import MeasurementExtractor
        extractor = MeasurementExtractor()
        
        measurements = extractor.extract_measurements(text, image_type)
        
        return {
            "success": True,
            "measurements": measurements,
            "message": f"Extracted {measurements.get('count', 0)} measurements"
        }
    except Exception as e:
        logger.error(f"Measurement extraction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
