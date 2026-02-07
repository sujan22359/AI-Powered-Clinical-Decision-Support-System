# FastAPI main application entry point
import io
import logging
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, status, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.config import Config
from backend.services.analysis_engine import AnalysisEngine, AnalysisEngineError
from backend.services.document_parser import DocumentParsingError
from backend.services.llm_service import LLMServiceError
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

# Startup event to validate configuration
@app.on_event("startup")
async def startup_event():
    """Validate configuration on startup with detailed error reporting"""
    global analysis_engine
    try:
        logger.info("Starting Clinical Report Analyzer API...")
        
        # Validate configuration
        Config.validate_config()
        logger.info("Configuration validation passed")
        
        # Initialize analysis engine after config validation
        analysis_engine = AnalysisEngine()
        logger.info("Analysis engine initialized successfully")
        
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