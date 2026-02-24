"""
Error handling utilities for Clinical Report Analyzer API

This module provides comprehensive error handling with proper HTTP status codes,
descriptive error messages, and logging integration for debugging.
"""

import logging
import traceback
from typing import Dict, Any, Optional
from enum import Enum

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

from backend.services.document_parser import DocumentParsingError
from backend.services.analysis_engine import AnalysisEngineError
from backend.utils.logger import setup_logger


class ErrorCategory(Enum):
    """Error categories for classification and handling"""
    VALIDATION = "validation"
    PARSING = "parsing"
    AI_SERVICE = "ai_service"
    ANALYSIS = "analysis"
    SYSTEM = "system"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    FILE_SIZE = "file_size"
    UNSUPPORTED_FORMAT = "unsupported_format"


class ErrorResponse:
    """Standardized error response format"""
    
    def __init__(
        self,
        error_code: str,
        message: str,
        category: ErrorCategory,
        status_code: int,
        details: Optional[str] = None,
        suggestions: Optional[list] = None,
        request_id: Optional[str] = None
    ):
        self.error_code = error_code
        self.message = message
        self.category = category
        self.status_code = status_code
        self.details = details
        self.suggestions = suggestions or []
        self.request_id = request_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error response to dictionary format"""
        response = {
            "success": False,
            "error": {
                "code": self.error_code,
                "message": self.message,
                "category": self.category.value,
                "details": self.details,
                "suggestions": self.suggestions
            }
        }
        
        if self.request_id:
            response["request_id"] = self.request_id
        
        return response


class ErrorHandler:
    """Comprehensive error handler for the Clinical Report Analyzer API"""
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        
        # Error mapping configuration
        self.error_mappings = {
            # Document parsing errors
            DocumentParsingError: self._handle_document_parsing_error,
            
            # Analysis engine errors
            AnalysisEngineError: self._handle_analysis_engine_error,
            
            # HTTP exceptions
            HTTPException: self._handle_http_exception,
            
            # Generic exceptions
            ValueError: self._handle_validation_error,
            FileNotFoundError: self._handle_file_not_found_error,
            PermissionError: self._handle_permission_error,
            MemoryError: self._handle_memory_error,
            TimeoutError: self._handle_timeout_error,
        }
    
    def handle_error(
        self, 
        error: Exception, 
        request: Optional[Request] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> JSONResponse:
        """
        Handle any error and return appropriate JSON response.
        
        Args:
            error: Exception that occurred
            request: FastAPI request object (optional)
            context: Additional context information (optional)
            
        Returns:
            JSONResponse with error details
        """
        # Generate request ID for tracking
        request_id = self._generate_request_id(request)
        
        # Get error handler for this exception type
        handler = self._get_error_handler(error)
        
        # Handle the error
        error_response = handler(error, request, context, request_id)
        
        # Log the error with appropriate level
        self._log_error(error, error_response, request, context)
        
        # Return JSON response
        return JSONResponse(
            status_code=error_response.status_code,
            content=error_response.to_dict()
        )
    
    def _get_error_handler(self, error: Exception):
        """Get appropriate error handler for exception type"""
        error_type = type(error)
        
        # Check for exact match first
        if error_type in self.error_mappings:
            return self.error_mappings[error_type]
        
        # Check for inheritance match
        for exception_type, handler in self.error_mappings.items():
            if isinstance(error, exception_type):
                return handler
        
        # Default to generic handler
        return self._handle_generic_error
    
    def _handle_document_parsing_error(
        self, 
        error: DocumentParsingError, 
        request: Optional[Request],
        context: Optional[Dict[str, Any]],
        request_id: str
    ) -> ErrorResponse:
        """Handle document parsing errors"""
        error_message = str(error)
        
        # Determine specific error type and suggestions
        if "empty" in error_message.lower():
            suggestions = [
                "Ensure the document contains readable text",
                "Try uploading a different document",
                "Check if the document is corrupted"
            ]
            error_code = "DOCUMENT_EMPTY"
        elif "corrupted" in error_message.lower() or "invalid" in error_message.lower():
            suggestions = [
                "Try uploading the document again",
                "Ensure the file is not corrupted",
                "Try converting the document to a different format"
            ]
            error_code = "DOCUMENT_CORRUPTED"
        elif "unsupported" in error_message.lower():
            suggestions = [
                "Use supported formats: PDF (.pdf) or Word (.docx)",
                "Convert your document to PDF or DOCX format"
            ]
            error_code = "UNSUPPORTED_FORMAT"
        else:
            suggestions = [
                "Ensure the document is readable and not password-protected",
                "Try uploading a different document",
                "Contact support if the issue persists"
            ]
            error_code = "DOCUMENT_PARSING_FAILED"
        
        return ErrorResponse(
            error_code=error_code,
            message=f"Document parsing failed: {error_message}",
            category=ErrorCategory.PARSING,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=error_message,
            suggestions=suggestions,
            request_id=request_id
        )
    
    def _handle_analysis_engine_error(
        self, 
        error: AnalysisEngineError, 
        request: Optional[Request],
        context: Optional[Dict[str, Any]],
        request_id: str
    ) -> ErrorResponse:
        """Handle analysis engine errors"""
        error_message = str(error)
        
        suggestions = [
            "Please try uploading the document again",
            "Ensure the document contains valid medical content",
            "Contact support if the issue persists"
        ]
        
        return ErrorResponse(
            error_code="ANALYSIS_ENGINE_ERROR",
            message=f"Analysis processing failed: {error_message}",
            category=ErrorCategory.ANALYSIS,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=error_message,
            suggestions=suggestions,
            request_id=request_id
        )
    
    def _handle_http_exception(
        self, 
        error: HTTPException, 
        request: Optional[Request],
        context: Optional[Dict[str, Any]],
        request_id: str
    ) -> ErrorResponse:
        """Handle FastAPI HTTP exceptions"""
        # Map HTTP status codes to error categories and suggestions
        status_mappings = {
            400: {
                "category": ErrorCategory.VALIDATION,
                "error_code": "BAD_REQUEST",
                "suggestions": [
                    "Check your request parameters",
                    "Ensure all required fields are provided"
                ]
            },
            401: {
                "category": ErrorCategory.AUTHENTICATION,
                "error_code": "UNAUTHORIZED",
                "suggestions": [
                    "Authentication is required",
                    "Check your API credentials"
                ]
            },
            413: {
                "category": ErrorCategory.FILE_SIZE,
                "error_code": "FILE_TOO_LARGE",
                "suggestions": [
                    "Reduce the file size",
                    f"Maximum file size is 10MB"
                ]
            },
            415: {
                "category": ErrorCategory.UNSUPPORTED_FORMAT,
                "error_code": "UNSUPPORTED_MEDIA_TYPE",
                "suggestions": [
                    "Use supported formats: PDF (.pdf) or Word (.docx)",
                    "Check the file extension and format"
                ]
            },
            422: {
                "category": ErrorCategory.VALIDATION,
                "error_code": "UNPROCESSABLE_ENTITY",
                "suggestions": [
                    "Check the document content and format",
                    "Ensure the document is readable"
                ]
            },
            429: {
                "category": ErrorCategory.RATE_LIMIT,
                "error_code": "TOO_MANY_REQUESTS",
                "suggestions": [
                    "Please wait before making another request",
                    "Rate limit exceeded"
                ]
            },
            503: {
                "category": ErrorCategory.SYSTEM,
                "error_code": "SERVICE_UNAVAILABLE",
                "suggestions": [
                    "The service is temporarily unavailable",
                    "Please try again later"
                ]
            }
        }
        
        mapping = status_mappings.get(error.status_code, {
            "category": ErrorCategory.SYSTEM,
            "error_code": "HTTP_ERROR",
            "suggestions": ["An error occurred processing your request"]
        })
        
        return ErrorResponse(
            error_code=mapping["error_code"],
            message=error.detail,
            category=mapping["category"],
            status_code=error.status_code,
            details=error.detail,
            suggestions=mapping["suggestions"],
            request_id=request_id
        )
    
    def _handle_validation_error(
        self, 
        error: ValueError, 
        request: Optional[Request],
        context: Optional[Dict[str, Any]],
        request_id: str
    ) -> ErrorResponse:
        """Handle validation errors"""
        return ErrorResponse(
            error_code="VALIDATION_ERROR",
            message=f"Validation failed: {str(error)}",
            category=ErrorCategory.VALIDATION,
            status_code=status.HTTP_400_BAD_REQUEST,
            details=str(error),
            suggestions=[
                "Check your input parameters",
                "Ensure all required fields are provided correctly"
            ],
            request_id=request_id
        )
    
    def _handle_file_not_found_error(
        self, 
        error: FileNotFoundError, 
        request: Optional[Request],
        context: Optional[Dict[str, Any]],
        request_id: str
    ) -> ErrorResponse:
        """Handle file not found errors"""
        return ErrorResponse(
            error_code="FILE_NOT_FOUND",
            message="Requested file was not found",
            category=ErrorCategory.SYSTEM,
            status_code=status.HTTP_404_NOT_FOUND,
            details=str(error),
            suggestions=[
                "Ensure the file exists and is accessible",
                "Check the file path and permissions"
            ],
            request_id=request_id
        )
    
    def _handle_permission_error(
        self, 
        error: PermissionError, 
        request: Optional[Request],
        context: Optional[Dict[str, Any]],
        request_id: str
    ) -> ErrorResponse:
        """Handle permission errors"""
        return ErrorResponse(
            error_code="PERMISSION_DENIED",
            message="Permission denied accessing resource",
            category=ErrorCategory.SYSTEM,
            status_code=status.HTTP_403_FORBIDDEN,
            details=str(error),
            suggestions=[
                "Check file permissions",
                "Contact system administrator"
            ],
            request_id=request_id
        )
    
    def _handle_memory_error(
        self, 
        error: MemoryError, 
        request: Optional[Request],
        context: Optional[Dict[str, Any]],
        request_id: str
    ) -> ErrorResponse:
        """Handle memory errors"""
        return ErrorResponse(
            error_code="INSUFFICIENT_MEMORY",
            message="Insufficient memory to process request",
            category=ErrorCategory.SYSTEM,
            status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
            details="The system ran out of memory processing your request",
            suggestions=[
                "Try uploading a smaller document",
                "Please try again later when system resources are available"
            ],
            request_id=request_id
        )
    
    def _handle_timeout_error(
        self, 
        error: TimeoutError, 
        request: Optional[Request],
        context: Optional[Dict[str, Any]],
        request_id: str
    ) -> ErrorResponse:
        """Handle timeout errors"""
        return ErrorResponse(
            error_code="REQUEST_TIMEOUT",
            message="Request processing timed out",
            category=ErrorCategory.SYSTEM,
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            details=str(error),
            suggestions=[
                "The request took too long to process",
                "Try uploading a smaller document",
                "Please try again later"
            ],
            request_id=request_id
        )
    
    def _handle_generic_error(
        self, 
        error: Exception, 
        request: Optional[Request],
        context: Optional[Dict[str, Any]],
        request_id: str
    ) -> ErrorResponse:
        """Handle generic/unexpected errors"""
        return ErrorResponse(
            error_code="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred",
            category=ErrorCategory.SYSTEM,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details="An internal server error occurred while processing your request",
            suggestions=[
                "Please try again later",
                "Contact support if the issue persists"
            ],
            request_id=request_id
        )
    
    def _generate_request_id(self, request: Optional[Request]) -> str:
        """Generate unique request ID for tracking"""
        import uuid
        import time
        
        # Use request headers if available, otherwise generate UUID
        if request and hasattr(request, 'headers'):
            request_id = request.headers.get('X-Request-ID')
            if request_id:
                return request_id
        
        # Generate timestamp-based ID
        timestamp = int(time.time() * 1000)  # milliseconds
        unique_id = str(uuid.uuid4())[:8]
        return f"req_{timestamp}_{unique_id}"
    
    def _log_error(
        self, 
        error: Exception, 
        error_response: ErrorResponse, 
        request: Optional[Request],
        context: Optional[Dict[str, Any]]
    ):
        """Log error with appropriate level and context"""
        # Determine log level based on error severity
        if error_response.status_code >= 500:
            log_level = logging.ERROR
        elif error_response.status_code >= 400:
            log_level = logging.WARNING
        else:
            log_level = logging.INFO
        
        # Prepare log message
        log_data = {
            "request_id": error_response.request_id,
            "error_code": error_response.error_code,
            "error_category": error_response.category.value,
            "status_code": error_response.status_code,
            "error_message": str(error),
            "error_type": type(error).__name__
        }
        
        # Add request context if available
        if request:
            log_data.update({
                "method": getattr(request, 'method', 'UNKNOWN'),
                "url": str(getattr(request, 'url', 'UNKNOWN')),
                "user_agent": getattr(request, 'headers', {}).get('user-agent', 'UNKNOWN')
            })
        
        # Add additional context
        if context:
            log_data["context"] = context
        
        # Log with stack trace for server errors
        if error_response.status_code >= 500:
            log_data["stack_trace"] = traceback.format_exc()
        
        # Log the error
        self.logger.log(
            log_level,
            f"Error {error_response.error_code}: {error_response.message}",
            extra=log_data
        )


# Global error handler instance
error_handler = ErrorHandler()