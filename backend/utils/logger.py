import logging
import sys
import json
from datetime import datetime
from backend.config import Config

def setup_logger(name: str) -> logging.Logger:
    """Set up logger with consistent formatting and structured logging support"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        
        # Use structured logging format for better debugging
        formatter = StructuredFormatter()
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, Config.LOG_LEVEL))
    
    return logger


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging with JSON output for errors"""
    
    def format(self, record):
        # Basic log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage()
        }
        
        # Add extra fields if present
        if hasattr(record, 'request_id'):
            log_entry["request_id"] = record.request_id
        
        if hasattr(record, 'error_code'):
            log_entry["error_code"] = record.error_code
            
        if hasattr(record, 'error_category'):
            log_entry["error_category"] = record.error_category
            
        if hasattr(record, 'status_code'):
            log_entry["status_code"] = record.status_code
            
        if hasattr(record, 'method'):
            log_entry["http_method"] = record.method
            
        if hasattr(record, 'url'):
            log_entry["url"] = record.url
            
        if hasattr(record, 'user_agent'):
            log_entry["user_agent"] = record.user_agent
            
        if hasattr(record, 'context'):
            log_entry["context"] = record.context
            
        if hasattr(record, 'stack_trace'):
            log_entry["stack_trace"] = record.stack_trace
        
        # For ERROR and CRITICAL levels, use JSON format for better parsing
        if record.levelno >= logging.ERROR:
            return json.dumps(log_entry, indent=None, separators=(',', ':'))
        else:
            # For other levels, use human-readable format
            return f"{log_entry['timestamp']} - {log_entry['logger']} - {log_entry['level']} - {log_entry['message']}"