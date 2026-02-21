"""
Reference Range Database

This module provides a database of reference ranges for clinical parameters.
Reference ranges are based on standard medical guidelines.
"""

from typing import Optional, List
from backend.models.clinical_parameters import ReferenceRange


class ReferenceRangeDatabase:
    """
    Database for storing and retrieving reference ranges for clinical parameters.
    
    This class provides O(1) lookup time for reference ranges using a dictionary-based
    implementation. Reference ranges are based on standard medical guidelines.
    """
    
    # Standard reference ranges for common clinical parameters
    REFERENCE_RANGES = {
        "Systolic Blood Pressure": ReferenceRange(min_value=90, max_value=120, unit="mmHg"),
        "Diastolic Blood Pressure": ReferenceRange(min_value=60, max_value=80, unit="mmHg"),
        "Blood Glucose": ReferenceRange(min_value=70, max_value=100, unit="mg/dL"),
        "Total Cholesterol": ReferenceRange(max_value=200, unit="mg/dL"),
        "LDL Cholesterol": ReferenceRange(max_value=100, unit="mg/dL"),
        "HDL Cholesterol": ReferenceRange(min_value=40, unit="mg/dL"),
        "Triglycerides": ReferenceRange(max_value=150, unit="mg/dL"),
        "Hemoglobin": ReferenceRange(min_value=13.5, max_value=17.5, unit="g/dL"),
        "WBC": ReferenceRange(min_value=4.0, max_value=11.0, unit="×10³/μL"),
        "Platelet": ReferenceRange(min_value=150, max_value=400, unit="×10³/μL"),
        "Creatinine": ReferenceRange(min_value=0.6, max_value=1.2, unit="mg/dL"),
    }
    
    def __init__(self):
        """Initialize the reference range database"""
        self._ranges = self.REFERENCE_RANGES.copy()
    
    def get_range(self, parameter_name: str) -> Optional[ReferenceRange]:
        """
        Retrieve the reference range for a given parameter name.
        
        Args:
            parameter_name: Name of the clinical parameter
        
        Returns:
            ReferenceRange object if found, None otherwise
        """
        return self._ranges.get(parameter_name)
    
    def add_range(self, parameter_name: str, range: ReferenceRange) -> None:
        """
        Add or update a reference range for a parameter.
        
        Args:
            parameter_name: Name of the clinical parameter
            range: ReferenceRange object to store
        """
        self._ranges[parameter_name] = range
    
    def list_parameters(self) -> List[str]:
        """
        List all parameters that have reference ranges.
        
        Returns:
            List of parameter names
        """
        return list(self._ranges.keys())
    
    def has_parameter(self, parameter_name: str) -> bool:
        """
        Check if a parameter has a reference range.
        
        Args:
            parameter_name: Name of the clinical parameter
        
        Returns:
            True if parameter has a reference range, False otherwise
        """
        return parameter_name in self._ranges
