"""
Parameter Extractor

This module extracts clinical parameters from unstructured text using regex patterns.
"""

import re
from typing import List, Dict, Any, Optional
import logging


class ParameterExtractor:
    """
    Extracts clinical parameters from unstructured clinical text.
    
    Supports extraction of 10+ common clinical parameters with various naming
    conventions and formats. Handles missing units through inference.
    """
    
    # Regex patterns for extracting clinical parameters
    # Format: parameter_name -> (pattern, standard_name, default_unit)
    # Each pattern captures: (value, optional_unit) or (systolic, diastolic, optional_unit) for BP
    PARAMETER_PATTERNS = {
        # Blood Pressure (special case - handled separately due to dual values)
        # Matches formats: "BP: 120/80", "Blood Pressure: 120/80 mmHg", "B.P.: 120 / 80"
        # Captures: group(1)=systolic, group(2)=diastolic, group(3)=unit
        "blood_pressure": (
            r"(?:BP|Blood\s+Pressure|B\.P\.|Systolic/Diastolic)[\s:]+(\d+)\s*/\s*(\d+)\s*(mmHg|mm\s*Hg)?",
            "Blood Pressure",
            "mmHg"
        ),
        # Blood Glucose
        # Matches formats: "Glucose: 100", "Blood Sugar: 100 mg/dL", "FBS: 100"
        # Captures: group(1)=value, group(2)=unit (optional)
        "glucose": (
            r"(?:Glucose|Blood\s+Sugar|Blood\s+Glucose|FBS|Fasting\s+Blood\s+Sugar)[\s:]+(\d+\.?\d*)\s*(mg/dL|mmol/L)?",
            "Blood Glucose",
            "mg/dL"
        ),
        # Total Cholesterol
        # Matches formats: "Cholesterol: 200", "Total Cholesterol: 200 mg/dL"
        # Captures: group(1)=value, group(2)=unit (optional)
        "total_cholesterol": (
            r"(?:Total\s+)?Cholesterol[\s:]+(\d+\.?\d*)\s*(mg/dL|mmol/L)?",
            "Total Cholesterol",
            "mg/dL"
        ),
        # LDL Cholesterol ("bad" cholesterol)
        # Matches formats: "LDL: 130", "LDL Cholesterol: 130 mg/dL"
        # Captures: group(1)=value, group(2)=unit (optional)
        "ldl": (
            r"LDL(?:\s+Cholesterol)?[\s:]+(\d+\.?\d*)\s*(mg/dL|mmol/L)?",
            "LDL Cholesterol",
            "mg/dL"
        ),
        # HDL Cholesterol ("good" cholesterol)
        # Matches formats: "HDL: 50", "HDL Cholesterol: 50 mg/dL"
        # Captures: group(1)=value, group(2)=unit (optional)
        "hdl": (
            r"HDL(?:\s+Cholesterol)?[\s:]+(\d+\.?\d*)\s*(mg/dL|mmol/L)?",
            "HDL Cholesterol",
            "mg/dL"
        ),
        # Triglycerides (blood fats)
        # Matches formats: "Triglycerides: 150", "Triglycerides: 150 mg/dL"
        # Captures: group(1)=value, group(2)=unit (optional)
        "triglycerides": (
            r"Triglycerides[\s:]+(\d+\.?\d*)\s*(mg/dL|mmol/L)?",
            "Triglycerides",
            "mg/dL"
        ),
        # Hemoglobin (oxygen-carrying protein in red blood cells)
        # Matches formats: "Hb: 14.5", "Hemoglobin: 14.5 g/dL"
        # Captures: group(1)=value, group(2)=unit (optional)
        "hemoglobin": (
            r"(?:Hb|Hemoglobin|Haemoglobin)[\s:]+(\d+\.?\d*)\s*(g/dL|g/L)?",
            "Hemoglobin",
            "g/dL"
        ),
        # White Blood Cell Count (immune system cells)
        # Matches formats: "WBC: 7.5", "White Blood Cell Count: 7.5 K/μL"
        # Captures: group(1)=value, group(2)=unit (optional)
        "wbc": (
            r"(?:WBC|White\s+Blood\s+Cell(?:\s+Count)?|Leukocytes)[\s:]+(\d+\.?\d*)\s*(×10³/μL|K/μL|10\^3/μL)?",
            "WBC",
            "×10³/μL"
        ),
        # Platelet Count (blood clotting cells)
        # Matches formats: "Platelet: 250", "PLT: 250 K/μL"
        # Captures: group(1)=value, group(2)=unit (optional)
        "platelet": (
            r"(?:Platelet(?:\s+Count)?|PLT)[\s:]+(\d+\.?\d*)\s*(×10³/μL|K/μL|10\^3/μL)?",
            "Platelet",
            "×10³/μL"
        ),
        # Creatinine (kidney function marker)
        # Matches formats: "Creatinine: 1.0", "Creatinine: 1.0 mg/dL"
        # Captures: group(1)=value, group(2)=unit (optional)
        "creatinine": (
            r"Creatinine[\s:]+(\d+\.?\d*)\s*(mg/dL|μmol/L)?",
            "Creatinine",
            "mg/dL"
        ),
    }
    
    # Parameter name variations for normalization
    NAME_VARIATIONS = {
        "BP": "Blood Pressure",
        "B.P.": "Blood Pressure",
        "Systolic/Diastolic": "Blood Pressure",
        "Blood Sugar": "Blood Glucose",
        "FBS": "Blood Glucose",
        "Fasting Blood Sugar": "Blood Glucose",
        "Hb": "Hemoglobin",
        "Haemoglobin": "Hemoglobin",
        "Leukocytes": "WBC",
        "White Blood Cell Count": "WBC",
        "White Blood Cell": "WBC",
        "PLT": "Platelet",
        "Platelet Count": "Platelet",
    }
    
    def __init__(self):
        """Initialize the parameter extractor"""
        self.logger = logging.getLogger(__name__)
    
    def extract_parameters(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract clinical parameters from text.
        
        Args:
            text: Clinical text to extract parameters from
        
        Returns:
            List of dictionaries containing parameter name, value, and unit
        """
        if not text or not text.strip():
            return []
        
        parameters = []
        
        # Extract blood pressure first (special case with two values)
        bp_params = self._parse_blood_pressure(text)
        parameters.extend(bp_params)
        
        # Extract other parameters
        for param_key, (pattern, standard_name, default_unit) in self.PARAMETER_PATTERNS.items():
            if param_key == "blood_pressure":
                continue  # Already handled
            
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    value_str = match.group(1)
                    unit = match.group(2) if len(match.groups()) > 1 and match.group(2) else None
                    
                    # Extract numeric value
                    value = self._extract_numeric_value(value_str)
                    if value is None:
                        continue
                    
                    # Extract or infer unit
                    final_unit = self._extract_unit(unit, standard_name, default_unit)
                    
                    parameters.append({
                        "name": standard_name,
                        "value": value,
                        "unit": final_unit
                    })
                except (ValueError, IndexError) as e:
                    self.logger.warning(f"Failed to extract parameter from match: {e}")
                    continue
        
        return parameters
    
    def _parse_blood_pressure(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse blood pressure values (systolic/diastolic).
        
        Args:
            text: Clinical text to extract blood pressure from
        
        Returns:
            List of dictionaries for systolic and diastolic parameters
        """
        pattern, _, default_unit = self.PARAMETER_PATTERNS["blood_pressure"]
        matches = re.finditer(pattern, text, re.IGNORECASE)
        
        parameters = []
        for match in matches:
            try:
                systolic_str = match.group(1)
                diastolic_str = match.group(2)
                unit = match.group(3) if len(match.groups()) > 2 and match.group(3) else None
                
                # Extract numeric values
                systolic = self._extract_numeric_value(systolic_str)
                diastolic = self._extract_numeric_value(diastolic_str)
                
                if systolic is None or diastolic is None:
                    continue
                
                # Extract or infer unit
                final_unit = self._extract_unit(unit, "Blood Pressure", default_unit)
                
                # Add both systolic and diastolic as separate parameters
                parameters.append({
                    "name": "Systolic Blood Pressure",
                    "value": systolic,
                    "unit": final_unit
                })
                parameters.append({
                    "name": "Diastolic Blood Pressure",
                    "value": diastolic,
                    "unit": final_unit
                })
            except (ValueError, IndexError) as e:
                self.logger.warning(f"Failed to extract blood pressure from match: {e}")
                continue
        
        return parameters
    
    def _extract_numeric_value(self, value_str: str) -> Optional[float]:
        """
        Extract numeric value from string.
        
        Args:
            value_str: String containing numeric value
        
        Returns:
            Float value or None if extraction fails
        """
        try:
            return float(value_str.strip())
        except (ValueError, AttributeError):
            return None
    
    def _extract_unit(self, unit: Optional[str], param_name: str, default_unit: str) -> str:
        """
        Extract or infer unit for a parameter.
        
        Args:
            unit: Unit string from regex match (may be None)
            param_name: Name of the parameter
            default_unit: Default unit for this parameter type
        
        Returns:
            Unit string (extracted or inferred)
        """
        if unit and unit.strip():
            # Clean up unit string
            return unit.strip()
        else:
            # Infer standard unit based on parameter type
            return default_unit
    
    def _normalize_parameter_name(self, name: str) -> str:
        """
        Normalize parameter name variations to standard names.
        
        Args:
            name: Parameter name to normalize
        
        Returns:
            Normalized parameter name
        """
        # Check if it's a known variation
        if name in self.NAME_VARIATIONS:
            return self.NAME_VARIATIONS[name]
        
        # Return as-is if not a known variation
        return name
