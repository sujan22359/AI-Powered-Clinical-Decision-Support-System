"""
Analysis Engine for Clinical Report Analyzer

This module provides the core analysis engine that orchestrates the AI processing
workflow, structures responses, and ensures medical safety compliance.
"""

import json
import re
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime

from backend.config import Config
from backend.services.gemini_service import GeminiService, LLMResponse
from backend.services.document_parser import DocumentParser, DocumentParsingError
# Threshold-based risk assessment modules
from backend.services.parameter_extractor import ParameterExtractor
from backend.services.reference_range_db import ReferenceRangeDatabase
from backend.services.threshold_evaluator import ThresholdEvaluator
from backend.services.risk_explainer import RiskExplainer
from backend.models.clinical_parameters import ClinicalParameter, EvaluatedParameter, RiskLevel
from backend.utils.logger import setup_logger


@dataclass
class RiskIndicator:
    """
    Represents a risk indicator found in clinical text.
    
    This dataclass supports both AI-detected risks and threshold-based risks.
    For threshold-based risks, additional fields provide detailed parameter information.
    
    Attributes:
        finding: Description of the risk finding
        category: Category of the risk (e.g., "cardiovascular", "metabolic")
        severity: Severity level - "low", "medium", or "high"
        description: Detailed description of the risk
        parameter_name: Optional name of the clinical parameter (for threshold-based risks)
        actual_value: Optional actual value of the parameter (for threshold-based risks)
        unit: Optional unit of measurement (for threshold-based risks)
        reference_range: Optional reference range string (for threshold-based risks)
        deviation_percent: Optional percentage deviation from normal (for threshold-based risks)
        threshold_based: Whether this is a threshold-based risk (default: False)
    """
    finding: str
    category: str
    severity: str  # "low", "medium", "high"
    description: str
    parameter_name: Optional[str] = None
    actual_value: Optional[float] = None
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    deviation_percent: Optional[float] = None
    threshold_based: bool = False


@dataclass
class AnalysisResult:
    """Structured analysis result"""
    summary: str
    key_findings: List[str]
    risk_indicators: List[RiskIndicator]
    follow_up_suggestions: List[str]
    medical_disclaimer: str
    analysis_timestamp: str
    processing_notes: Optional[str] = None


class AnalysisEngineError(Exception):
    """Custom exception for analysis engine errors"""
    pass


class AnalysisEngine:
    """
    Core analysis engine for processing clinical text and generating structured insights.
    
    Features:
    - Orchestrates AI processing workflow
    - Structures responses with required sections
    - Generates medical disclaimers and safety filters
    - Detects and highlights risk indicators
    - Ensures medical safety compliance
    """
    
    def __init__(self, llm_service: Optional[GeminiService] = None, 
                 document_parser: Optional[DocumentParser] = None):
        """
        Initialize analysis engine with required services.
        
        Args:
            llm_service: Gemini service for AI processing (creates new if None)
            document_parser: Document parser service (creates new if None)
        """
        self.logger = setup_logger(__name__)
        
        # Initialize services with Gemini
        self.llm_service = llm_service or GeminiService()
        self.document_parser = document_parser or DocumentParser()
        self.logger.info("Using Gemini AI for text analysis")
        
        # Initialize threshold-based risk assessment services
        self.parameter_extractor = ParameterExtractor()
        self.reference_range_db = ReferenceRangeDatabase()
        self.threshold_evaluator = ThresholdEvaluator()
        self.risk_explainer = RiskExplainer()
        
        # Medical disclaimer template
        self.medical_disclaimer = (
            "IMPORTANT MEDICAL DISCLAIMER: This analysis is for informational purposes only "
            "and is not intended as medical advice, diagnosis, or treatment. The AI-generated "
            "insights should not replace professional medical consultation. Always consult "
            "with qualified healthcare professionals for medical decisions. This system does "
            "not provide medical diagnoses or treatment recommendations."
        )
        
        # Risk indicator patterns for detection (order matters - check specific patterns first)
        self.risk_patterns = {
            "high": [
                # Critical/Emergency terms
                r"critical|severe|urgent|emergency|immediate attention|life-threatening",
                # Serious injuries and conditions
                r"complete tear|full.*thickness.*tear|rupture|avulsion|dislocation",
                r"fracture|broken|displaced|comminuted|compound",
                r"hemorrhage|bleeding|hematoma|contusion",
                # Serious pathology
                r"malignant|cancer|tumor|metastasis|mass",
                r"infarction|stroke|ischemia|necrosis",
                r"obstruction|occlusion|stenosis.*severe",
                # Abnormal values
                r"abnormal.*high|extremely elevated|dangerously|critically.*abnormal",
                # Organ damage
                r"perforation|abscess|gangrene|sepsis"
            ],
            "medium": [
                # Moderate injuries
                r"partial tear|grade.*[23]|moderate.*tear",
                r"sprain|strain|edema|effusion|inflammation",
                r"degenerative|arthritis|osteoarthritis",
                # Moderate abnormalities
                r"abnormal|elevated|concerning|requires attention",
                r"outside.*range|above.*normal|below.*normal",
                r"moderate.*risk|borderline",
                # Moderate pathology
                r"nodule|lesion|cyst|polyp"
            ],
            "low": [
                # Minor issues
                r"minor.*tear|grade.*1|mild.*tear|minimal",
                r"slightly.*(elevated|abnormal)|mildly.*(abnormal|elevated)|minor.*concern",
                r"watch|monitor|follow.*up",
                r"incidental.*finding|normal.*variant"
            ]
        }
        
        # Safety filter patterns (content to flag or modify)
        self.safety_filters = [
            r"diagnos[ei]s|diagnose[ds]",  # Diagnostic language
            r"treat(ment)?.*recommend|prescrib[ei]",  # Treatment recommendations
            r"you (should|must|need to).*medic",  # Direct medical advice
            r"this (is|indicates).*disease|condition"  # Definitive medical statements
        ]
        
        self.logger.info("Analysis engine initialized successfully")
    
    def _create_threshold_risk_indicator(self, evaluated_param: EvaluatedParameter) -> RiskIndicator:
        """
        Convert an EvaluatedParameter to a RiskIndicator.
        
        Args:
            evaluated_param: EvaluatedParameter from threshold evaluation
            
        Returns:
            RiskIndicator with threshold-based fields populated
        """
        param = evaluated_param.parameter
        
        # Map RiskLevel to severity string
        severity_map = {
            RiskLevel.LOW: "low",
            RiskLevel.MEDIUM: "medium",
            RiskLevel.HIGH: "high",
            RiskLevel.UNKNOWN: "medium"  # Default to medium for unknown
        }
        severity = severity_map.get(evaluated_param.risk_level, "medium")
        
        # Format reference range string
        reference_range_str = None
        if param.reference_range is not None:
            if param.reference_range.min_value is not None and param.reference_range.max_value is not None:
                reference_range_str = f"{param.reference_range.min_value}-{param.reference_range.max_value} {param.reference_range.unit}"
            elif param.reference_range.min_value is not None:
                reference_range_str = f">{param.reference_range.min_value} {param.reference_range.unit}"
            else:
                reference_range_str = f"<{param.reference_range.max_value} {param.reference_range.unit}"
        
        # Calculate deviation percent
        deviation_percent = None
        if param.reference_range is not None:
            deviation_percent = self.threshold_evaluator._calculate_deviation(
                param.value, param.reference_range
            )
        
        # Format finding text with value and range
        finding_text = f"{param.name}: {param.value} {param.unit}"
        if reference_range_str:
            finding_text += f" (Normal: {reference_range_str})"
        
        # Determine category based on parameter name
        category = self._categorize_risk_type(param.name)
        
        return RiskIndicator(
            finding=finding_text,
            category=category,
            severity=severity,
            description=evaluated_param.explanation,
            parameter_name=param.name,
            actual_value=param.value,
            unit=param.unit,
            reference_range=reference_range_str,
            deviation_percent=deviation_percent,
            threshold_based=True
        )
    
    def _merge_risk_indicators(
        self, 
        ai_risks: List[RiskIndicator], 
        threshold_risks: List[RiskIndicator]
    ) -> List[RiskIndicator]:
        """
        Merge AI-detected and threshold-based risk indicators.
        
        This function intelligently combines risks from both sources:
        1. Starts with threshold-based risks (more reliable, objective measurements)
        2. Adds AI-detected risks that don't duplicate threshold findings
        3. Deduplicates by checking if AI findings mention same parameters
        4. Sorts final list by severity (HIGH -> MEDIUM -> LOW)
        
        Deduplication Logic:
        - Threshold-based risks take precedence (objective measurements)
        - AI risks are checked against threshold parameter names
        - If an AI finding mentions a parameter already in threshold risks, it's skipped
        - This prevents showing "Blood Glucose elevated" from both AI and threshold
        
        Example:
        - Threshold: "Blood Glucose: 250 mg/dL (Normal: 70-100 mg/dL)" - HIGH
        - AI: "Elevated blood glucose detected" - Would be skipped (duplicate)
        - AI: "Possible diabetic retinopathy" - Would be included (different finding)
        
        Args:
            ai_risks: Risk indicators from AI analysis
            threshold_risks: Risk indicators from threshold evaluation
            
        Returns:
            Merged and sorted list of risk indicators (HIGH -> MEDIUM -> LOW)
        """
        # Start with threshold-based risks (more reliable, objective measurements)
        merged_risks = list(threshold_risks)
        
        # Track parameter names from threshold-based risks for deduplication
        # Example: {"blood glucose", "cholesterol", "blood pressure"}
        threshold_param_names = set()
        for risk in threshold_risks:
            if risk.parameter_name:
                # Normalize parameter names for comparison (lowercase, trimmed)
                threshold_param_names.add(risk.parameter_name.lower().strip())
        
        # Add AI risks that don't duplicate threshold-based risks
        for ai_risk in ai_risks:
            # Check if this AI risk mentions a parameter we already have from threshold analysis
            is_duplicate = False
            ai_finding_lower = ai_risk.finding.lower()
            
            for param_name in threshold_param_names:
                # Check if the parameter name appears in the AI finding
                # Example: "blood glucose" in "elevated blood glucose detected"
                if param_name in ai_finding_lower:
                    is_duplicate = True
                    self.logger.debug(
                        f"Skipping duplicate AI risk for parameter: {param_name}"
                    )
                    break
            
            # Only add AI risk if it's not a duplicate
            if not is_duplicate:
                merged_risks.append(ai_risk)
        
        # Sort by severity: HIGH > MEDIUM > LOW > NORMAL
        # This ensures critical findings appear first in the UI
        # Example order: [HIGH glucose, HIGH cholesterol, MEDIUM BP, LOW hemoglobin]
        severity_order = {"high": 3, "medium": 2, "low": 1, "normal": 0}
        merged_risks.sort(
            key=lambda x: severity_order.get(x.severity.lower(), 0),
            reverse=True  # Descending order (highest severity first)
        )
        
        self.logger.info(
            f"Risk indicators merged: {len(threshold_risks)} threshold-based, "
            f"{len(ai_risks)} AI-detected, {len(merged_risks)} total after deduplication"
        )
        
        return merged_risks
    
    def analyze_document(self, file_content: Union[bytes, Any], filename: str) -> AnalysisResult:
        """
        Analyze a clinical document and generate structured insights.
        
        Args:
            file_content: Document content as bytes or file-like object
            filename: Original filename
            
        Returns:
            AnalysisResult with structured insights
            
        Raises:
            AnalysisEngineError: If analysis fails
        """
        try:
            self.logger.info(f"Starting document analysis for: {filename}")
            
            # Step 1: Parse document to extract text
            clinical_text = self.document_parser.parse_document(file_content, filename)
            
            # Step 2: Analyze the extracted text
            return self.analyze_clinical_text(clinical_text)
            
        except DocumentParsingError as e:
            error_msg = f"Document parsing failed: {str(e)}"
            self.logger.error(error_msg)
            raise AnalysisEngineError(error_msg)
        except Exception as e:
            error_msg = f"Document analysis failed: {str(e)}"
            self.logger.error(error_msg)
            raise AnalysisEngineError(error_msg)
    
    def analyze_clinical_text(self, clinical_text: str) -> AnalysisResult:
        """
        Analyze clinical text and generate structured medical insights.
        
        Args:
            clinical_text: Raw clinical text to analyze
            
        Returns:
            AnalysisResult with structured insights
            
        Raises:
            AnalysisEngineError: If analysis fails
        """
        if not clinical_text or not clinical_text.strip():
            raise AnalysisEngineError("Clinical text is empty or invalid")
        
        try:
            self.logger.debug("Starting clinical text analysis")
            
            # Step 1: Get AI analysis
            llm_response = self.llm_service.analyze_clinical_text(clinical_text)
            
            if not llm_response.success:
                error_msg = f"LLM analysis failed: {llm_response.error_message}"
                self.logger.error(error_msg)
                raise AnalysisEngineError(error_msg)
            
            # Step 2: Parse and structure the AI response
            structured_data = self._parse_llm_response(llm_response.content)
            
            # Step 3: Apply safety filters and enhancements
            enhanced_data = self._apply_safety_filters(structured_data)
            
            # Step 3.5: Apply content organization
            enhanced_data = self._organize_content(enhanced_data, clinical_text)
            
            # Step 4: Detect and categorize AI-detected risk indicators
            ai_risk_indicators = self._detect_risk_indicators(
                enhanced_data.get("risk_indicators", []),
                clinical_text
            )
            
            # Step 5: Threshold-based risk assessment pipeline
            threshold_risk_indicators = []
            processing_notes = []
            
            try:
                self.logger.debug("Starting threshold-based risk assessment")
                
                # Extract clinical parameters from text
                extracted_params_data = self.parameter_extractor.extract_parameters(clinical_text)
                
                if extracted_params_data:
                    self.logger.info(f"Extracted {len(extracted_params_data)} clinical parameters")
                    
                    # Convert extracted data to ClinicalParameter objects
                    clinical_parameters = []
                    seen_parameters = set()  # Track parameters to avoid duplicates
                    
                    for param_data in extracted_params_data:
                        param_name = param_data["name"]
                        
                        # Handle blood pressure specially (has two values)
                        if param_name == "Blood Pressure" and isinstance(param_data["value"], dict):
                            # Create separate parameters for systolic and diastolic
                            systolic_value = param_data["value"]["systolic"]
                            diastolic_value = param_data["value"]["diastolic"]
                            
                            # Check for duplicates
                            systolic_key = f"Systolic Blood Pressure_{systolic_value}"
                            diastolic_key = f"Diastolic Blood Pressure_{diastolic_value}"
                            
                            if systolic_key not in seen_parameters:
                                seen_parameters.add(systolic_key)
                                # Get reference ranges
                                systolic_range = self.reference_range_db.get_range("Systolic Blood Pressure")
                                
                                clinical_parameters.append(ClinicalParameter(
                                    name="Systolic Blood Pressure",
                                    value=systolic_value,
                                    unit=param_data["unit"],
                                    reference_range=systolic_range
                                ))
                            
                            if diastolic_key not in seen_parameters:
                                seen_parameters.add(diastolic_key)
                                diastolic_range = self.reference_range_db.get_range("Diastolic Blood Pressure")
                                
                                clinical_parameters.append(ClinicalParameter(
                                    name="Diastolic Blood Pressure",
                                    value=diastolic_value,
                                    unit=param_data["unit"],
                                    reference_range=diastolic_range
                                ))
                        else:
                            # Single-value parameter
                            param_value = param_data["value"]
                            param_unit = param_data["unit"]
                            
                            # Create unique key for deduplication
                            param_key = f"{param_name}_{param_value}_{param_unit}"
                            
                            # Skip if we've already seen this exact parameter
                            if param_key in seen_parameters:
                                self.logger.debug(f"Skipping duplicate parameter: {param_name} = {param_value} {param_unit}")
                                continue
                            
                            seen_parameters.add(param_key)
                            
                            # Look up reference range
                            ref_range = self.reference_range_db.get_range(param_name)
                            
                            clinical_parameters.append(ClinicalParameter(
                                name=param_name,
                                value=param_value,
                                unit=param_unit,
                                reference_range=ref_range
                            ))
                    
                    # Evaluate parameters against thresholds
                    if clinical_parameters:
                        evaluated_params = self.threshold_evaluator.evaluate_parameters(clinical_parameters)
                        
                        # Track seen risk indicators to prevent duplicates
                        seen_risks = set()
                        
                        # Convert evaluated parameters to risk indicators
                        # Only include parameters with MEDIUM or HIGH risk
                        for evaluated_param in evaluated_params:
                            if evaluated_param.risk_level in [RiskLevel.MEDIUM, RiskLevel.HIGH]:
                                # Create unique key for this risk
                                risk_key = f"{evaluated_param.parameter.name}_{evaluated_param.parameter.value}_{evaluated_param.risk_level.value}"
                                
                                # Skip if we've already added this risk
                                if risk_key in seen_risks:
                                    self.logger.debug(f"Skipping duplicate risk indicator: {evaluated_param.parameter.name}")
                                    continue
                                
                                seen_risks.add(risk_key)
                                risk_indicator = self._create_threshold_risk_indicator(evaluated_param)
                                threshold_risk_indicators.append(risk_indicator)
                        
                        self.logger.info(
                            f"Threshold evaluation complete: {len(threshold_risk_indicators)} risks identified"
                        )
                else:
                    self.logger.info("No clinical parameters extracted from text")
                    
            except Exception as e:
                # Log error but continue with AI analysis
                self.logger.error(f"Threshold analysis failed: {e}", exc_info=True)
                processing_notes.append(
                    "Note: Threshold-based risk assessment was unavailable for this analysis. "
                    "Results are based on AI analysis only."
                )
            
            # Step 6: Merge AI and threshold-based risk indicators
            merged_risk_indicators = self._merge_risk_indicators(
                ai_risk_indicators,
                threshold_risk_indicators
            )
            
            # Step 7: Generate final structured result
            processing_note = None
            if llm_response.retry_count > 0:
                processing_note = f"Processed with {llm_response.retry_count} retries"
            if processing_notes:
                if processing_note:
                    processing_note += "; " + "; ".join(processing_notes)
                else:
                    processing_note = "; ".join(processing_notes)
            
            result = AnalysisResult(
                summary=enhanced_data.get("summary", ""),
                key_findings=enhanced_data.get("key_findings", []),
                risk_indicators=merged_risk_indicators,
                follow_up_suggestions=enhanced_data.get("follow_up_suggestions", []),
                medical_disclaimer=self.medical_disclaimer,
                analysis_timestamp=datetime.now().isoformat(),
                processing_notes=processing_note
            )
            
            self.logger.info("Clinical text analysis completed successfully")
            return result
            
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse AI response as JSON: {str(e)}"
            self.logger.error(error_msg)
            raise AnalysisEngineError(error_msg)
        except Exception as e:
            error_msg = f"Clinical text analysis failed: {str(e)}"
            self.logger.error(error_msg)
            raise AnalysisEngineError(error_msg)
    
    def _parse_llm_response(self, llm_content: str) -> Dict[str, Any]:
        """
        Parse LLM response and extract structured data.
        
        Args:
            llm_content: Raw LLM response content
            
        Returns:
            Dictionary with structured data
            
        Raises:
            json.JSONDecodeError: If response is not valid JSON
        """
        try:
            # Try to parse as JSON directly
            return json.loads(llm_content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', llm_content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # Try to find JSON-like content
            json_match = re.search(r'(\{.*\})', llm_content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # If no JSON found, create a basic structure
            self.logger.warning("Could not parse LLM response as JSON, creating basic structure")
            return {
                "summary": llm_content[:500] + "..." if len(llm_content) > 500 else llm_content,
                "key_findings": [],
                "risk_indicators": [],
                "follow_up_suggestions": [],
                "medical_disclaimer": self.medical_disclaimer
            }
    
    def _apply_safety_filters(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply safety filters to ensure medical compliance.
        
        Args:
            data: Structured data from LLM response
            
        Returns:
            Filtered and safe data
        """
        filtered_data = data.copy()
        
        # Filter summary
        if "summary" in filtered_data:
            filtered_data["summary"] = self._filter_text_content(filtered_data["summary"])
        
        # Filter key findings
        if "key_findings" in filtered_data:
            filtered_data["key_findings"] = [
                self._filter_text_content(finding) 
                for finding in filtered_data["key_findings"]
            ]
        
        # Filter follow-up suggestions
        if "follow_up_suggestions" in filtered_data:
            filtered_data["follow_up_suggestions"] = [
                self._filter_text_content(suggestion) 
                for suggestion in filtered_data["follow_up_suggestions"]
            ]
        
        # Ensure medical disclaimer is present and correct
        filtered_data["medical_disclaimer"] = self.medical_disclaimer
        
        return filtered_data
    
    def _organize_content(self, data: Dict[str, Any], clinical_text: str) -> Dict[str, Any]:
        """
        Organize and format content for better presentation.
        
        Args:
            data: Structured data from LLM response
            clinical_text: Original clinical text
            
        Returns:
            Organized and formatted data
        """
        organized_data = data.copy()
        
        # Organize key findings by importance
        if "key_findings" in organized_data:
            organized_data["key_findings"] = self._organize_findings_by_importance(
                organized_data["key_findings"]
            )
        
        # Format numerical values in all text fields
        for field in ["summary", "key_findings", "follow_up_suggestions"]:
            if field in organized_data:
                if isinstance(organized_data[field], str):
                    organized_data[field] = self._format_numerical_values(organized_data[field])
                elif isinstance(organized_data[field], list):
                    organized_data[field] = [
                        self._format_numerical_values(item) for item in organized_data[field]
                    ]
        
        # Handle cases with no significant findings
        organized_data = self._handle_no_significant_findings(organized_data)
        
        return organized_data
    
    def _scan_for_additional_risks(self, clinical_text: str) -> List[RiskIndicator]:
        """
        Scan clinical text for additional risk indicators not caught by LLM.
        
        Args:
            clinical_text: Original clinical text
            
        Returns:
            List of additional RiskIndicator objects
        """
        additional_risks = []
        text_lower = clinical_text.lower()
        
        # Define specific patterns to look for
        specific_patterns = {
            "high": [
                (r"critical.*value|emergency.*level", "Critical values detected"),
                (r"immediate.*attention|urgent.*follow.*up", "Urgent attention required")
            ],
            "medium": [
                (r"abnormal.*result|outside.*normal.*range", "Abnormal results found"),
                (r"elevated.*level|increased.*risk", "Elevated levels detected")
            ],
            "low": [
                (r"monitor.*closely|follow.*up.*recommended", "Monitoring recommended"),
                (r"slightly.*abnormal|borderline.*result", "Borderline findings")
            ]
        }
        
        for severity, patterns in specific_patterns.items():
            for pattern, description in patterns:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    # Extract context around the match
                    start = max(0, match.start() - 50)
                    end = min(len(clinical_text), match.end() + 50)
                    context = clinical_text[start:end].strip()
                    
                    additional_risks.append(RiskIndicator(
                        finding=context,
                        category=self._categorize_risk_type(context),
                        severity=severity,
                        description=description
                    ))
        
        return additional_risks
    
    def _filter_text_content(self, text: str) -> str:
        """
        Filter text content to remove unsafe medical language.
        
        Args:
            text: Text content to filter (can be string, dict, or list)
            
        Returns:
            Filtered text content as readable string
        """
        # Handle non-string inputs
        if not text:
            return text
        
        # Convert to readable string if not already
        if not isinstance(text, str):
            # If it's a dict, extract the 'value' or create readable text
            if isinstance(text, dict):
                # Try to extract meaningful text from dict
                if 'value' in text:
                    text = str(text['value'])
                elif 'explanation' in text:
                    text = str(text['explanation'])
                elif 'terminology' in text and 'value' in text:
                    # Format: "Value - Explanation (Terminology)"
                    parts = []
                    if text.get('value'):
                        parts.append(str(text['value']))
                    if text.get('explanation'):
                        parts.append(str(text['explanation']))
                    if text.get('terminology'):
                        parts.append(f"({text['terminology']})")
                    text = ' - '.join(parts) if parts else str(text)
                else:
                    # Fallback: create readable text from dict
                    text = ' - '.join(f"{v}" for k, v in text.items() if v and k != 'reference_range')
            # If it's a list, join items
            elif isinstance(text, list):
                text = ', '.join(str(item) for item in text)
            else:
                text = str(text)
        
        filtered_text = text
        
        # Apply safety filters
        for pattern in self.safety_filters:
            # Replace diagnostic language with safer alternatives
            if re.search(pattern, filtered_text, re.IGNORECASE):
                if "diagnos" in pattern:
                    filtered_text = re.sub(
                        r"diagnos[ei]s|diagnose[ds]", 
                        "findings suggest", 
                        filtered_text, 
                        flags=re.IGNORECASE
                    )
                elif "treat" in pattern:
                    filtered_text = re.sub(
                        r"treat(ment)?.*recommend|prescrib[ei]", 
                        "may benefit from discussing with healthcare provider", 
                        filtered_text, 
                        flags=re.IGNORECASE
                    )
                elif "you (should|must|need to)" in pattern:
                    filtered_text = re.sub(
                        r"you (should|must|need to)", 
                        "consider discussing with your healthcare provider about", 
                        filtered_text, 
                        flags=re.IGNORECASE
                    )
                elif "this (is|indicates)" in pattern:
                    filtered_text = re.sub(
                        r"this (is|indicates)", 
                        "this may suggest", 
                        filtered_text, 
                        flags=re.IGNORECASE
                    )
        
        return filtered_text
    
    def _detect_risk_indicators(self, raw_indicators: List[str], clinical_text: str) -> List[RiskIndicator]:
        """
        Detect and categorize risk indicators from findings and clinical text.
        
        Args:
            raw_indicators: Raw risk indicators from LLM response
            clinical_text: Original clinical text for additional detection
            
        Returns:
            List of categorized RiskIndicator objects
        """
        risk_indicators = []
        
        # Process raw indicators from LLM
        for indicator in raw_indicators:
            # Handle both string and dict formats
            if isinstance(indicator, dict):
                # Extract text from dict (could be various formats)
                indicator_text = indicator.get('finding', '') or indicator.get('description', '') or str(indicator)
            elif isinstance(indicator, str):
                indicator_text = indicator
            else:
                # Skip non-string, non-dict items
                continue
            
            if not indicator_text or not indicator_text.strip():
                continue
            
            severity = self._categorize_risk_severity(indicator_text)
            category = self._categorize_risk_type(indicator_text)
            
            risk_indicators.append(RiskIndicator(
                finding=indicator_text.strip(),
                category=category,
                severity=severity,
                description=f"Identified {severity} risk indicator in {category} category"
            ))
        
        # Additional detection from clinical text
        additional_risks = self._scan_for_additional_risks(clinical_text)
        risk_indicators.extend(additional_risks)
        
        # Remove duplicates and sort by severity
        unique_risks = self._deduplicate_risks(risk_indicators)
        return sorted(unique_risks, key=lambda x: {"high": 3, "medium": 2, "low": 1}[x.severity], reverse=True)
    
    def _categorize_risk_severity(self, indicator: str) -> str:
        """
        Categorize risk indicator severity based on content.
        
        Args:
            indicator: Risk indicator text
            
        Returns:
            Severity level: "high", "medium", or "low"
        """
        indicator_lower = indicator.lower()
        
        # Check high severity patterns first
        for pattern in self.risk_patterns["high"]:
            if re.search(pattern, indicator_lower):
                return "high"
        
        # Check low severity patterns before medium (more specific)
        for pattern in self.risk_patterns["low"]:
            if re.search(pattern, indicator_lower):
                return "low"
        
        # Check medium severity patterns
        for pattern in self.risk_patterns["medium"]:
            if re.search(pattern, indicator_lower):
                return "medium"
        
        # Default to medium if no pattern matches
        return "medium"
    
    def _categorize_risk_type(self, indicator: str) -> str:
        """
        Categorize risk indicator by medical type/system.
        
        Args:
            indicator: Risk indicator text
            
        Returns:
            Risk category
        """
        indicator_lower = indicator.lower()
        
        # Define category patterns
        categories = {
            "cardiovascular": [r"heart|cardiac|blood pressure|cholesterol|ecg|ekg"],
            "metabolic": [r"glucose|diabetes|sugar|insulin|metabolic"],
            "hematologic": [r"blood|hemoglobin|hematocrit|platelet|white.*cell"],
            "hepatic": [r"liver|hepatic|alt|ast|bilirubin"],
            "renal": [r"kidney|renal|creatinine|urea|gfr"],
            "respiratory": [r"lung|respiratory|oxygen|breathing"],
            "neurological": [r"brain|neuro|cognitive|mental"],
            "general": [r".*"]  # Catch-all
        }
        
        for category, patterns in categories.items():
            for pattern in patterns:
                if re.search(pattern, indicator_lower):
                    return category
        
        return "general"
    
    def _organize_findings_by_importance(self, findings: List[str]) -> List[str]:
        """
        Organize findings by medical importance and category.
        
        Args:
            findings: List of medical findings
            
        Returns:
            Organized list of findings
        """
        if not findings:
            return findings
        
        # Define importance keywords (order matters)
        importance_patterns = {
            "critical": [r"critical|severe|urgent|emergency|immediate|dangerous"],
            "high": [r"abnormal|elevated|high|low|concerning|significant"],
            "medium": [r"borderline|mild|slight|moderate"],
            "low": [r"normal|within.*range|stable|unremarkable"]
        }
        
        # Categorize findings by importance
        categorized = {"critical": [], "high": [], "medium": [], "low": []}
        
        for finding in findings:
            finding_lower = finding.lower()
            assigned = False
            
            for importance, patterns in importance_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, finding_lower):
                        categorized[importance].append(finding)
                        assigned = True
                        break
                if assigned:
                    break
            
            # If no pattern matches, assign to medium importance
            if not assigned:
                categorized["medium"].append(finding)
        
        # Combine in order of importance
        organized = []
        for importance in ["critical", "high", "medium", "low"]:
            organized.extend(categorized[importance])
        
        return organized
    
    def _format_numerical_values(self, text: str) -> str:
        """
        Format numerical values and ranges for clear presentation.
        
        Args:
            text: Text containing numerical values
            
        Returns:
            Text with formatted numerical values
        """
        if not text:
            return text
        
        # Pattern to match common medical values with units
        patterns = [
            # Blood pressure: 120/80 mmHg
            (r'(\d+)/(\d+)\s*(mmhg|mm\s*hg)', r'\1/\2 mmHg'),
            # Temperature: 98.6F or 37C
            (r'(\d+\.?\d*)\s*°?\s*([cf])\b', r'\1°\2'),
            # Lab values with units: 150 mg/dL, 5.2 g/L
            (r'(\d+\.?\d*)\s*(mg/dl|g/l|mmol/l|μl|ml|l)\b', r'\1 \2'),
            # Percentages: 15%
            (r'(\d+\.?\d*)\s*%', r'\1%'),
            # Ranges: 120-140, 5.0-7.0
            (r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)', r'\1-\2'),
        ]
        
        formatted_text = text
        for pattern, replacement in patterns:
            formatted_text = re.sub(pattern, replacement, formatted_text, flags=re.IGNORECASE)
        
        return formatted_text
    
    def _handle_no_significant_findings(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle cases where no significant findings are present.
        
        Args:
            analysis_data: Analysis data dictionary
            
        Returns:
            Updated analysis data with appropriate messaging
        """
        # Check if findings are minimal or indicate normal results
        findings = analysis_data.get("key_findings", [])
        risks = analysis_data.get("risk_indicators", [])
        
        # Count significant findings
        significant_findings = 0
        for finding in findings:
            finding_lower = finding.lower()
            if any(word in finding_lower for word in ["abnormal", "elevated", "high", "low", "concerning"]):
                significant_findings += 1
        
        # If very few significant findings, add reassuring message
        if significant_findings == 0 and len(risks) == 0:
            if not analysis_data.get("summary"):
                analysis_data["summary"] = "The clinical document shows predominantly normal findings with no significant abnormalities identified."
            
            if not findings:
                analysis_data["key_findings"] = ["No significant abnormal findings identified in the clinical data"]
            
            if not analysis_data.get("follow_up_suggestions"):
                analysis_data["follow_up_suggestions"] = [
                    "Continue routine healthcare maintenance as recommended by your healthcare provider",
                    "Maintain healthy lifestyle practices",
                    "Follow up as scheduled for regular check-ups"
                ]
        
        return analysis_data
    
    def _scan_for_additional_risks(self, clinical_text: str) -> List[RiskIndicator]:
        additional_risks = []
        text_lower = clinical_text.lower()
        
        # Define specific patterns to look for
        specific_patterns = {
            "high": [
                (r"critical.*value|emergency.*level", "Critical values detected"),
                (r"immediate.*attention|urgent.*follow.*up", "Urgent attention required")
            ],
            "medium": [
                (r"abnormal.*result|outside.*normal.*range", "Abnormal results found"),
                (r"elevated.*level|increased.*risk", "Elevated levels detected")
            ],
            "low": [
                (r"monitor.*closely|follow.*up.*recommended", "Monitoring recommended"),
                (r"slightly.*abnormal|borderline.*result", "Borderline findings")
            ]
        }
        
        for severity, patterns in specific_patterns.items():
            for pattern, description in patterns:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    # Extract context around the match
                    start = max(0, match.start() - 50)
                    end = min(len(clinical_text), match.end() + 50)
                    context = clinical_text[start:end].strip()
                    
                    additional_risks.append(RiskIndicator(
                        finding=context,
                        category=self._categorize_risk_type(context),
                        severity=severity,
                        description=description
                    ))
        
        return additional_risks
    
    def _deduplicate_risks(self, risks: List[RiskIndicator]) -> List[RiskIndicator]:
        """
        Remove duplicate risk indicators based on similarity.
        
        Args:
            risks: List of RiskIndicator objects
            
        Returns:
            Deduplicated list of RiskIndicator objects
        """
        if not risks:
            return risks
        
        unique_risks = []
        seen_findings = set()
        
        for risk in risks:
            # Create a normalized version for comparison
            normalized = re.sub(r'\s+', ' ', risk.finding.lower().strip())
            
            # Check if we've seen a similar finding
            is_duplicate = False
            for seen in seen_findings:
                # Simple similarity check - if 80% of words overlap
                seen_words = set(seen.split())
                current_words = set(normalized.split())
                
                if len(seen_words) > 0 and len(current_words) > 0:
                    overlap = len(seen_words.intersection(current_words))
                    similarity = overlap / max(len(seen_words), len(current_words))
                    
                    if similarity > 0.6:  # Lowered threshold for better deduplication
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_risks.append(risk)
                seen_findings.add(normalized)
        
        return unique_risks
    
    def to_dict(self, result: AnalysisResult) -> Dict[str, Any]:
        """
        Convert AnalysisResult to dictionary format.
        
        Args:
            result: AnalysisResult object
            
        Returns:
            Dictionary representation
        """
        result_dict = asdict(result)
        
        # Convert RiskIndicator objects to dictionaries
        result_dict["risk_indicators"] = [
            asdict(risk) for risk in result.risk_indicators
        ]
        
        return result_dict
    
    def get_engine_info(self) -> Dict[str, Any]:
        """
        Get information about the analysis engine configuration.
        
        Returns:
            Dictionary with engine information
        """
        return {
            "engine_version": "1.0.0",
            "llm_service_info": self.llm_service.get_model_info(),
            "supported_formats": list(self.document_parser.get_supported_formats()),
            "safety_filters_count": len(self.safety_filters),
            "risk_categories": list(self.risk_patterns.keys())
        }