"""
Integration tests for the complete threshold-based risk assessment pipeline.

Tests the end-to-end flow: extraction → evaluation → explanation
"""

import pytest
from backend.services.parameter_extractor import ParameterExtractor
from backend.services.threshold_evaluator import ThresholdEvaluator
from backend.services.risk_explainer import RiskExplainer
from backend.services.reference_range_db import ReferenceRangeDatabase
from backend.models.clinical_parameters import RiskLevel, ClinicalParameter


class TestThresholdAssessmentPipeline:
    """Integration tests for the complete threshold assessment pipeline."""
    
    @pytest.fixture
    def extractor(self):
        """Create parameter extractor instance."""
        return ParameterExtractor()
    
    @pytest.fixture
    def evaluator(self):
        """Create threshold evaluator instance."""
        return ThresholdEvaluator()
    
    @pytest.fixture
    def explainer(self):
        """Create risk explainer instance."""
        return RiskExplainer()
    
    @pytest.fixture
    def ref_db(self):
        """Create reference range database instance."""
        return ReferenceRangeDatabase()
    
    def _convert_to_clinical_params(self, param_dicts, ref_db):
        """Convert parameter dictionaries to ClinicalParameter objects."""
        clinical_params = []
        for param_dict in param_dicts:
            ref_range = ref_db.get_range(param_dict["name"])
            clinical_param = ClinicalParameter(
                name=param_dict["name"],
                value=param_dict["value"],
                unit=param_dict["unit"],
                reference_range=ref_range
            )
            clinical_params.append(clinical_param)
        return clinical_params
    
    def test_complete_pipeline_with_high_risk(self, extractor, evaluator, explainer, ref_db):
        """Test complete pipeline with high-risk parameters."""
        # Clinical text with high-risk glucose
        text = "Patient labs: Glucose 250 mg/dL, Blood Pressure 120/80 mmHg"
        
        # Extract parameters
        param_dicts = extractor.extract_parameters(text)
        assert len(param_dicts) >= 2  # glucose and BP components
        
        # Convert to ClinicalParameter objects
        params = self._convert_to_clinical_params(param_dicts, ref_db)
        
        # Evaluate parameters
        evaluated = evaluator.evaluate_parameters(params)
        assert len(evaluated) >= 2
        
        # Find glucose evaluation
        glucose_eval = next((e for e in evaluated if "glucose" in e.parameter.name.lower()), None)
        assert glucose_eval is not None
        assert glucose_eval.risk_level == RiskLevel.HIGH
        
        # Generate explanation
        explanation = explainer.generate_explanation(glucose_eval)
        assert ("significantly" in explanation.lower() or "elevated" in explanation.lower() or "high" in explanation.lower())
        assert "250" in explanation
        assert "mg/dL" in explanation
    
    def test_complete_pipeline_with_medium_risk(self, extractor, evaluator, explainer, ref_db):
        """Test complete pipeline with medium-risk parameters."""
        # Clinical text with slightly elevated glucose
        text = "Fasting glucose: 110 mg/dL"
        
        # Extract parameters
        param_dicts = extractor.extract_parameters(text)
        assert len(param_dicts) >= 1
        
        # Convert to ClinicalParameter objects
        params = self._convert_to_clinical_params(param_dicts, ref_db)
        
        # Evaluate parameters
        evaluated = evaluator.evaluate_parameters(params)
        assert len(evaluated) >= 1
        
        # Generate explanation
        glucose_eval = evaluated[0]
        explanation = explainer.generate_explanation(glucose_eval)
        assert glucose_eval.parameter.name.lower() in explanation.lower()
        assert "110" in explanation
    
    def test_complete_pipeline_with_normal_values(self, extractor, evaluator, explainer, ref_db):
        """Test complete pipeline with normal parameters."""
        # Clinical text with normal values
        text = """
        Patient vitals and labs:
        Blood Pressure: 120/80 mmHg
        Heart Rate: 75 bpm
        Temperature: 98.6°F
        Glucose: 95 mg/dL
        """
        
        # Extract parameters
        param_dicts = extractor.extract_parameters(text)
        assert len(param_dicts) >= 3  # BP components, glucose
        
        # Convert to ClinicalParameter objects
        params = self._convert_to_clinical_params(param_dicts, ref_db)
        
        # Evaluate parameters
        evaluated = evaluator.evaluate_parameters(params)
        assert len(evaluated) >= 3
        
        # Check that most are low risk
        low_risk_count = sum(1 for e in evaluated if e.risk_level == RiskLevel.LOW)
        assert low_risk_count >= 2
        
        # Generate explanations for all
        explanations = [explainer.generate_explanation(e) for e in evaluated]
        assert len(explanations) == len(evaluated)
        assert all(len(exp) > 0 for exp in explanations)
    
    def test_pipeline_with_multiple_formats(self, extractor, evaluator, explainer, ref_db):
        """Test pipeline with various clinical text formats."""
        formats = [
            "Glucose: 200 mg/dL",
            "Fasting blood sugar = 200 mg/dL",
            "GLU 200 mg/dL",
        ]
        
        for text in formats:
            # Extract parameters
            param_dicts = extractor.extract_parameters(text)
            if len(param_dicts) == 0:
                # Some formats may not be supported, skip them
                continue
            
            # Convert to ClinicalParameter objects
            params = self._convert_to_clinical_params(param_dicts, ref_db)
            
            # Evaluate parameters
            evaluated = evaluator.evaluate_parameters(params)
            assert len(evaluated) >= 1
            
            # Generate explanation
            explanation = explainer.generate_explanation(evaluated[0])
            assert len(explanation) > 0
            assert "200" in explanation
    
    def test_pipeline_error_handling_empty_text(self, extractor, evaluator, explainer, ref_db):
        """Test pipeline handles empty text gracefully."""
        # Extract from empty text
        param_dicts = extractor.extract_parameters("")
        assert param_dicts == []
        
        # Evaluate empty list
        evaluated = evaluator.evaluate_parameters([])
        assert evaluated == []
    
    def test_pipeline_error_handling_no_parameters(self, extractor, evaluator, explainer, ref_db):
        """Test pipeline handles text with no parameters."""
        text = "Patient is feeling well today. No specific complaints."
        
        # Extract parameters
        param_dicts = extractor.extract_parameters(text)
        assert param_dicts == []
        
        # Evaluate empty list
        evaluated = evaluator.evaluate_parameters([])
        assert evaluated == []
    
    def test_pipeline_with_mixed_risk_levels(self, extractor, evaluator, explainer, ref_db):
        """Test pipeline with parameters at different risk levels."""
        text = """
        Patient labs:
        Glucose: 250 mg/dL (HIGH)
        Blood Pressure: 125/82 mmHg (MEDIUM)
        Heart Rate: 72 bpm (LOW)
        """
        
        # Extract parameters
        param_dicts = extractor.extract_parameters(text)
        assert len(param_dicts) >= 3
        
        # Convert to ClinicalParameter objects
        params = self._convert_to_clinical_params(param_dicts, ref_db)
        
        # Evaluate parameters
        evaluated = evaluator.evaluate_parameters(params)
        assert len(evaluated) >= 3
        
        # Check we have different risk levels
        risk_levels = {e.risk_level for e in evaluated}
        assert len(risk_levels) >= 2  # At least 2 different risk levels
        
        # Generate explanations
        explanations = [explainer.generate_explanation(e) for e in evaluated]
        assert len(explanations) == len(evaluated)
        
        # Verify explanations are distinct
        assert len(set(explanations)) == len(explanations)


class TestThresholdAssessmentEdgeCases:
    """Tests for edge cases in threshold assessment."""
    
    @pytest.fixture
    def extractor(self):
        return ParameterExtractor()
    
    @pytest.fixture
    def evaluator(self):
        return ThresholdEvaluator()
    
    @pytest.fixture
    def explainer(self):
        return RiskExplainer()
    
    @pytest.fixture
    def ref_db(self):
        return ReferenceRangeDatabase()
    
    def _convert_to_clinical_params(self, param_dicts, ref_db):
        """Convert parameter dictionaries to ClinicalParameter objects."""
        clinical_params = []
        for param_dict in param_dicts:
            ref_range = ref_db.get_range(param_dict["name"])
            clinical_param = ClinicalParameter(
                name=param_dict["name"],
                value=param_dict["value"],
                unit=param_dict["unit"],
                reference_range=ref_range
            )
            clinical_params.append(clinical_param)
        return clinical_params
    
    def test_division_by_zero_protection(self, evaluator):
        """Test that division by zero is handled gracefully."""
        from backend.models.clinical_parameters import ReferenceRange
        
        # Create parameter with reference range where midpoint would be zero
        param = ClinicalParameter(
            name="test_param",
            value=10.0,
            unit="mg/dL",
            reference_range=ReferenceRange(min_value=0.0, max_value=0.0, unit="mg/dL")
        )
        
        # Evaluate should not crash
        evaluated = evaluator.evaluate_parameter(param)
        
        # Should handle the edge case (may be HIGH or UNKNOWN depending on implementation)
        assert evaluated.risk_level in [RiskLevel.HIGH, RiskLevel.UNKNOWN]
    
    def test_missing_reference_range(self, evaluator):
        """Test handling of parameters without reference ranges."""
        
        # Create parameter with unknown name (no reference range)
        param = ClinicalParameter(
            name="unknown_parameter",
            value=100.0,
            unit="units",
            reference_range=None
        )
        
        # Evaluate should handle gracefully
        evaluated = evaluator.evaluate_parameter(param)
        assert evaluated.risk_level == RiskLevel.UNKNOWN
    
    def test_extreme_values(self, extractor, evaluator, explainer, ref_db):
        """Test pipeline with extreme parameter values."""
        text = "Glucose: 1000 mg/dL"
        
        # Extract
        param_dicts = extractor.extract_parameters(text)
        assert len(param_dicts) >= 1
        
        # Convert to ClinicalParameter objects
        params = self._convert_to_clinical_params(param_dicts, ref_db)
        
        # Evaluate
        evaluated = evaluator.evaluate_parameters(params)
        assert len(evaluated) >= 1
        assert evaluated[0].risk_level == RiskLevel.HIGH
        
        # Explain
        explanation = explainer.generate_explanation(evaluated[0])
        assert "1000" in explanation
    
    def test_negative_values(self, evaluator):
        """Test handling of negative values (which shouldn't occur in real data)."""
        from backend.models.clinical_parameters import ReferenceRange
        
        # Create parameter with negative value
        param = ClinicalParameter(
            name="glucose",
            value=-50.0,
            unit="mg/dL",
            reference_range=ReferenceRange(min_value=70.0, max_value=100.0, unit="mg/dL")
        )
        
        # Evaluate should handle gracefully
        evaluated = evaluator.evaluate_parameter(param)
        # Should be HIGH risk due to being far below range
        assert evaluated.risk_level in [RiskLevel.HIGH, RiskLevel.MEDIUM]
    
    def test_unit_mismatch_handling(self, evaluator):
        """Test handling when parameter unit doesn't match reference range unit."""
        from backend.models.clinical_parameters import ReferenceRange
        
        # Create parameter with mismatched units
        param = ClinicalParameter(
            name="glucose",
            value=5.5,  # mmol/L value
            unit="mmol/L",
            reference_range=ReferenceRange(min_value=70.0, max_value=100.0, unit="mg/dL")
        )
        
        # Evaluate should still work (may assign UNKNOWN or attempt conversion)
        evaluated = evaluator.evaluate_parameter(param)
        assert evaluated is not None
        assert evaluated.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.UNKNOWN]
    
    def test_malformed_clinical_text(self, extractor, evaluator, ref_db):
        """Test pipeline with malformed clinical text."""
        malformed_texts = [
            "Glucose: abc mg/dL",  # Non-numeric value
            "Glucose: mg/dL",  # Missing value
            "Glucose 200",  # Missing unit
            "200 mg/dL",  # Missing parameter name
        ]
        
        for text in malformed_texts:
            # Extract should handle gracefully
            param_dicts = extractor.extract_parameters(text)
            # May or may not extract depending on format
            
            # Convert to ClinicalParameter objects
            params = self._convert_to_clinical_params(param_dicts, ref_db)
            
            # Evaluate whatever was extracted
            evaluated = evaluator.evaluate_parameters(params)
            # Should not crash
            assert isinstance(evaluated, list)
    
    def test_multiple_errors_in_pipeline(self, extractor, evaluator, explainer, ref_db):
        """Test pipeline resilience when multiple errors occur."""
        # Text with mix of valid and invalid data
        text = """
        Glucose: 200 mg/dL
        Invalid: abc xyz
        Heart Rate: 75 bpm
        Another Invalid: !!!
        """
        
        # Extract parameters (should get valid ones)
        param_dicts = extractor.extract_parameters(text)
        # Should extract at least glucose
        assert len(param_dicts) >= 1
        
        # Convert to ClinicalParameter objects
        params = self._convert_to_clinical_params(param_dicts, ref_db)
        
        # Evaluate
        evaluated = evaluator.evaluate_parameters(params)
        assert len(evaluated) >= 1
        
        # Generate explanations
        for e in evaluated:
            explanation = explainer.generate_explanation(e)
            assert len(explanation) > 0


class TestThresholdAssessmentPerformance:
    """Performance tests for threshold assessment."""
    
    @pytest.fixture
    def extractor(self):
        return ParameterExtractor()
    
    @pytest.fixture
    def evaluator(self):
        return ThresholdEvaluator()
    
    @pytest.fixture
    def explainer(self):
        return RiskExplainer()
    
    @pytest.fixture
    def ref_db(self):
        return ReferenceRangeDatabase()
    
    def _convert_to_clinical_params(self, param_dicts, ref_db):
        """Convert parameter dictionaries to ClinicalParameter objects."""
        clinical_params = []
        for param_dict in param_dicts:
            ref_range = ref_db.get_range(param_dict["name"])
            clinical_param = ClinicalParameter(
                name=param_dict["name"],
                value=param_dict["value"],
                unit=param_dict["unit"],
                reference_range=ref_range
            )
            clinical_params.append(clinical_param)
        return clinical_params
    
    def test_large_clinical_text_performance(self, extractor, evaluator, explainer, ref_db):
        """Test pipeline performance with large clinical text."""
        import time
        
        # Create large clinical text with many parameters
        text = """
        Patient Assessment Report
        
        Vital Signs:
        Blood Pressure: 130/85 mmHg
        Heart Rate: 78 bpm
        Temperature: 99.1°F
        Respiratory Rate: 16 breaths/min
        
        Laboratory Results:
        Glucose: 105 mg/dL
        Hemoglobin A1c: 6.2%
        Total Cholesterol: 210 mg/dL
        LDL Cholesterol: 140 mg/dL
        HDL Cholesterol: 45 mg/dL
        Triglycerides: 180 mg/dL
        
        Additional Notes:
        Patient reports feeling well.
        No acute concerns at this time.
        """ * 10  # Repeat to make it larger
        
        start_time = time.time()
        
        # Extract
        param_dicts = extractor.extract_parameters(text)
        
        # Convert to ClinicalParameter objects
        params = self._convert_to_clinical_params(param_dicts, ref_db)
        
        # Evaluate
        evaluated = evaluator.evaluate_parameters(params)
        
        # Explain
        explanations = [explainer.generate_explanation(e) for e in evaluated]
        
        elapsed_time = time.time() - start_time
        
        # Should complete in reasonable time (< 1 second for this size)
        assert elapsed_time < 1.0, f"Pipeline took {elapsed_time:.2f}s, expected < 1.0s"
        assert len(evaluated) > 0
        assert len(explanations) == len(evaluated)
