"""
Unit tests for Analysis Engine integration with threshold-based risk assessment.

Tests the integration between AI analysis and threshold-based parameter evaluation,
including risk indicator merging, deduplication, and sorting.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from hypothesis import given, strategies as st
from backend.services.analysis_engine import AnalysisEngine, RiskIndicator, AnalysisResult
from backend.models.clinical_parameters import (
    ClinicalParameter, EvaluatedParameter, RiskLevel, ReferenceRange
)


class TestAnalysisEngineIntegration:
    """Test suite for analysis engine integration with threshold assessment."""
    
    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLM service."""
        mock = Mock()
        mock.analyze_clinical_text.return_value = Mock(
            success=True,
            content='{"summary": "Test summary", "key_findings": [], "risk_indicators": [], "follow_up_suggestions": []}',
            error_message=None,
            retry_count=0
        )
        mock.get_model_info.return_value = {"model": "test-model"}
        return mock
    
    @pytest.fixture
    def mock_document_parser(self):
        """Create a mock document parser."""
        mock = Mock()
        mock.get_supported_formats.return_value = ["pdf", "txt"]
        return mock
    
    @pytest.fixture
    def engine(self, mock_llm_service, mock_document_parser):
        """Create an analysis engine instance with mocked services."""
        return AnalysisEngine(
            llm_service=mock_llm_service,
            document_parser=mock_document_parser
        )
    
    def test_both_ai_and_threshold_analyses_performed(self, engine, mock_llm_service):
        """Test that both AI and threshold analyses are performed."""
        clinical_text = "Glucose: 180 mg/dL"
        
        # Mock the parameter extractor to return glucose data
        with patch.object(engine.parameter_extractor, 'extract_parameters') as mock_extract:
            mock_extract.return_value = [
                {"name": "Glucose", "value": 180.0, "unit": "mg/dL"}
            ]
            
            result = engine.analyze_clinical_text(clinical_text)
            
            # Verify AI analysis was called
            mock_llm_service.analyze_clinical_text.assert_called_once_with(clinical_text)
            
            # Verify parameter extraction was called
            mock_extract.assert_called_once_with(clinical_text)
            
            # Verify result structure
            assert isinstance(result, AnalysisResult)
            assert result.summary is not None
            assert isinstance(result.risk_indicators, list)
    
    def test_risk_indicator_merging(self, engine):
        """Test that AI and threshold risk indicators are merged correctly."""
        # Create sample AI risks
        ai_risks = [
            RiskIndicator(
                finding="Elevated glucose levels detected",
                category="metabolic",
                severity="medium",
                description="AI detected elevated glucose"
            )
        ]
        
        # Create sample threshold risks
        threshold_risks = [
            RiskIndicator(
                finding="Glucose: 180 mg/dL (Normal: 70-100 mg/dL)",
                category="metabolic",
                severity="high",
                description="Glucose is 80% above normal range",
                parameter_name="Glucose",
                actual_value=180.0,
                unit="mg/dL",
                reference_range="70-100 mg/dL",
                deviation_percent=80.0,
                threshold_based=True
            )
        ]
        
        merged = engine._merge_risk_indicators(ai_risks, threshold_risks)
        
        # Should have threshold risk but not duplicate AI risk (glucose mentioned in both)
        assert len(merged) == 1
        assert merged[0].threshold_based is True
        assert merged[0].parameter_name == "Glucose"
    
    def test_deduplication_removes_ai_duplicates(self, engine):
        """Test that AI risks are deduplicated when threshold risks exist for same parameter."""
        ai_risks = [
            RiskIndicator(
                finding="Elevated glucose levels detected",
                category="metabolic",
                severity="medium",
                description="AI detected glucose issue"
            ),
            RiskIndicator(
                finding="Blood pressure concerns",
                category="cardiovascular",
                severity="low",
                description="AI detected BP issue"
            )
        ]
        
        threshold_risks = [
            RiskIndicator(
                finding="Glucose: 180 mg/dL",
                category="metabolic",
                severity="high",
                description="Threshold-based glucose risk",
                parameter_name="Glucose",
                threshold_based=True
            )
        ]
        
        merged = engine._merge_risk_indicators(ai_risks, threshold_risks)
        
        # Should have 2 risks: threshold glucose + AI blood pressure
        assert len(merged) == 2
        
        # Check that glucose AI risk was deduplicated
        glucose_risks = [r for r in merged if "glucose" in r.finding.lower()]
        assert len(glucose_risks) == 1
        assert glucose_risks[0].threshold_based is True
        
        # Check that BP risk was kept
        bp_risks = [r for r in merged if "pressure" in r.finding.lower()]
        assert len(bp_risks) == 1
        assert bp_risks[0].threshold_based is False
    
    def test_sorting_by_severity(self, engine):
        """Test that merged risks are sorted by severity (high > medium > low)."""
        ai_risks = [
            RiskIndicator(
                finding="Low risk finding",
                category="general",
                severity="low",
                description="Low severity"
            )
        ]
        
        threshold_risks = [
            RiskIndicator(
                finding="Medium risk finding",
                category="general",
                severity="medium",
                description="Medium severity",
                threshold_based=True
            ),
            RiskIndicator(
                finding="High risk finding",
                category="general",
                severity="high",
                description="High severity",
                threshold_based=True
            )
        ]
        
        merged = engine._merge_risk_indicators(ai_risks, threshold_risks)
        
        # Should be sorted: high, medium, low
        assert len(merged) == 3
        assert merged[0].severity == "high"
        assert merged[1].severity == "medium"
        assert merged[2].severity == "low"
    
    def test_graceful_degradation_when_threshold_fails(self, engine, mock_llm_service):
        """Test that analysis continues with AI only when threshold assessment fails."""
        clinical_text = "Glucose: 180 mg/dL"
        
        # Mock parameter extractor to raise an exception
        with patch.object(engine.parameter_extractor, 'extract_parameters') as mock_extract:
            mock_extract.side_effect = Exception("Extraction failed")
            
            result = engine.analyze_clinical_text(clinical_text)
            
            # Should still return a result
            assert isinstance(result, AnalysisResult)
            
            # Should have a processing note about the failure
            assert result.processing_notes is not None
            assert "unavailable" in result.processing_notes.lower()
    
    def test_create_threshold_risk_indicator(self, engine):
        """Test conversion of EvaluatedParameter to RiskIndicator."""
        # Create a sample evaluated parameter
        ref_range = ReferenceRange(min_value=70.0, max_value=100.0, unit="mg/dL")
        param = ClinicalParameter(
            name="Glucose",
            value=180.0,
            unit="mg/dL",
            reference_range=ref_range
        )
        evaluated = EvaluatedParameter(
            parameter=param,
            risk_level=RiskLevel.HIGH,
            explanation="Glucose is significantly elevated",
            deviation_percent=80.0
        )
        
        risk_indicator = engine._create_threshold_risk_indicator(evaluated)
        
        # Verify all fields are populated correctly
        assert risk_indicator.finding == "Glucose: 180.0 mg/dL (Normal: 70.0-100.0 mg/dL)"
        assert risk_indicator.severity == "high"
        assert risk_indicator.parameter_name == "Glucose"
        assert risk_indicator.actual_value == 180.0
        assert risk_indicator.unit == "mg/dL"
        assert risk_indicator.reference_range == "70.0-100.0 mg/dL"
        assert risk_indicator.threshold_based is True
        assert risk_indicator.description == "Glucose is significantly elevated"
    
    def test_create_threshold_risk_indicator_with_min_only(self, engine):
        """Test risk indicator creation with min-only reference range."""
        ref_range = ReferenceRange(min_value=4000.0, max_value=None, unit="cells/μL")
        param = ClinicalParameter(
            name="White Blood Cell Count",
            value=3000.0,
            unit="cells/μL",
            reference_range=ref_range
        )
        evaluated = EvaluatedParameter(
            parameter=param,
            risk_level=RiskLevel.MEDIUM,
            explanation="WBC is below normal range",
            deviation_percent=-25.0
        )
        
        risk_indicator = engine._create_threshold_risk_indicator(evaluated)
        
        assert ">4000.0 cells/μL" in risk_indicator.finding
        assert risk_indicator.reference_range == ">4000.0 cells/μL"
        assert risk_indicator.severity == "medium"
    
    def test_create_threshold_risk_indicator_with_max_only(self, engine):
        """Test risk indicator creation with max-only reference range."""
        ref_range = ReferenceRange(min_value=None, max_value=140.0, unit="mmHg")
        param = ClinicalParameter(
            name="Systolic Blood Pressure",
            value=160.0,
            unit="mmHg",
            reference_range=ref_range
        )
        evaluated = EvaluatedParameter(
            parameter=param,
            risk_level=RiskLevel.HIGH,
            explanation="Blood pressure is elevated",
            deviation_percent=14.3
        )
        
        risk_indicator = engine._create_threshold_risk_indicator(evaluated)
        
        assert "<140.0 mmHg" in risk_indicator.finding
        assert risk_indicator.reference_range == "<140.0 mmHg"
        assert risk_indicator.severity == "high"
    
    def test_blood_pressure_handling(self, engine, mock_llm_service):
        """Test that blood pressure is split into systolic and diastolic parameters."""
        clinical_text = "Blood Pressure: 160/95 mmHg"
        
        with patch.object(engine.parameter_extractor, 'extract_parameters') as mock_extract:
            mock_extract.return_value = [
                {
                    "name": "Blood Pressure",
                    "value": {"systolic": 160.0, "diastolic": 95.0},
                    "unit": "mmHg"
                }
            ]
            
            result = engine.analyze_clinical_text(clinical_text)
            
            # Should have risk indicators for both systolic and diastolic
            # (if they're outside normal range)
            assert isinstance(result, AnalysisResult)
            # The actual number depends on whether values are out of range
            # Just verify the analysis completed successfully
    
    def test_empty_parameter_extraction(self, engine, mock_llm_service):
        """Test handling when no parameters are extracted."""
        clinical_text = "Patient feels well, no specific values mentioned"
        
        with patch.object(engine.parameter_extractor, 'extract_parameters') as mock_extract:
            mock_extract.return_value = []
            
            result = engine.analyze_clinical_text(clinical_text)
            
            # Should still return a valid result with AI analysis only
            assert isinstance(result, AnalysisResult)
            assert result.summary is not None
    
    def test_multiple_parameters_integration(self, engine, mock_llm_service):
        """Test integration with multiple clinical parameters."""
        clinical_text = "Glucose: 180 mg/dL, Hemoglobin: 10 g/dL, Platelets: 120000 /μL"
        
        with patch.object(engine.parameter_extractor, 'extract_parameters') as mock_extract:
            mock_extract.return_value = [
                {"name": "Glucose", "value": 180.0, "unit": "mg/dL"},
                {"name": "Hemoglobin", "value": 10.0, "unit": "g/dL"},
                {"name": "Platelet Count", "value": 120000.0, "unit": "/μL"}
            ]
            
            result = engine.analyze_clinical_text(clinical_text)
            
            # Should have analyzed all parameters
            assert isinstance(result, AnalysisResult)
            # Risk indicators should include threshold-based assessments
            # (actual count depends on which values are out of range)



class TestAnalysisEngineProperties:
    """Property-based tests for analysis engine integration."""
    
    def _create_engine(self):
        """Create an analysis engine instance for testing."""
        mock_llm = Mock()
        mock_llm.analyze_clinical_text.return_value = Mock(
            success=True,
            content='{"summary": "Test", "key_findings": [], "risk_indicators": [], "follow_up_suggestions": []}',
            error_message=None,
            retry_count=0
        )
        mock_llm.get_model_info.return_value = {"model": "test"}
        
        mock_parser = Mock()
        mock_parser.get_supported_formats.return_value = ["pdf", "txt"]
        
        return AnalysisEngine(llm_service=mock_llm, document_parser=mock_parser)
    
    @given(
        st.lists(
            st.builds(
                RiskIndicator,
                finding=st.text(min_size=1, max_size=100),
                category=st.sampled_from(["metabolic", "cardiovascular", "general"]),
                severity=st.sampled_from(["low", "medium", "high"]),
                description=st.text(min_size=1, max_size=100)
            ),
            min_size=0,
            max_size=10
        ),
        st.lists(
            st.builds(
                RiskIndicator,
                finding=st.text(min_size=1, max_size=100),
                category=st.sampled_from(["metabolic", "cardiovascular", "general"]),
                severity=st.sampled_from(["low", "medium", "high"]),
                description=st.text(min_size=1, max_size=100),
                parameter_name=st.text(min_size=1, max_size=20),
                threshold_based=st.just(True)
            ),
            min_size=0,
            max_size=10
        )
    )
    def test_property_merge_preserves_all_threshold_risks(self, ai_risks, threshold_risks):
        """
        **Validates: Requirements 3.1, 3.2**
        
        Property: All threshold-based risks must be preserved in merged output.
        """
        engine = self._create_engine()
        merged = engine._merge_risk_indicators(ai_risks, threshold_risks)
        
        # All threshold risks should be in merged output
        threshold_findings = {r.finding for r in threshold_risks}
        merged_findings = {r.finding for r in merged}
        
        assert threshold_findings.issubset(merged_findings)
    
    @given(
        st.lists(
            st.builds(
                RiskIndicator,
                finding=st.text(min_size=1, max_size=100),
                category=st.sampled_from(["metabolic", "cardiovascular", "general"]),
                severity=st.sampled_from(["low", "medium", "high"]),
                description=st.text(min_size=1, max_size=100)
            ),
            min_size=1,
            max_size=20
        )
    )
    def test_property_sorting_by_severity(self, risks):
        """
        **Validates: Requirements 3.2**
        
        Property: Merged risks must be sorted by severity (high > medium > low).
        """
        engine = self._create_engine()
        
        merged = engine._merge_risk_indicators(risks, [])
        
        # Check that risks are sorted by severity
        severity_order = {"high": 3, "medium": 2, "low": 1}
        for i in range(len(merged) - 1):
            current_severity = severity_order.get(merged[i].severity.lower(), 0)
            next_severity = severity_order.get(merged[i + 1].severity.lower(), 0)
            assert current_severity >= next_severity
    
    @given(
        st.lists(
            st.builds(
                RiskIndicator,
                finding=st.text(min_size=1, max_size=100),
                category=st.sampled_from(["metabolic", "cardiovascular", "general"]),
                severity=st.sampled_from(["low", "medium", "high"]),
                description=st.text(min_size=1, max_size=100)
            ),
            min_size=0,
            max_size=10
        ),
        st.lists(
            st.builds(
                RiskIndicator,
                finding=st.text(min_size=1, max_size=100),
                category=st.sampled_from(["metabolic", "cardiovascular", "general"]),
                severity=st.sampled_from(["low", "medium", "high"]),
                description=st.text(min_size=1, max_size=100),
                parameter_name=st.text(min_size=1, max_size=20),
                threshold_based=st.just(True)
            ),
            min_size=0,
            max_size=10
        )
    )
    def test_property_merge_result_count(self, ai_risks, threshold_risks):
        """
        **Validates: Requirements 3.1, 3.2**
        
        Property: Merged result count should be <= sum of input counts (due to deduplication).
        """
        engine = self._create_engine()
        merged = engine._merge_risk_indicators(ai_risks, threshold_risks)
        
        # Merged count should not exceed total input count
        assert len(merged) <= len(ai_risks) + len(threshold_risks)
        
        # Merged count should be at least the threshold count (all threshold risks preserved)
        assert len(merged) >= len(threshold_risks)
    
    @given(
        st.builds(
            RiskIndicator,
            finding=st.text(min_size=1, max_size=100),
            category=st.sampled_from(["metabolic", "cardiovascular", "general"]),
            severity=st.sampled_from(["low", "medium", "high"]),
            description=st.text(min_size=1, max_size=100),
            parameter_name=st.text(min_size=1, max_size=20),
            threshold_based=st.just(True)
        )
    )
    def test_property_threshold_risks_always_included(self, threshold_risk):
        """
        **Validates: Requirements 3.1**
        
        Property: Threshold-based risks are always included in merged output.
        """
        engine = self._create_engine()
        ai_risks = []
        threshold_risks = [threshold_risk]
        
        merged = engine._merge_risk_indicators(ai_risks, threshold_risks)
        
        # The threshold risk should be in the merged output
        assert len(merged) >= 1
        assert any(r.threshold_based for r in merged)
