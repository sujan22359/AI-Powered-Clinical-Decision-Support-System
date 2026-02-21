"""
Unit tests for reference range database
"""

import pytest
from backend.services.reference_range_db import ReferenceRangeDatabase
from backend.models.clinical_parameters import ReferenceRange


class TestReferenceRangeDatabase:
    """Tests for ReferenceRangeDatabase class"""
    
    def test_database_initialization(self):
        """Test that database initializes with standard reference ranges"""
        db = ReferenceRangeDatabase()
        parameters = db.list_parameters()
        
        # Verify all required parameters are present
        assert "Systolic Blood Pressure" in parameters
        assert "Diastolic Blood Pressure" in parameters
        assert "Blood Glucose" in parameters
        assert "Total Cholesterol" in parameters
        assert "LDL Cholesterol" in parameters
        assert "HDL Cholesterol" in parameters
        assert "Triglycerides" in parameters
        assert "Hemoglobin" in parameters
        assert "WBC" in parameters
        assert "Platelet" in parameters
        assert "Creatinine" in parameters
        
        # Verify we have at least 10 parameters
        assert len(parameters) >= 10
    
    def test_get_range_returns_correct_range(self):
        """Test that get_range returns the correct reference range for valid parameters"""
        db = ReferenceRangeDatabase()
        
        # Test Blood Glucose
        glucose_range = db.get_range("Blood Glucose")
        assert glucose_range is not None
        assert glucose_range.min_value == 70
        assert glucose_range.max_value == 100
        assert glucose_range.unit == "mg/dL"
        
        # Test Systolic Blood Pressure
        systolic_range = db.get_range("Systolic Blood Pressure")
        assert systolic_range is not None
        assert systolic_range.min_value == 90
        assert systolic_range.max_value == 120
        assert systolic_range.unit == "mmHg"
        
        # Test Hemoglobin
        hb_range = db.get_range("Hemoglobin")
        assert hb_range is not None
        assert hb_range.min_value == 13.5
        assert hb_range.max_value == 17.5
        assert hb_range.unit == "g/dL"
    
    def test_get_range_returns_none_for_invalid_parameter(self):
        """Test that get_range returns None for parameters not in database"""
        db = ReferenceRangeDatabase()
        
        result = db.get_range("Invalid Parameter")
        assert result is None
        
        result = db.get_range("NonExistent Test")
        assert result is None
    
    def test_range_based_thresholds(self):
        """Test parameters with both min and max values (range-based thresholds)"""
        db = ReferenceRangeDatabase()
        
        # Blood Glucose has both min and max
        glucose_range = db.get_range("Blood Glucose")
        assert glucose_range.min_value is not None
        assert glucose_range.max_value is not None
        
        # Systolic BP has both min and max
        systolic_range = db.get_range("Systolic Blood Pressure")
        assert systolic_range.min_value is not None
        assert systolic_range.max_value is not None
        
        # WBC has both min and max
        wbc_range = db.get_range("WBC")
        assert wbc_range.min_value is not None
        assert wbc_range.max_value is not None
    
    def test_directional_thresholds_max_only(self):
        """Test parameters with only max value (directional threshold)"""
        db = ReferenceRangeDatabase()
        
        # Total Cholesterol has only max
        cholesterol_range = db.get_range("Total Cholesterol")
        assert cholesterol_range.min_value is None
        assert cholesterol_range.max_value is not None
        assert cholesterol_range.max_value == 200
        
        # LDL Cholesterol has only max
        ldl_range = db.get_range("LDL Cholesterol")
        assert ldl_range.min_value is None
        assert ldl_range.max_value is not None
        assert ldl_range.max_value == 100
        
        # Triglycerides has only max
        trig_range = db.get_range("Triglycerides")
        assert trig_range.min_value is None
        assert trig_range.max_value is not None
        assert trig_range.max_value == 150
    
    def test_directional_thresholds_min_only(self):
        """Test parameters with only min value (directional threshold)"""
        db = ReferenceRangeDatabase()
        
        # HDL Cholesterol has only min
        hdl_range = db.get_range("HDL Cholesterol")
        assert hdl_range.min_value is not None
        assert hdl_range.max_value is None
        assert hdl_range.min_value == 40
    
    def test_all_ranges_have_units(self):
        """Test that all reference ranges have units specified"""
        db = ReferenceRangeDatabase()
        
        for param_name in db.list_parameters():
            ref_range = db.get_range(param_name)
            assert ref_range.unit is not None
            assert ref_range.unit != ""
    
    def test_add_range_new_parameter(self):
        """Test adding a new parameter to the database"""
        db = ReferenceRangeDatabase()
        
        # Add a new parameter
        new_range = ReferenceRange(min_value=3.5, max_value=5.5, unit="mmol/L")
        db.add_range("Potassium", new_range)
        
        # Verify it was added
        assert "Potassium" in db.list_parameters()
        retrieved_range = db.get_range("Potassium")
        assert retrieved_range.min_value == 3.5
        assert retrieved_range.max_value == 5.5
        assert retrieved_range.unit == "mmol/L"
    
    def test_add_range_update_existing_parameter(self):
        """Test updating an existing parameter in the database"""
        db = ReferenceRangeDatabase()
        
        # Get original range
        original_range = db.get_range("Blood Glucose")
        assert original_range.max_value == 100
        
        # Update with new range
        new_range = ReferenceRange(min_value=70, max_value=110, unit="mg/dL")
        db.add_range("Blood Glucose", new_range)
        
        # Verify it was updated
        updated_range = db.get_range("Blood Glucose")
        assert updated_range.max_value == 110
    
    def test_list_parameters_returns_all_parameters(self):
        """Test that list_parameters returns all parameter names"""
        db = ReferenceRangeDatabase()
        
        parameters = db.list_parameters()
        assert isinstance(parameters, list)
        assert len(parameters) >= 10
        
        # Verify it's a list of strings
        for param in parameters:
            assert isinstance(param, str)
    
    def test_has_parameter_returns_true_for_existing(self):
        """Test that has_parameter returns True for existing parameters"""
        db = ReferenceRangeDatabase()
        
        assert db.has_parameter("Blood Glucose") is True
        assert db.has_parameter("Systolic Blood Pressure") is True
        assert db.has_parameter("Hemoglobin") is True
    
    def test_has_parameter_returns_false_for_nonexistent(self):
        """Test that has_parameter returns False for non-existent parameters"""
        db = ReferenceRangeDatabase()
        
        assert db.has_parameter("Invalid Parameter") is False
        assert db.has_parameter("NonExistent Test") is False
    
    def test_database_isolation(self):
        """Test that multiple database instances are independent"""
        db1 = ReferenceRangeDatabase()
        db2 = ReferenceRangeDatabase()
        
        # Add parameter to db1
        new_range = ReferenceRange(min_value=3.5, max_value=5.5, unit="mmol/L")
        db1.add_range("Potassium", new_range)
        
        # Verify db1 has it but db2 doesn't
        assert db1.has_parameter("Potassium") is True
        assert db2.has_parameter("Potassium") is False
    
    def test_all_standard_parameters_have_valid_ranges(self):
        """Test that all standard parameters have valid reference ranges"""
        db = ReferenceRangeDatabase()
        
        for param_name in db.list_parameters():
            ref_range = db.get_range(param_name)
            
            # Verify it's a ReferenceRange object
            assert isinstance(ref_range, ReferenceRange)
            
            # Verify at least one threshold is set
            assert ref_range.min_value is not None or ref_range.max_value is not None
            
            # Verify unit is set
            assert ref_range.unit != ""
