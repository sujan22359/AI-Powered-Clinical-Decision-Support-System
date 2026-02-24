"""
Measurement Extraction Service
Extracts quantitative measurements from medical images and reports
"""

import re
from typing import Dict, List, Optional, Tuple
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class MeasurementExtractor:
    """
    Extracts measurements from medical images and reports
    Supports: tumor size, organ volume, cardiac metrics, etc.
    """
    
    # Common measurement patterns
    MEASUREMENT_PATTERNS = {
        'size': r'(\d+\.?\d*)\s*(mm|cm|m)\s*[xX×]\s*(\d+\.?\d*)\s*(mm|cm|m)',
        'diameter': r'diameter[:\s]+(\d+\.?\d*)\s*(mm|cm|m)',
        'volume': r'volume[:\s]+(\d+\.?\d*)\s*(ml|cc|cm3|L)',
        'area': r'area[:\s]+(\d+\.?\d*)\s*(mm2|cm2|m2)',
        'length': r'length[:\s]+(\d+\.?\d*)\s*(mm|cm|m)',
        'width': r'width[:\s]+(\d+\.?\d*)\s*(mm|cm|m)',
        'height': r'height[:\s]+(\d+\.?\d*)\s*(mm|cm|m)',
        'thickness': r'thickness[:\s]+(\d+\.?\d*)\s*(mm|cm|m)',
    }
    
    # Unit conversions to mm
    UNIT_CONVERSIONS = {
        'mm': 1.0,
        'cm': 10.0,
        'm': 1000.0,
        'ml': 1.0,
        'cc': 1.0,
        'cm3': 1.0,
        'L': 1000.0,
        'mm2': 1.0,
        'cm2': 100.0,
        'm2': 1000000.0,
    }
    
    def __init__(self):
        """Initialize measurement extractor"""
        logger.info("Measurement Extractor initialized")
    
    def extract_measurements(self, text: str, image_type: str = "auto") -> Dict:
        """
        Extract all measurements from text
        
        Args:
            text: Text containing measurements (from image analysis or report)
            image_type: Type of medical image
        
        Returns:
            Dictionary with extracted measurements
        """
        try:
            measurements = {
                'tumor_measurements': self._extract_tumor_measurements(text),
                'organ_measurements': self._extract_organ_measurements(text),
                'cardiac_measurements': self._extract_cardiac_measurements(text),
                'bone_measurements': self._extract_bone_measurements(text),
                'general_measurements': self._extract_general_measurements(text),
                'summary': []
            }
            
            # Create summary
            total_count = sum(len(v) for v in measurements.values() if isinstance(v, list))
            measurements['summary'] = self._create_measurement_summary(measurements)
            measurements['count'] = total_count
            
            logger.info(f"Extracted {total_count} measurements")
            return measurements
            
        except Exception as e:
            logger.error(f"Measurement extraction failed: {str(e)}")
            return {'error': str(e), 'count': 0}
    
    def _extract_tumor_measurements(self, text: str) -> List[Dict]:
        """Extract tumor-related measurements"""
        measurements = []
        
        # Look for tumor size patterns
        tumor_patterns = [
            r'tumor[:\s]+(\d+\.?\d*)\s*(mm|cm)\s*[xX×]\s*(\d+\.?\d*)\s*(mm|cm)',
            r'mass[:\s]+(\d+\.?\d*)\s*(mm|cm)\s*[xX×]\s*(\d+\.?\d*)\s*(mm|cm)',
            r'lesion[:\s]+(\d+\.?\d*)\s*(mm|cm)\s*[xX×]\s*(\d+\.?\d*)\s*(mm|cm)',
            r'nodule[:\s]+(\d+\.?\d*)\s*(mm|cm)',
        ]
        
        for pattern in tumor_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 4:
                    measurements.append({
                        'type': 'tumor_size',
                        'length': float(match.group(1)),
                        'length_unit': match.group(2),
                        'width': float(match.group(3)),
                        'width_unit': match.group(4),
                        'length_mm': self._convert_to_mm(float(match.group(1)), match.group(2)),
                        'width_mm': self._convert_to_mm(float(match.group(3)), match.group(4)),
                    })
                elif len(match.groups()) >= 2:
                    measurements.append({
                        'type': 'tumor_diameter',
                        'diameter': float(match.group(1)),
                        'unit': match.group(2),
                        'diameter_mm': self._convert_to_mm(float(match.group(1)), match.group(2)),
                    })
        
        return measurements
    
    def _extract_organ_measurements(self, text: str) -> List[Dict]:
        """Extract organ-related measurements"""
        measurements = []
        
        # Organ size patterns
        organs = ['liver', 'kidney', 'spleen', 'heart', 'lung', 'brain']
        
        for organ in organs:
            # Size pattern
            pattern = rf'{organ}[:\s]+(\d+\.?\d*)\s*(mm|cm)\s*[xX×]\s*(\d+\.?\d*)\s*(mm|cm)'
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                measurements.append({
                    'type': f'{organ}_size',
                    'organ': organ,
                    'length': float(match.group(1)),
                    'width': float(match.group(3)),
                    'unit': match.group(2),
                    'length_mm': self._convert_to_mm(float(match.group(1)), match.group(2)),
                    'width_mm': self._convert_to_mm(float(match.group(3)), match.group(4)),
                })
        
        return measurements
    
    def _extract_cardiac_measurements(self, text: str) -> List[Dict]:
        """Extract cardiac measurements"""
        measurements = []
        
        # Cardiothoracic ratio
        ctr_pattern = r'cardiothoracic\s+ratio[:\s]+(\d+\.?\d*)%?'
        matches = re.finditer(ctr_pattern, text, re.IGNORECASE)
        for match in matches:
            ctr_value = float(match.group(1))
            measurements.append({
                'type': 'cardiothoracic_ratio',
                'value': ctr_value,
                'unit': 'ratio',
                'normal': ctr_value < 50,
                'interpretation': 'Normal' if ctr_value < 50 else 'Enlarged heart'
            })
        
        # Ejection fraction
        ef_pattern = r'ejection\s+fraction[:\s]+(\d+\.?\d*)%'
        matches = re.finditer(ef_pattern, text, re.IGNORECASE)
        for match in matches:
            ef_value = float(match.group(1))
            measurements.append({
                'type': 'ejection_fraction',
                'value': ef_value,
                'unit': '%',
                'normal': ef_value >= 50,
                'interpretation': self._interpret_ejection_fraction(ef_value)
            })
        
        return measurements
    
    def _extract_bone_measurements(self, text: str) -> List[Dict]:
        """Extract bone-related measurements"""
        measurements = []
        
        # Fracture displacement
        displacement_pattern = r'displacement[:\s]+(\d+\.?\d*)\s*(mm|cm)'
        matches = re.finditer(displacement_pattern, text, re.IGNORECASE)
        for match in matches:
            measurements.append({
                'type': 'fracture_displacement',
                'value': float(match.group(1)),
                'unit': match.group(2),
                'value_mm': self._convert_to_mm(float(match.group(1)), match.group(2)),
            })
        
        # Angulation
        angulation_pattern = r'angulation[:\s]+(\d+\.?\d*)\s*degrees?'
        matches = re.finditer(angulation_pattern, text, re.IGNORECASE)
        for match in matches:
            measurements.append({
                'type': 'fracture_angulation',
                'value': float(match.group(1)),
                'unit': 'degrees',
            })
        
        return measurements
    
    def _extract_general_measurements(self, text: str) -> List[Dict]:
        """Extract general measurements"""
        measurements = []
        
        for measure_type, pattern in self.MEASUREMENT_PATTERNS.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 2:
                    measurements.append({
                        'type': measure_type,
                        'value': float(match.group(1)),
                        'unit': match.group(2),
                    })
        
        return measurements
    
    def _convert_to_mm(self, value: float, unit: str) -> float:
        """Convert measurement to millimeters"""
        conversion_factor = self.UNIT_CONVERSIONS.get(unit.lower(), 1.0)
        return value * conversion_factor
    
    def _interpret_ejection_fraction(self, ef: float) -> str:
        """Interpret ejection fraction value"""
        if ef >= 55:
            return "Normal"
        elif ef >= 50:
            return "Mildly reduced"
        elif ef >= 40:
            return "Moderately reduced"
        elif ef >= 30:
            return "Severely reduced"
        else:
            return "Critically reduced"
    
    def _create_measurement_summary(self, measurements: Dict) -> List[str]:
        """Create human-readable summary of measurements"""
        summary = []
        
        # Tumor measurements
        tumors = measurements.get('tumor_measurements', [])
        if tumors:
            summary.append(f"Found {len(tumors)} tumor/mass measurement(s)")
        
        # Organ measurements
        organs = measurements.get('organ_measurements', [])
        if organs:
            summary.append(f"Found {len(organs)} organ measurement(s)")
        
        # Cardiac measurements
        cardiac = measurements.get('cardiac_measurements', [])
        if cardiac:
            summary.append(f"Found {len(cardiac)} cardiac measurement(s)")
        
        # Bone measurements
        bones = measurements.get('bone_measurements', [])
        if bones:
            summary.append(f"Found {len(bones)} bone/fracture measurement(s)")
        
        return summary
    
    def calculate_tumor_volume(self, length_mm: float, width_mm: float, height_mm: Optional[float] = None) -> float:
        """
        Calculate tumor volume using ellipsoid formula
        
        Args:
            length_mm: Length in millimeters
            width_mm: Width in millimeters
            height_mm: Height in millimeters (optional, assumes width if not provided)
        
        Returns:
            Volume in cubic millimeters
        """
        if height_mm is None:
            height_mm = width_mm
        
        # Ellipsoid volume formula: (4/3) * π * a * b * c
        import math
        volume = (4/3) * math.pi * (length_mm/2) * (width_mm/2) * (height_mm/2)
        return volume
    
    def assess_tumor_growth(self, previous_size_mm: float, current_size_mm: float, days_between: int) -> Dict:
        """
        Assess tumor growth rate
        
        Args:
            previous_size_mm: Previous measurement in mm
            current_size_mm: Current measurement in mm
            days_between: Days between measurements
        
        Returns:
            Growth assessment dictionary
        """
        size_change = current_size_mm - previous_size_mm
        percent_change = (size_change / previous_size_mm) * 100 if previous_size_mm > 0 else 0
        growth_rate_per_month = (size_change / days_between) * 30 if days_between > 0 else 0
        
        # Interpret growth
        if percent_change > 20:
            status = "Rapid growth"
            urgency = "high"
        elif percent_change > 10:
            status = "Moderate growth"
            urgency = "medium"
        elif percent_change > 0:
            status = "Slow growth"
            urgency = "low"
        elif percent_change == 0:
            status = "Stable"
            urgency = "low"
        else:
            status = "Decreasing"
            urgency = "low"
        
        return {
            'previous_size_mm': previous_size_mm,
            'current_size_mm': current_size_mm,
            'size_change_mm': size_change,
            'percent_change': percent_change,
            'growth_rate_mm_per_month': growth_rate_per_month,
            'status': status,
            'urgency': urgency,
            'days_between': days_between
        }


# Example usage
if __name__ == "__main__":
    extractor = MeasurementExtractor()
    
    # Test text
    test_text = """
    Findings: A mass measuring 25mm x 18mm is identified in the right upper lobe.
    The tumor shows irregular borders. Cardiothoracic ratio is 52%.
    Fracture with 5mm displacement and 15 degrees angulation noted.
    """
    
    measurements = extractor.extract_measurements(test_text)
    print("Extracted measurements:")
    print(f"Total: {measurements['count']}")
    print(f"Summary: {measurements['summary']}")
