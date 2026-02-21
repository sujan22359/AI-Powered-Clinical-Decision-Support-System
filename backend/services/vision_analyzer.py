"""
Medical Image Analysis Service using Google Gemini Vision
Analyzes medical images (X-rays, CT, MRI) and provides diagnosis and recommendations
"""

import google.generativeai as genai
from PIL import Image
import io
import base64
from typing import Dict, List, Optional, Tuple
import re
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class MedicalImageAnalyzer:
    """
    Analyzes medical images using Google Gemini Vision (VLM)
    Provides diagnosis, identifies issues, and gives follow-up suggestions
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the Medical Image Analyzer
        
        Args:
            api_key: Google Gemini API key
        """
        genai.configure(api_key=api_key)
        # Use Gemini 2.5 Flash for fast, accurate vision analysis
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        logger.info("Medical Image Analyzer initialized with Gemini 2.5 Flash")
    
    def analyze_medical_image(
        self, 
        image_data: bytes,
        image_type: str = "auto",
        clinical_context: Optional[str] = None,
        patient_info: Optional[Dict] = None
    ) -> Dict:
        """
        Analyze a medical image and provide diagnosis with suggestions
        
        Args:
            image_data: Image file bytes
            image_type: Type of medical image (chest_xray, ct_brain, mri, bone_xray, auto)
            clinical_context: Additional clinical information (symptoms, history)
            patient_info: Patient details (age, gender, medical history)
        
        Returns:
            Dictionary containing:
            - diagnosis: Main diagnosis
            - issues: List of identified issues
            - suggestions: Follow-up recommendations
            - confidence: Confidence level (0-100)
            - findings: Detailed findings
            - urgency: Urgency level (low, medium, high, critical)
        """
        try:
            # Load image
            image = Image.open(io.BytesIO(image_data))
            logger.info(f"Analyzing {image_type} image, size: {image.size}")
            
            # Create specialized prompt based on image type
            prompt = self._create_medical_analysis_prompt(
                image_type, 
                clinical_context, 
                patient_info
            )
            
            # Generate analysis using Gemini Vision
            response = self.model.generate_content([prompt, image])
            
            # Parse the response into structured format
            analysis = self._parse_medical_response(response.text, image_type)
            
            logger.info(f"Analysis complete: {analysis['diagnosis']}")
            
            return {
                "success": True,
                "image_type": image_type,
                "diagnosis": analysis["diagnosis"],
                "issues": analysis["issues"],
                "suggestions": analysis["suggestions"],
                "confidence": analysis["confidence"],
                "findings": analysis["findings"],
                "urgency": analysis["urgency"],
                "raw_analysis": response.text
            }
            
        except Exception as e:
            logger.error(f"Image analysis failed: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "diagnosis": "Analysis failed",
                "issues": [],
                "suggestions": ["Please try again or consult a healthcare professional"],
                "confidence": 0,
                "urgency": "medium",
                "findings": {}
            }
    
    def _create_medical_analysis_prompt(
        self, 
        image_type: str, 
        clinical_context: Optional[str],
        patient_info: Optional[Dict]
    ) -> str:
        """
        Create a specialized prompt for medical image analysis
        
        This prompt is carefully designed to get:
        1. Clear diagnosis
        2. Specific issues identified
        3. Actionable follow-up suggestions
        4. Confidence level
        5. Urgency assessment
        """
        
        # Build patient context
        patient_context = ""
        if patient_info:
            age = patient_info.get('age', 'Unknown')
            gender = patient_info.get('gender', 'Unknown')
            history = patient_info.get('medical_history', 'None provided')
            patient_context = f"""
Patient Information:
- Age: {age}
- Gender: {gender}
- Medical History: {history}
"""
        
        # Build clinical context
        clinical_info = ""
        if clinical_context:
            clinical_info = f"""
Clinical Context:
{clinical_context}
"""
        
        # Base prompt - works for all image types
        base_prompt = f"""You are an expert radiologist analyzing a medical image. Provide a comprehensive analysis following this EXACT format:

{patient_context}
{clinical_info}

Please analyze this medical image and provide:

**DIAGNOSIS:**
[Provide the primary diagnosis in one clear sentence]

**ISSUES IDENTIFIED:**
[List specific abnormalities, findings, or concerns - be specific with locations and descriptions]
- Issue 1: [Description]
- Issue 2: [Description]
- Issue 3: [Description]

**CONFIDENCE LEVEL:**
[Provide confidence as a percentage: 0-100%]

**URGENCY:**
[Rate as: LOW, MEDIUM, HIGH, or CRITICAL]

**DETAILED FINDINGS:**
1. Image Quality: [Comment on technical quality]
2. Normal Structures: [What appears normal]
3. Abnormal Findings: [Detailed description of abnormalities]
4. Measurements: [Any relevant measurements]
5. Comparison: [Compare with normal anatomy]

**FOLLOW-UP SUGGESTIONS:**
[Provide 4-6 specific, actionable recommendations]
1. [Immediate action needed]
2. [Medical consultation required]
3. [Additional tests recommended]
4. [Lifestyle or treatment suggestions]
5. [Follow-up timeline]
6. [Warning signs to watch for]

**IMPORTANT NOTES:**
- Be specific about locations (e.g., "right lower lobe" not just "lung")
- Include measurements when visible
- Prioritize patient safety
- Recommend immediate medical attention if critical findings
"""
        
        # Add image-type specific instructions
        specific_instructions = self._get_image_type_instructions(image_type)
        
        return base_prompt + "\n\n" + specific_instructions
    
    def _get_image_type_instructions(self, image_type: str) -> str:
        """
        Get specific instructions for different image types
        """
        
        instructions = {
            "chest_xray": """
**SPECIFIC FOCUS FOR CHEST X-RAY:**
- Lung fields: Check for infiltrates, nodules, masses, pneumothorax
- Heart: Assess size (cardiothoracic ratio), borders
- Mediastinum: Check for widening, masses, lymphadenopathy
- Pleural spaces: Look for effusions, thickening
- Bones: Check ribs, clavicles, spine for fractures or lesions
- Lines/tubes: If present, assess positioning
- Diaphragm: Check position and contour

Common diagnoses to consider:
- Pneumonia (infiltrates)
- Lung cancer (masses, nodules)
- Heart failure (cardiomegaly, pulmonary edema)
- Pneumothorax (collapsed lung)
- Pleural effusion (fluid)
- Tuberculosis (upper lobe infiltrates)
""",
            
            "ct_brain": """
**SPECIFIC FOCUS FOR BRAIN CT:**
- Gray-white matter differentiation
- Ventricles: Size, symmetry, hydrocephalus
- Hemorrhage: Look for blood (appears bright)
- Infarction: Look for dark areas (stroke)
- Mass effect: Midline shift, herniation
- Skull: Check for fractures
- Sinuses: Check for fluid or opacification

Common diagnoses to consider:
- Stroke (ischemic or hemorrhagic)
- Intracranial hemorrhage
- Brain tumor or mass
- Hydrocephalus
- Traumatic brain injury
- Subdural/epidural hematoma
""",
            
            "bone_xray": """
**SPECIFIC FOCUS FOR BONE X-RAY:**
- Fracture lines: Look for breaks, cracks
- Alignment: Check for displacement, angulation
- Joint spaces: Assess for arthritis, dislocation
- Bone density: Look for osteoporosis signs
- Soft tissue: Check for swelling
- Growth plates: In children, check for injuries

Common diagnoses to consider:
- Fractures (simple, comminuted, displaced)
- Dislocations
- Arthritis (osteoarthritis, rheumatoid)
- Bone tumors or lesions
- Osteomyelitis (infection)
- Osteoporosis
""",
            
            "mri": """
**SPECIFIC FOCUS FOR MRI:**
- Signal intensity: T1, T2, FLAIR sequences
- Anatomical structures: Detailed soft tissue
- Lesions: Size, location, characteristics
- Contrast enhancement: If applicable
- Edema: Surrounding inflammation
- Vascular structures: If visible

Common diagnoses to consider:
- Tumors (benign or malignant)
- Multiple sclerosis (white matter lesions)
- Disc herniation (spine MRI)
- Ligament tears (joint MRI)
- Stroke (diffusion-weighted imaging)
- Infections or abscesses
""",
            
            "ultrasound": """
**SPECIFIC FOCUS FOR ULTRASOUND:**
- Echogenicity: Hyper, hypo, or anechoic
- Organ size and shape
- Masses or cysts: Size, characteristics
- Fluid collections: Free fluid, effusions
- Vascular flow: If Doppler used
- Fetal development: If obstetric

Common diagnoses to consider:
- Gallstones (cholecystitis)
- Kidney stones (nephrolithiasis)
- Liver disease (cirrhosis, fatty liver)
- Pregnancy complications
- Abdominal masses or cysts
- Deep vein thrombosis (with Doppler)
""",
            
            "auto": """
**GENERAL MEDICAL IMAGE ANALYSIS:**
First, identify the type of medical image (X-ray, CT, MRI, ultrasound).
Then apply appropriate analysis based on the image type.
Focus on:
- Identifying the body part/region
- Detecting any obvious abnormalities
- Comparing with normal anatomy
- Assessing urgency of findings
"""
        }
        
        return instructions.get(image_type, instructions["auto"])
    
    def _parse_medical_response(self, response_text: str, image_type: str) -> Dict:
        """
        Parse the AI response into structured format
        
        Extracts:
        - Diagnosis
        - Issues list
        - Suggestions list
        - Confidence level
        - Urgency level
        - Detailed findings
        """
        
        result = {
            "diagnosis": "Unable to determine",
            "issues": [],
            "suggestions": [],
            "confidence": 0,
            "urgency": "medium",
            "findings": {}
        }
        
        try:
            # Extract diagnosis
            diagnosis_match = re.search(
                r'\*\*DIAGNOSIS:\*\*\s*\n(.+?)(?=\n\*\*|$)', 
                response_text, 
                re.DOTALL
            )
            if diagnosis_match:
                result["diagnosis"] = diagnosis_match.group(1).strip()
            
            # Extract issues
            issues_match = re.search(
                r'\*\*ISSUES IDENTIFIED:\*\*\s*\n(.+?)(?=\n\*\*|$)', 
                response_text, 
                re.DOTALL
            )
            if issues_match:
                issues_text = issues_match.group(1)
                # Extract bullet points or numbered items
                issues = re.findall(r'[-•]\s*(.+?)(?=\n[-•]|\n\n|$)', issues_text, re.DOTALL)
                result["issues"] = [issue.strip() for issue in issues if issue.strip()]
            
            # Extract confidence
            confidence_match = re.search(
                r'\*\*CONFIDENCE LEVEL:\*\*\s*\n?(\d+)%?', 
                response_text
            )
            if confidence_match:
                result["confidence"] = int(confidence_match.group(1))
            else:
                # Default confidence based on diagnosis clarity
                result["confidence"] = 75
            
            # Extract urgency
            urgency_match = re.search(
                r'\*\*URGENCY:\*\*\s*\n?(\w+)', 
                response_text, 
                re.IGNORECASE
            )
            if urgency_match:
                urgency = urgency_match.group(1).lower()
                result["urgency"] = urgency if urgency in ["low", "medium", "high", "critical"] else "medium"
            
            # Extract suggestions
            suggestions_match = re.search(
                r'\*\*FOLLOW-UP SUGGESTIONS:\*\*\s*\n(.+?)(?=\n\*\*|$)', 
                response_text, 
                re.DOTALL
            )
            if suggestions_match:
                suggestions_text = suggestions_match.group(1)
                # Extract numbered or bullet points
                suggestions = re.findall(r'[\d•-]+\.\s*(.+?)(?=\n[\d•-]+\.|\n\n|$)', suggestions_text, re.DOTALL)
                result["suggestions"] = [sug.strip() for sug in suggestions if sug.strip()]
            
            # Extract detailed findings
            findings_match = re.search(
                r'\*\*DETAILED FINDINGS:\*\*\s*\n(.+?)(?=\n\*\*|$)', 
                response_text, 
                re.DOTALL
            )
            if findings_match:
                result["findings"]["detailed"] = findings_match.group(1).strip()
            
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            # Fallback: use raw response
            result["diagnosis"] = "Analysis completed - see detailed findings"
            result["findings"]["raw"] = response_text
        
        return result
    
    def analyze_multiple_images(
        self, 
        images: List[Tuple[bytes, str]], 
        clinical_context: Optional[str] = None
    ) -> List[Dict]:
        """
        Analyze multiple medical images
        
        Args:
            images: List of (image_data, image_type) tuples
            clinical_context: Clinical information
        
        Returns:
            List of analysis results
        """
        results = []
        
        for image_data, image_type in images:
            result = self.analyze_medical_image(
                image_data, 
                image_type, 
                clinical_context
            )
            results.append(result)
        
        return results
    
    def get_supported_image_types(self) -> List[Dict[str, str]]:
        """
        Get list of supported medical image types
        
        Returns:
            List of dictionaries with image type info
        """
        return [
            {
                "type": "chest_xray",
                "name": "Chest X-ray",
                "description": "Detects pneumonia, lung cancer, heart problems, fractures"
            },
            {
                "type": "ct_brain",
                "name": "Brain CT Scan",
                "description": "Detects stroke, bleeding, tumors, fractures"
            },
            {
                "type": "bone_xray",
                "name": "Bone X-ray",
                "description": "Detects fractures, arthritis, bone lesions"
            },
            {
                "type": "mri",
                "name": "MRI Scan",
                "description": "Detailed soft tissue imaging, tumors, disc herniation"
            },
            {
                "type": "ultrasound",
                "name": "Ultrasound",
                "description": "Organ imaging, pregnancy, fluid collections"
            },
            {
                "type": "auto",
                "name": "Auto-detect",
                "description": "Automatically detect image type and analyze"
            }
        ]


# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Initialize analyzer
    analyzer = MedicalImageAnalyzer(api_key=os.getenv('GEMINI_API_KEY'))
    
    # Test with a sample image
    print("Medical Image Analyzer initialized successfully!")
    print("\nSupported image types:")
    for img_type in analyzer.get_supported_image_types():
        print(f"- {img_type['name']}: {img_type['description']}")
