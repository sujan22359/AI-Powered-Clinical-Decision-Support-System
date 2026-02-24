"""
Blood Group Predictor

This module predicts blood group from fingerprint images using a trained ML model.
Supports Grad-CAM visualization for model interpretability.
"""

import logging
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import io
import base64


class BloodGroupPredictor:
    """
    Predicts blood group from fingerprint images using a trained ML model.
    
    Supports multiple model formats:
    - TensorFlow/Keras (.h5, .keras)
    - PyTorch (.pt, .pth) with Grad-CAM visualization
    - Scikit-learn (.pkl, .joblib)
    - ONNX (.onnx)
    """
    
    # Blood group classes (matching your model's order)
    BLOOD_GROUPS = ['A-', 'A+', 'B-', 'B+', 'AB-', 'AB+', 'O-', 'O+']
    
    def __init__(self, model_path: Optional[str] = None, model_type: str = "pytorch_cnn", enable_gradcam: bool = True):
        """
        Initialize the blood group predictor.
        
        Args:
            model_path: Path to the trained model file
            model_type: Type of model ("tensorflow", "pytorch", "pytorch_cnn", "sklearn", "onnx")
            enable_gradcam: Enable Grad-CAM visualization for PyTorch models
        """
        self.logger = logging.getLogger(__name__)
        self.model_type = model_type.lower()
        self.model = None
        self.model_loaded = False
        self.enable_gradcam = enable_gradcam
        self.gradcam_generator = None  # Cache Grad-CAM generator
        
        # Default model path
        if model_path is None:
            if self.model_type == "pytorch_cnn":
                model_path = Path(__file__).parent.parent / "models" / "ml_models" / "best_basic_cnn_with_residual.pth"
            else:
                model_path = Path(__file__).parent.parent / "models" / "ml_models" / "blood_group_model"
        
        self.model_path = Path(model_path)
        
        # Model configuration
        self.input_size = (224, 224)  # Default for CNN model
        self.preprocessing_required = True
        
        # Try to load the model
        try:
            self._load_model()
            # Warmup model for faster first prediction
            if self.model_loaded and self.model_type in ["pytorch", "pytorch_cnn"]:
                self._warmup_model()
        except Exception as e:
            self.logger.warning(f"Model not loaded during initialization: {e}")
            self.logger.info("Model will be loaded on first prediction")
    
    def _warmup_model(self):
        """Warmup model with a dummy prediction for faster first inference."""
        try:
            import torch
            dummy_input = torch.randn(1, 3, *self.input_size).to(self.device)
            with torch.no_grad():
                _ = self.model(dummy_input)
            self.logger.info("Model warmed up successfully")
        except Exception as e:
            self.logger.warning(f"Model warmup failed: {e}")
    
    def _load_model(self):
        """Load the trained model based on model type."""
        if self.model_loaded:
            return
        
        try:
            if self.model_type == "tensorflow":
                self._load_tensorflow_model()
            elif self.model_type == "pytorch" or self.model_type == "pytorch_cnn":
                self._load_pytorch_model()
            elif self.model_type == "sklearn":
                self._load_sklearn_model()
            elif self.model_type == "onnx":
                self._load_onnx_model()
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            self.model_loaded = True
            self.logger.info(f"Blood group prediction model loaded successfully ({self.model_type})")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_tensorflow_model(self):
        """Load TensorFlow/Keras model."""
        try:
            import tensorflow as tf
            
            # Try different file extensions
            possible_paths = [
                self.model_path.with_suffix('.h5'),
                self.model_path.with_suffix('.keras'),
                self.model_path,  # SavedModel format (directory)
            ]
            
            for path in possible_paths:
                if path.exists():
                    self.model = tf.keras.models.load_model(str(path))
                    self.logger.info(f"Loaded TensorFlow model from: {path}")
                    return
            
            raise FileNotFoundError(f"Model not found at: {self.model_path}")
            
        except ImportError:
            raise ImportError("TensorFlow not installed. Install with: pip install tensorflow")
    
    def _load_pytorch_model(self):
        """Load PyTorch model (including custom CNN architecture)."""
        try:
            import torch
            import torch.nn as nn
            
            # Define the ResidualBlock and BasicCNN architecture
            class ResidualBlock(nn.Module):
                def __init__(self, channels):
                    super(ResidualBlock, self).__init__()
                    self.conv_block = nn.Sequential(
                        nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                        nn.BatchNorm2d(channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                        nn.BatchNorm2d(channels)
                    )
                
                def forward(self, x):
                    identity = x
                    out = self.conv_block(x)
                    out += identity
                    return torch.relu(out)
            
            class BasicCNN(nn.Module):
                def __init__(self, num_classes):
                    super(BasicCNN, self).__init__()
                    self.features = nn.Sequential(
                        nn.Conv2d(3, 32, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(32, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, 3, padding=1),
                        nn.ReLU(),
                        ResidualBlock(128),
                        nn.MaxPool2d(2),
                        nn.Conv2d(128, 256, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2)
                    )
                    self.classifier = nn.Sequential(
                        nn.Linear(256 * 14 * 14, 512),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(512, num_classes)
                    )
                
                def forward(self, x):
                    x = self.features(x)
                    x = x.view(x.size(0), -1)
                    x = self.classifier(x)
                    return x
            
            # Try different file extensions
            possible_paths = [
                self.model_path.with_suffix('.pt'),
                self.model_path.with_suffix('.pth'),
                self.model_path,
            ]
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            for path in possible_paths:
                if path.exists():
                    if self.model_type == "pytorch_cnn":
                        # Load custom CNN architecture
                        self.model = BasicCNN(num_classes=len(self.BLOOD_GROUPS)).to(device)
                        self.model.load_state_dict(torch.load(str(path), map_location=device))
                    else:
                        # Load generic PyTorch model
                        self.model = torch.load(str(path), map_location=device)
                    
                    self.model.eval()  # Set to evaluation mode
                    self.device = device
                    self.logger.info(f"Loaded PyTorch model from: {path}")
                    return
            
            raise FileNotFoundError(f"Model not found at: {self.model_path}")
            
        except ImportError:
            raise ImportError("PyTorch not installed. Install with: pip install torch torchvision")
    
    def _load_sklearn_model(self):
        """Load Scikit-learn model."""
        try:
            import joblib
            
            # Try different file extensions
            possible_paths = [
                self.model_path.with_suffix('.pkl'),
                self.model_path.with_suffix('.joblib'),
            ]
            
            for path in possible_paths:
                if path.exists():
                    self.model = joblib.load(str(path))
                    self.logger.info(f"Loaded Scikit-learn model from: {path}")
                    return
            
            raise FileNotFoundError(f"Model not found at: {self.model_path}")
            
        except ImportError:
            raise ImportError("Joblib not installed. Install with: pip install joblib")
    
    def _load_onnx_model(self):
        """Load ONNX model."""
        try:
            import onnxruntime as ort
            
            path = self.model_path.with_suffix('.onnx')
            if path.exists():
                self.model = ort.InferenceSession(str(path))
                self.logger.info(f"Loaded ONNX model from: {path}")
                return
            
            raise FileNotFoundError(f"Model not found at: {path}")
            
        except ImportError:
            raise ImportError("ONNX Runtime not installed. Install with: pip install onnxruntime")
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess fingerprint image for model input.
        
        Args:
            image: PIL Image object
        
        Returns:
            Preprocessed numpy array or torch tensor
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize(self.input_size)
        
        if self.model_type in ["pytorch", "pytorch_cnn"]:
            # PyTorch preprocessing with ImageNet normalization
            import torch
            from torchvision import transforms
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
            return img_tensor.to(self.device)
        else:
            # Standard preprocessing for other frameworks
            img_array = np.array(image)
            img_array = img_array.astype(np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
    
    def predict(self, image_data: bytes) -> Dict[str, Any]:
        """
        Predict blood group from fingerprint image.
        
        Args:
            image_data: Image data as bytes
        
        Returns:
            Dictionary containing prediction results with optional Grad-CAM visualization
        """
        try:
            # Load model if not already loaded
            if not self.model_loaded:
                self._load_model()
            
            # Load and preprocess image
            image = Image.open(io.BytesIO(image_data))
            preprocessed = self.preprocess_image(image)
            
            # Store original image for Grad-CAM
            rgb_img = None
            gradcam_image = None
            
            if self.model_type in ["pytorch", "pytorch_cnn"] and self.enable_gradcam:
                # Keep RGB image for Grad-CAM visualization
                rgb_img = np.array(image.resize(self.input_size)) / 255.0
            
            # Make prediction based on model type
            if self.model_type in ["pytorch", "pytorch_cnn"]:
                predictions, gradcam_image = self._predict_pytorch(preprocessed, rgb_img)
            elif self.model_type == "tensorflow":
                predictions = self._predict_tensorflow(preprocessed)
            elif self.model_type == "sklearn":
                predictions = self._predict_sklearn(preprocessed)
            elif self.model_type == "onnx":
                predictions = self._predict_onnx(preprocessed)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            # Get predicted class and confidence
            predicted_class_idx = np.argmax(predictions)
            confidence = float(predictions[predicted_class_idx])
            predicted_blood_group = self.BLOOD_GROUPS[predicted_class_idx]
            
            # Create probability distribution
            probabilities = {
                blood_group: float(prob)
                for blood_group, prob in zip(self.BLOOD_GROUPS, predictions)
            }
            
            # Sort by probability
            sorted_probabilities = dict(
                sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            )
            
            result = {
                "success": True,
                "predicted_blood_group": predicted_blood_group,
                "confidence": confidence,
                "confidence_percent": f"{confidence * 100:.2f}%",
                "probabilities": sorted_probabilities,
                "top_3_predictions": list(sorted_probabilities.items())[:3],
                "model_type": self.model_type
            }
            
            # Add Grad-CAM visualization if available
            if gradcam_image is not None:
                result["gradcam_available"] = True
                result["gradcam_image"] = gradcam_image  # Base64 encoded image
            else:
                result["gradcam_available"] = False
            
            self.logger.info(
                f"Blood group prediction: {predicted_blood_group} "
                f"(confidence: {confidence * 100:.2f}%)"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Blood group prediction failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "predicted_blood_group": None,
                "confidence": 0.0,
                "gradcam_available": False
            }
    
    def _predict_tensorflow(self, preprocessed: np.ndarray) -> np.ndarray:
        """Make prediction using TensorFlow model."""
        predictions = self.model.predict(preprocessed, verbose=0)
        return predictions[0]  # Remove batch dimension
    
    def _predict_pytorch(self, preprocessed, rgb_img=None) -> Tuple[np.ndarray, Optional[str]]:
        """Make prediction using PyTorch model with optional Grad-CAM visualization."""
        import torch
        
        gradcam_base64 = None
        
        with torch.no_grad():
            predictions = self.model(preprocessed)
            predictions = torch.softmax(predictions, dim=1)
        
        predictions_np = predictions.cpu().numpy()[0]  # Remove batch dimension
        
        # Generate Grad-CAM if enabled and RGB image provided
        if self.enable_gradcam and rgb_img is not None and self.model_type == "pytorch_cnn":
            try:
                gradcam_base64 = self._generate_gradcam(preprocessed, rgb_img)
            except Exception as e:
                self.logger.warning(f"Grad-CAM generation failed: {e}")
        
        return predictions_np, gradcam_base64
    
    def _generate_gradcam(self, input_tensor, rgb_img: np.ndarray) -> str:
        """
        Generate Grad-CAM visualization for PyTorch CNN model.
        Uses cached Grad-CAM generator for better performance.
        
        Args:
            input_tensor: Preprocessed input tensor
            rgb_img: Original RGB image as numpy array
        
        Returns:
            Base64 encoded Grad-CAM heatmap image
        """
        try:
            self.logger.info("Starting Grad-CAM generation...")
            
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend for speed
            import matplotlib.pyplot as plt
            
            self.logger.info("Grad-CAM libraries imported successfully")
            
            # Create or reuse Grad-CAM generator (caching for performance)
            if self.gradcam_generator is None:
                self.logger.info("Creating new Grad-CAM generator...")
                target_layers = [self.model.features[-2]]
                self.gradcam_generator = GradCAM(model=self.model, target_layers=target_layers)
                self.logger.info("Grad-CAM generator created successfully")
            else:
                self.logger.info("Reusing cached Grad-CAM generator")
            
            # Generate Grad-CAM (fast operation)
            self.logger.info(f"Generating Grad-CAM heatmap... Input shape: {input_tensor.shape}")
            grayscale_cam = self.gradcam_generator(input_tensor=input_tensor, targets=None)[0]
            self.logger.info(f"Grad-CAM heatmap generated. Shape: {grayscale_cam.shape}")
            
            # Overlay Grad-CAM on original image
            self.logger.info(f"Overlaying Grad-CAM on image... RGB shape: {rgb_img.shape}")
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            self.logger.info("Grad-CAM overlay complete")
            
            # Convert to base64 (optimized)
            self.logger.info("Converting to base64...")
            fig, ax = plt.subplots(figsize=(6, 6), dpi=80)  # Reduced DPI for speed
            ax.imshow(cam_image)
            ax.axis('off')
            plt.tight_layout(pad=0)
            
            # Save to bytes buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0, dpi=80)
            plt.close(fig)
            buffer.seek(0)
            
            # Encode to base64
            gradcam_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            
            self.logger.info(f"Grad-CAM generation complete! Base64 length: {len(gradcam_base64)}")
            
            return gradcam_base64
            
        except ImportError as e:
            self.logger.error(f"Grad-CAM import error: {e}")
            self.logger.error("Install with: pip install grad-cam")
            return None
        except Exception as e:
            self.logger.error(f"Grad-CAM generation error: {e}", exc_info=True)
            return None
    
    def _predict_sklearn(self, preprocessed: np.ndarray) -> np.ndarray:
        """Make prediction using Scikit-learn model."""
        # Flatten the image
        flattened = preprocessed.reshape(1, -1)
        
        # Get probabilities
        if hasattr(self.model, 'predict_proba'):
            predictions = self.model.predict_proba(flattened)
            return predictions[0]
        else:
            # If no predict_proba, use predict and create one-hot
            prediction = self.model.predict(flattened)
            one_hot = np.zeros(len(self.BLOOD_GROUPS))
            one_hot[prediction[0]] = 1.0
            return one_hot
    
    def _predict_onnx(self, preprocessed: np.ndarray) -> np.ndarray:
        """Make prediction using ONNX model."""
        # Get input name
        input_name = self.model.get_inputs()[0].name
        
        # Run inference
        predictions = self.model.run(None, {input_name: preprocessed})
        
        return predictions[0][0]  # Remove batch dimension
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_loaded": self.model_loaded,
            "model_type": self.model_type,
            "model_path": str(self.model_path),
            "input_size": self.input_size,
            "supported_blood_groups": self.BLOOD_GROUPS,
            "num_classes": len(self.BLOOD_GROUPS)
        }
