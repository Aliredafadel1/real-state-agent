"""
Model Inference Module - Load trained model and make predictions.

This module handles loading the trained ML model and generating predictions
on new data. It supports both single and batch predictions.
"""

import json
import sys
from pathlib import Path
from typing import List, Union, Dict, Any, Optional

import numpy as np
import pandas as pd
import joblib

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schemas import (
    ModelInput,
    PredictionOutput,
    BatchPredictionOutput,
    PredictionResponse,
)
from app.llm_stage2 import generate_explanation


class ModelInference:
    """Handle model loading and inference."""
    
    MODEL_PATH = Path(__file__).parent.parent / "artifacts" / "best_model.joblib"
    
    def __init__(self, model_path: Union[str, Path] = None):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the trained model. Defaults to artifacts/best_model.pkl
        """
        if model_path is None:
            model_path = self.MODEL_PATH
        
        self.model_path = Path(model_path)
        self.pipeline = None
        self.load_model()
    
    def load_model(self):
        """
        Load the trained model from disk.
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                f"Train a model first using src.models.train"
            )
        
        try:
            self.pipeline = joblib.load(self.model_path)
            print(f"[OK] Model loaded from {self.model_path}")
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")
    
    def predict_single(self, features: np.ndarray) -> float:
        """
        Make a single prediction.
        
        Args:
            features: Input feature vector (1D array)
            
        Returns:
            Predicted value
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Reshape to 2D array for sklearn (1 sample, n features)
        features_2d = features.reshape(1, -1)
        prediction = self.pipeline.predict(features_2d)[0]
        
        return float(prediction)
    
    def predict_batch(self, features: Union[np.ndarray, pd.DataFrame, List[List[float]]], feature_names: List[str] = None) -> np.ndarray:
        """
        Make batch predictions.
        
        Args:
            features: Input features - can be numpy array, DataFrame, or list of lists
            feature_names: Optional list of column names (required if features is ndarray or list)
            
        Returns:
            Predictions array
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Convert to DataFrame if needed (sklearn pipeline expects DataFrame with column names)
        if isinstance(features, np.ndarray):
            default_features = [
                "OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF",
                "FullBath", "YearBuilt", "TotalSF", "HouseAge",
                "Neighborhood", "HouseStyle", "GarageType", "ExterQual"
            ]
            cols = feature_names if feature_names else default_features[:features.shape[1]]
            features = pd.DataFrame(features, columns=cols)
        elif isinstance(features, list):
            default_features = [
                "OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF",
                "FullBath", "YearBuilt", "TotalSF", "HouseAge",
                "Neighborhood", "HouseStyle", "GarageType", "ExterQual"
            ]
            cols = feature_names if feature_names else default_features[:len(features[0])]
            features = pd.DataFrame(features, columns=cols)
        
        predictions = self.pipeline.predict(features)
        return predictions
    
    def predict_from_dict(self, data: Dict[str, Any]) -> float:
        """
        Make prediction from dictionary of features.
        
        Args:
            data: Dictionary with feature names and values
            
        Returns:
            Predicted value
        """
        # Convert dict to DataFrame (required by sklearn pipeline)
        df = pd.DataFrame([data])
        prediction = self.pipeline.predict(df)[0]
        return float(prediction)
    
    def predict_from_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions from a pandas DataFrame.
        
        Args:
            df: DataFrame with feature columns
            
        Returns:
            Predictions array
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        predictions = self.pipeline.predict(df)
        return predictions


def create_inference_engine(model_path: Union[str, Path] = None) -> ModelInference:
    """
    Factory function to create and initialize an inference engine.
    
    Args:
        model_path: Optional path to model file
        
    Returns:
        Initialized ModelInference instance
    """
    return ModelInference(model_path)


def predict(model_input: ModelInput) -> PredictionOutput:
    """
    High-level prediction function that works with schema objects.
    
    Args:
        model_input: ModelInput schema with all required features
        
    Returns:
        PredictionOutput with prediction and confidence
    """
    try:
        # Initialize inference engine
        engine = create_inference_engine()
        
        # Convert schema to dict for prediction
        input_dict = model_input.model_dump()
        
        # Make prediction
        prediction = engine.predict_from_dict(input_dict)
        
        # Calculate confidence (example: using model probability if available)
        # For regression, we use prediction normalized to 0-1 range
        # This is a simple heuristic; adjust based on your model
        confidence = min(1.0, max(0.0, 0.8))  # Placeholder confidence
        
        return PredictionOutput(
            prediction=prediction,
            confidence=confidence
        )
        
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")


def predict_with_explanation(
    model_input: ModelInput,
    extracted_features: Optional[Dict[str, Any]] = None,
    missing_fields: Optional[List[str]] = None,
    dataset_summary: Optional[Dict[str, Any]] = None,
    llm_provider: Optional[str] = None,
) -> PredictionResponse:
    """
    High-level prediction function with Stage 2 explanation generation.

    Args:
        model_input: ModelInput schema with required model features.
        extracted_features: Optional Stage 1 output as a plain dict.
        missing_fields: Optional list of missing Stage 1 fields.
        dataset_summary: Optional dataset-level context for explanation.
        llm_provider: Optional provider override for LLM explanation.

    Returns:
        PredictionResponse containing prediction and explanation text.
    """
    # Reuse existing inference logic to keep behavior consistent.
    prediction_result = predict(model_input)
    input_dict = model_input.model_dump()

    explanation = generate_explanation(
        extracted_features=extracted_features or {},
        model_input=input_dict,
        predicted_price=prediction_result.prediction,
        missing_fields=missing_fields or [],
        dataset_summary=dataset_summary,
        provider=llm_provider,
    )

    return PredictionResponse(
        extracted_features=extracted_features or {},
        model_input=model_input,
        predicted_price=prediction_result.prediction,
        explanation=explanation,
    )


def batch_predict(features_list: List[List[float]]) -> BatchPredictionOutput:
    """
    Make batch predictions.
    
    Args:
        features_list: List of feature vectors
        
    Returns:
        BatchPredictionOutput with all predictions
    """
    try:
        # Initialize inference engine
        engine = create_inference_engine()
        
        # Convert to numpy array
        features_array = np.array(features_list)
        
        # Make predictions
        predictions = engine.predict_batch(features_array)
        
        return BatchPredictionOutput(
            predictions=predictions.tolist(),
            count=len(predictions)
        )
        
    except Exception as e:
        raise Exception(f"Batch prediction failed: {str(e)}")


if __name__ == "__main__":
    # Test inference
    print("Testing Model Inference Module...")
    print("=" * 60)
    
    try:
        # Initialize engine
        engine = create_inference_engine()
        
        # Test with sample data
        sample_features = {
            "OverallQual": 8,
            "GrLivArea": 2500.0,
            "GarageCars": 2.0,
            "TotalBsmtSF": 1000.0,
            "FullBath": 2,
            "YearBuilt": 2005,
            "TotalSF": 3500.0,
            "HouseAge": 19,
            "Neighborhood": "Downtown",
            "HouseStyle": "Ranch",
            "GarageType": "Attached",
            "ExterQual": "Good"
        }
        
        # Single prediction
        prediction = engine.predict_from_dict(sample_features)
        print(f"\n[OK] Single Prediction: ${prediction:,.2f}")
        
        # Batch prediction - with all required columns (converted to dict then back to list as proper format)
        batch_dicts = [
            {
                "OverallQual": 8, "GrLivArea": 2500.0, "GarageCars": 2.0,
                "TotalBsmtSF": 1000.0, "FullBath": 2, "YearBuilt": 2005,
                "TotalSF": 3500.0, "HouseAge": 19,
                "Neighborhood": "Downtown", "HouseStyle": "Ranch",
                "GarageType": "Attached", "ExterQual": "Good"
            },
            {
                "OverallQual": 7, "GrLivArea": 2200.0, "GarageCars": 1.0,
                "TotalBsmtSF": 900.0, "FullBath": 1, "YearBuilt": 2010,
                "TotalSF": 3100.0, "HouseAge": 14,
                "Neighborhood": "Midtown", "HouseStyle": "Colonial",
                "GarageType": "Detached", "ExterQual": "Average"
            }
        ]
        
        batch_df = pd.DataFrame(batch_dicts)
        batch_predictions = engine.predict_batch(batch_df)
        print(f"\n[OK] Batch Predictions ({len(batch_predictions)} samples):")
        for i, pred in enumerate(batch_predictions):
            print(f"  Sample {i+1}: ${pred:,.2f}")
        
        print("\n" + "=" * 60)
        print("[OK] All inference tests passed!")
        
    except FileNotFoundError as e:
        print(f"\n[WARN] Warning: {e}")
        print("Train a model first using: python -m src.models.train")
    except Exception as e:
        print(f"\n[ERR] Error: {e}")
