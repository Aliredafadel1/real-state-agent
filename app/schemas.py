from typing import List, Optional

from pydantic import BaseModel, Field


class ExtractedFeatures(BaseModel):
    # Raw user/LLM extracted features
    overall_qual: Optional[int] = Field(default=None, description="Overall quality rating")
    gr_liv_area: Optional[float] = Field(default=None, description="Above ground living area in square feet")
    garage_cars: Optional[float] = Field(default=None, description="Number of garage car spaces")
    total_bsmt_sf: Optional[float] = Field(default=None, description="Total basement square footage")
    full_bath: Optional[int] = Field(default=None, description="Number of full bathrooms")
    year_built: Optional[int] = Field(default=None, description="Year the house was built")
    neighborhood: Optional[str] = Field(default=None, description="Neighborhood location")
    house_style: Optional[str] = Field(default=None, description="House style")
    garage_type: Optional[str] = Field(default=None, description="Type of garage")
    exter_qual: Optional[str] = Field(default=None, description="Exterior quality")

    provided_fields: List[str] = Field(default_factory=list, description="Fields extracted from the user query")
    missing_fields: List[str] = Field(default_factory=list, description="Fields still missing after extraction")


class ModelInput(BaseModel):
    # Final features expected by the ML model
    OverallQual: int
    GrLivArea: float
    GarageCars: float
    TotalBsmtSF: float
    FullBath: int
    YearBuilt: int
    TotalSF: float
    HouseAge: int
    Neighborhood: str
    HouseStyle: str
    GarageType: str
    ExterQual: str


class PredictionOutput(BaseModel):
    """Schema for single prediction output."""
    prediction: float = Field(..., description="Predicted house price")
    confidence: Optional[float] = Field(None, description="Confidence score (0-1)")


class BatchPredictionOutput(BaseModel):
    """Schema for batch prediction output."""
    predictions: List[float] = Field(..., description="List of predictions")
    count: int = Field(..., description="Number of predictions")


class PredictionResponse(BaseModel):
    extracted_features: ExtractedFeatures
    model_input: ModelInput
    predicted_price: float
    explanation: str
    
    model_config = {"protected_namespaces": ()}


if __name__ == "__main__":
    # Quick test
    print("Testing ExtractedFeatures...")
    ef = ExtractedFeatures(overall_qual=8, provided_fields=["overall_qual"])
    print(f"✓ {ef}")
    
    print("\nTesting ModelInput...")
    mi = ModelInput(
        OverallQual=8, GrLivArea=2500.0, GarageCars=2.0,
        TotalBsmtSF=1000.0, FullBath=2, YearBuilt=2005,
        TotalSF=3500.0, HouseAge=19, Neighborhood="Downtown",
        HouseStyle="Ranch", GarageType="Attached", ExterQual="Good"
    )
    print(f"✓ {mi}")
    print("\nAll tests passed!")