"""Pydantic Schemas for FastAPI Request Validation and Response Serialization."""

from typing import List
from pydantic import BaseModel, ConfigDict, Field


class PatientHeartRecord(BaseModel):
    """Input clinical features for a single patient heart disease risk assessment."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 58,
                "sex": 1,
                "cp": 4,
                "trestbps": 140,
                "chol": 260,
                "fbs": 0,
                "restecg": 2,
                "thalach": 135,
                "exang": 1,
                "oldpeak": 2.2,
                "slope": 2,
                "ca": 2,
                "thal": 7,
            }
        }
    )

    age: float = Field(..., ge=18, le=115, description="Patient age in years")
    sex: int = Field(..., ge=0, le=1, description="Biological sex (1 = Male, 0 = Female)")
    cp: int = Field(
        ...,
        ge=1,
        le=4,
        description="Chest pain type (1=typical angina, 2=atypical, 3=non-anginal, 4=asymptomatic)",
    )
    trestbps: float = Field(
        ..., ge=60, le=260, description="Resting blood pressure in mm Hg on admission"
    )
    chol: float = Field(..., ge=80, le=650, description="Serum cholesterol in mg/dL")
    fbs: int = Field(
        ..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dL (1 = True, 0 = False)"
    )
    restecg: int = Field(
        ...,
        ge=0,
        le=2,
        description="Resting ECG results (0=normal, 1=ST-T abnormality, 2=LV hypertrophy)",
    )
    thalach: float = Field(..., ge=50, le=250, description="Maximum heart rate achieved (bpm)")
    exang: int = Field(
        ..., ge=0, le=1, description="Exercise-induced angina (1 = Yes, 0 = No)"
    )
    oldpeak: float = Field(
        ..., ge=0.0, le=10.0, description="ST depression induced by exercise relative to rest"
    )
    slope: int = Field(
        ..., ge=1, le=3, description="Slope of peak exercise ST segment (1=up, 2=flat, 3=down)"
    )
    ca: int = Field(
        ..., ge=0, le=3, description="Number of major vessels (0-3) colored by fluoroscopy"
    )
    thal: int = Field(
        ...,
        ge=0,
        le=7,
        description="Thalassemia status (3 = normal, 6 = fixed defect, 7 = reversible defect)",
    )


class BatchPatientRequest(BaseModel):
    """Batch prediction request payload."""

    patients: List[PatientHeartRecord] = Field(
        ..., min_length=1, max_length=100, description="List of patient clinical records"
    )


class PredictionResponse(BaseModel):
    """Structured prediction response returned by `/predict`."""

    prediction: int = Field(..., description="Binary class prediction (0=No Disease, 1=Heart Disease)")
    prediction_label: str = Field(..., description="Human-readable diagnosis label")
    risk_probability: float = Field(..., description="Estimated probability of heart disease [0, 1]")
    confidence: float = Field(..., description="Model confidence in the predicted class [0, 1]")
    risk_tier: str = Field(..., description="Clinical risk stratification (LOW, MODERATE, HIGH)")
    model_name: str = Field(..., description="Active classifier architecture")
    model_version: str = Field(..., description="Packaged model version")
    request_id: str = Field(..., description="Unique request trace identifier")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")


class HealthResponse(BaseModel):
    """Service health and readiness status."""

    status: str
    model_loaded: bool
    model_name: str
    model_version: str
    uptime_seconds: float
    total_predictions_served: int
