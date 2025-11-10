from typing import List, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class TextRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    prediction: bool
    probability: float


class BatchRequest(BaseModel):
    texts: List[str]


class BatchPrediction(BaseModel):
    text: str
    prediction: bool
    probability: float


class BatchResponse(BaseModel):
    predictions: List[BatchPrediction]


class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    max_length: int
    version: Optional[str] = None
