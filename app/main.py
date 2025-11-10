import logging

from fastapi import FastAPI, HTTPException

from app.model import classifier
from app.schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    TextRequest,
)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Russian Toxicity Classifier API",
    description="API для классификации токсичности русских текстов",
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event() -> None:
    """Загрузка модели при старте приложения"""
    try:
        classifier.load_model()
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise e


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Проверка работоспособности сервиса"""
    return HealthResponse(status="OK")


@app.post("/predict", response_model=PredictionResponse)
async def predict_single_text(
    request: TextRequest,
) -> PredictionResponse:
    """Предсказание класса для одного текста"""
    try:
        prediction, probability = classifier.predict(request.text)
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
        )
    except Exception as e:
        logger.error(f"Error in single prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch", response_model=BatchResponse)
async def predict_batch_texts(request: BatchRequest) -> BatchResponse:
    """Предсказание классов для списка текстов"""
    try:
        predictions = classifier.predict_batch(request.texts)
        return BatchResponse(predictions=predictions)
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model_info", response_model=ModelInfoResponse)
async def get_model_info() -> ModelInfoResponse:
    """Получение информации о модели"""
    try:
        model_info = classifier.get_model_info()
        return ModelInfoResponse(**model_info)
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
