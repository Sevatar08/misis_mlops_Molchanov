import logging
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)


class ToxicityClassifier:
    def __init__(self) -> None:
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.classifier: Optional[Any] = None
        self.model_name: str = "s-nlp/russian_toxicity_classifier"
        self.is_loaded: bool = False

    def load_model(self) -> None:
        """Загрузка модели"""
        try:
            logger.info("Loading toxicity classification model...")
            self.classifier = pipeline(
                "text-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                max_length=512,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self.is_loaded = True
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise e

    def predict(self, text: str) -> Tuple[bool, float]:
        """Предсказание для одного текста"""
        if not self.is_loaded:
            self.load_model()

        try:
            # Добавляем проверку для mypy
            if self.classifier is None:
                raise RuntimeError("Classifier not initialized")

            result = self.classifier(text)[0]
            prediction = result["label"] == "toxic"
            probability = result["score"]
            return prediction, probability
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise e

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Предсказание для списка текстов"""
        if not self.is_loaded:
            self.load_model()

        try:
            # Добавляем проверку для mypy
            if self.classifier is None:
                raise RuntimeError("Classifier not initialized")

            results = self.classifier(texts)
            predictions = []
            for text, result in zip(texts, results):
                prediction = result["label"] == "toxic"
                probability = result["score"]
                predictions.append(
                    {
                        "text": text,
                        "prediction": prediction,
                        "probability": probability,
                    }
                )
            return predictions
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise e

    def get_model_info(self) -> Dict[str, Any]:
        """Получение информации о модели"""
        return {
            "model_name": self.model_name,
            "model_type": "BERT for sequence classification",
            "max_length": 512,
            "version": "1.0",
        }


# Глобальный экземпляр классификатора
classifier = ToxicityClassifier()
