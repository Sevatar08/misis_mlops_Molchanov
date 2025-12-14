import json
from typing import Any, List

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class TritonPythonModel:
    def initialize(self, args: dict) -> None:
        """Инициализация модели при запуске"""
        self.logger = pb_utils.Logger
        self.model_config = json.loads(args["model_config"])

        model_name = "s-nlp/russian_toxicity_classifier"

        self.logger.log_info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        self.model.eval()
        self.logger.log_info("Model loaded successfully")

    def execute(self, requests: List[Any]) -> List[Any]:
        """Обработка запросов"""
        responses = []

        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
            texts = input_tensor.as_numpy()

            decoded_texts = []
            for text_array in texts:
                for text in text_array:
                    decoded_texts.append(text.decode("utf-8"))

            inputs = self.tokenizer(
                decoded_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            probabilities = torch.nn.functional.softmax(logits, dim=-1)

            predictions = torch.argmax(logits, dim=-1)

            toxicity_scores = probabilities[:, 1]

            logits_tensor = pb_utils.Tensor("LOGITS",
                                             logits.numpy().astype(np.float32))

            probabilities_tensor = pb_utils.Tensor(
                "PROBABILITIES", probabilities.numpy().astype(np.float32)
            )

            prediction_tensor = pb_utils.Tensor(
                "PREDICTION", predictions.numpy().astype(np.int32)
            )

            toxicity_tensor = pb_utils.Tensor(
                "TOXICITY_SCORE",
                toxicity_scores.numpy().astype(np.float32).reshape(-1, 1),
            )

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    logits_tensor,
                    probabilities_tensor,
                    prediction_tensor,
                    toxicity_tensor,
                ]
            )
            responses.append(inference_response)

        return responses

    def finalize(self) -> None:
        """Очистка ресурсов при завершении"""
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
