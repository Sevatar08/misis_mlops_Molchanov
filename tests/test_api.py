from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    """Тест проверки работоспособности сервиса"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}


def test_predict_single_text():
    """Тест предсказания для одного текста"""
    test_text = "Это нормальный текст без оскорблений"
    response = client.post("/predict", json={"text": test_text})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert isinstance(data["prediction"], bool)
    assert 0 <= data["probability"] <= 1


def test_predict_batch():
    """Тест предсказания для списка текстов"""
    texts = ["Это хороший текст", "Ты идиот и тупица", "Спасибо за помощь"]
    response = client.post("/predict_batch", json={"texts": texts})
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == len(texts)
    for pred in data["predictions"]:
        assert "text" in pred
        assert "prediction" in pred
        assert "probability" in pred


def test_model_info():
    """Тест получения информации о модели"""
    response = client.get("/model_info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "model_type" in data


def test_empty_batch():
    """Тест пустого списка текстов"""
    response = client.post("/predict_batch", json={"texts": []})
    assert response.status_code == 200
    data = response.json()
    assert data["predictions"] == []
