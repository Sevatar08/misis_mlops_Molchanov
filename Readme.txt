uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

curl -X GET "http://localhost:8000/health"
http://localhost:8000/health
pytest tests/ -v
docker build -t toxicity-classifier .

cat > test_data.json << 'EOF'
{
  "texts": [
    "Спасибо за помощь",
    "Ты глупый человек",
    "Отличная работа!"
  ]
}
EOF

curl -X POST "http://localhost:8000/predict_batch" -H "Content-Type: application/json" -d "@test_data.json"
