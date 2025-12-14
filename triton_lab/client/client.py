import numpy as np
import tritonclient.http as httpclient

# Тестовые тексты
texts = [
    "Это прекрасный день для прогулки в парке.",
    "Ты полный идиот и ничего не понимаешь!",
    "Научные исследования показывают интересные результаты.",
    "Короче конченный, конченный фильм.",
    "Ненавижу докер!",
    "Ненавижу этот поганый докер!",
    "Будь проклят тот день, когда я сел за руль этого пылесоса!",
    "Как же я ненавижу эту отвратительную погоду!",
]

client = httpclient.InferenceServerClient(url="localhost:8000")

input_data = np.array([[text] for text in texts], dtype=object)

inputs = [httpclient.InferInput("TEXT", input_data.shape, "BYTES")]
inputs[0].set_data_from_numpy(input_data)

outputs = [httpclient.InferRequestedOutput("LOGITS")]

response = client.infer("toxicity_classifier", inputs, outputs=outputs)

logits = response.as_numpy("LOGITS")

print("Результаты классификации:")
for i, text in enumerate(texts):
    toxicity = logits[i][1]
    print(f"\nТекст: {text[:50]}...")
    print(f"  Логиты: [{logits[i][0]:.2f}, {logits[i][1]:.2f}]")
    print(f"  Вердикт: {'ТОКСИЧНЫЙ' if toxicity > 0 else 'НЕТОКСИЧНЫЙ'}")
