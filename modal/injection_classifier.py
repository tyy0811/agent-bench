"""Deploy DeBERTa-v3-base injection classifier on Modal.

Usage:
    modal deploy modal/injection_classifier.py
    modal serve modal/injection_classifier.py  # Dev mode

Endpoint: POST /classify {"text": "..."}
Returns:  {"label": "INJECTION" | "SAFE", "score": 0.95}
"""

import modal

MODELS_DIR = "/models"

classifier_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers>=4.40.0",
        "torch>=2.0.0",
        "sentencepiece",
        "protobuf",
    )
)

app = modal.App("agent-bench-injection-classifier")
model_volume = modal.Volume.from_name("injection-model-cache", create_if_missing=True)


@app.cls(
    image=classifier_image,
    gpu="T4",
    scaledown_window=300,
    timeout=120,
    volumes={MODELS_DIR: model_volume},
)
class InjectionClassifier:
    @modal.enter()
    def load(self):
        from transformers import pipeline

        self.pipe = pipeline(
            "text-classification",
            model="deepset/deberta-v3-base-injection",
            device="cuda",
            model_kwargs={"cache_dir": MODELS_DIR},
        )

    @modal.method()
    def classify(self, text: str) -> dict:
        result = self.pipe(text, truncation=True, max_length=512)[0]
        return {"label": result["label"], "score": result["score"]}


@app.function(image=classifier_image, gpu="T4", volumes={MODELS_DIR: model_volume})
@modal.web_endpoint(method="POST")
def classify_endpoint(item: dict) -> dict:
    """HTTP endpoint wrapper for the classifier."""
    classifier = InjectionClassifier()
    return classifier.classify.remote(item["text"])
