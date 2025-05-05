import bentoml
import joblib
import numpy as np
from bentoml.models import HuggingFaceModel
from transformers import pipeline
from typing import List, Any
from bentoml.models import BentoModel
from sklearn import datasets

# Run two models in the same Service on the same hardware device
@bentoml.service(
    resources={"cpu": 16,"gpu": 1, "memory": "4GiB"},
    traffic={"timeout": 20},
    http={"cors": {
            "enabled": True,
            "access_control_allow_origins": ["http://localhost:4200", "https://myorg.com:8080"],
            "access_control_allow_methods": ["GET", "OPTIONS", "POST", "HEAD", "PUT"],
            "access_control_allow_credentials": True,
            "access_control_allow_headers": ["*"],
            "access_control_allow_origin_regex": "https://.*\.my_org\.com",
            "access_control_max_age": 1200,
            "access_control_expose_headers": ["Content-Length"]
    }}
)
class MultiModelService:
    model_a_path = HuggingFaceModel("FacebookAI/roberta-large-mnli")
    model_b_path = HuggingFaceModel("distilbert/distilbert-base-uncased")
    model_c_path = bentoml.models.HuggingFaceModel("sshleifer/distilbart-cnn-12-6")

    def __init__(self) -> None:
        self.pipeline_a = pipeline(task="zero-shot-classification", model=self.model_a_path, hypothesis_template="This text is about {}")
        self.pipeline_b = pipeline(task="sentiment-analysis", model=self.model_b_path)
        self.pipeline_c = pipeline('summarization', model=self.model_c_path)

    @bentoml.api
    def process_a(self, input_data: str, labels: List[str] = ["positive", "negative", "neutral"]) -> dict:
        return self.pipeline_a(input_data, labels)

    @bentoml.api
    def process_b(self, input_data: str) -> dict:
        return self.pipeline_b(input_data)[0]

    @bentoml.api
    def test_iris(self) -> np.ndarray:
        iris_clf_runner = bentoml.sklearn.load_model("iris_clf:latest")
        iris = datasets.load_iris()
        return iris_clf_runner.predict(iris.data)

    @bentoml.api
    def summarize(self, text: str) -> dict[Any, str]:
        if text == "":
            return {
                "result": "The input text is empty",
            }
        result = self.pipeline_c(text)
        resultExport = {
            "result": f"Hello world! Here's your summary: {result[0]['summary_text']}"
        }
        return resultExport