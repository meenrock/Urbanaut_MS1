from __future__ import annotations

import datetime
from typing import Any, Generator

import bentoml
import numpy as np
import torch
from bentoml import Service, api
from bentoml.io import JSON, NumpyNdarray
from fastapi.middleware.cors import CORSMiddleware
from transformers.pipelines.pt_utils import PipelineIterator
from bentoml.io import IODescriptor

with bentoml.importing():
    from transformers import pipeline
    import sedona


EXAMPLE_INPUT = "Breaking News: In an astonishing turn of events, the small \
town of Willow Creek has been taken by storm as local resident Jerry Thompson's cat, \
Whiskers, performed what witnesses are calling a 'miraculous and gravity-defying leap.' \
Eyewitnesses report that Whiskers, an otherwise unremarkable tabby cat, jumped \
a record-breaking 20 feet into the air to catch a fly. The event, which took \
place in Thompson's backyard, is now being investigated by scientists for potential \
breaches in the laws of physics. Local authorities are considering a town festival \
to celebrate what is being hailed as 'The Leap of the Century."


my_image = bentoml.images.PythonImage(python_version="3.11") \
        .python_packages("torch", "transformers")

trend_runner = bentoml.sklearn.get("trend_transformer").to_runner()

svc = bentoml.Service("trend_transformer", runners=[trend_runner])

@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 30},
    http={
        "cors": {
            "enabled": True,
            "access_control_allow_origins": ["http://localhost:4200", "https://myorg.com:8080"],
            "access_control_allow_methods": ["GET", "OPTIONS", "POST", "HEAD", "PUT"],
            "access_control_allow_credentials": True,
            "access_control_allow_headers": ["*"],
            "access_control_allow_origin_regex": "https://.*\.my_org\.com",
            "access_control_max_age": 1200,
            "access_control_expose_headers": ["Content-Length"]
        }
    }
)

class MultiModelRunner:
    def __init__(self):
        self.summarizer = bentoml.pytorch.load_model("summarization_model")
        self.trend_predictor = bentoml.pytorch.load_model("trend_transformer")
        self.summary_pipeline = pipeline(
            'summarization',
            model=self.summarizer
        )

    @svc.api(input=JSON(), output=JSON())
    async def summarize(self, text: str) -> dict:
        result = self.summary_pipeline(text)
        return {
            "summary": result[0]['summary_text'],
            "model": "distilbart-cnn-12-6",
            "timestamp": datetime.datetime.now().isoformat()
        }

    @svc.api(input=NumpyNdarray(), output=JSON())
    async def predict_trend(self, input_array: np.ndarray) -> dict:
        input_tensor = torch.from_numpy(input_array)
        with torch.no_grad():
            prediction = await self.trend_predictor.to_runner().async_run(input_tensor)

        return {
            "prediction": prediction.tolist(),
            "model": "trend_transformer",
            "timestamp": datetime.datetime.now().isoformat()
        }

class TrendPredictorRunner:
    def __init__(self):
        self.model = bentoml.pytorch.get("trend_transformer").to_runner()

    @bentoml.api
    async def predict(self, input_tensor: Any) -> Generator[Any, None, None]:
        with torch.no_grad():
            return await self.model.predict.async_run(input_tensor)

class Summarization:
    model_path = bentoml.models.HuggingFaceModel("sshleifer/distilbart-cnn-12-6")

    def __init__(self) -> None:
        # Load model into pipeline
        self.pipeline = pipeline('summarization', model=self.model_path)
    
    @bentoml.api
    def summarize(self, text: str = EXAMPLE_INPUT) -> dict[Any, str]:
        result = self.pipeline(text)
        resultExport = {
            "result": f"Hello world! Here's your summary: {result[0]['summary_text']}"
        }
        return resultExport

    @bentoml.api
    async def predict(input_array: np.ndarray) -> dict:
        svc = Service(name="multi_trend_predictor")
        input_tensor = torch.from_numpy(input_array).unsqueeze(1)
        runner = bentoml.pytorch.get_runner("trend_transformer")
        with torch.no_grad():
            prediction = await runner.predict.async_run(input_tensor)

        return {
            "prediction": prediction.tolist(),
            "model": "trend_transformer",
            "timestamp": datetime.datetime.now().isoformat()
        }
