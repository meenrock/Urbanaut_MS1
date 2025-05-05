from __future__ import annotations

from typing import Any

import bentoml

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



@bentoml.service(
    image=my_image,
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
class Summarization:
    # Define the Hugging Face model as a class variable
    # test = shp_input.ShapeFileInputSpark()
    # test.read_shape_file("C:/GIS/input/N_COASTAL_CHANGE_20231203_192816.shp")
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
