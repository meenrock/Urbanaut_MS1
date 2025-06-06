import bentoml
import numpy as np
from bentoml.models import HuggingFaceModel
from pandas import DataFrame
from transformers import pipeline
from typing import List, Any
from sklearn import datasets, metrics
import geopandas as gpd
import torchvision.transforms as transforms
import torch
from PIL import Image
from bentoml.io import Image as BentoImage

from src.data_processing.models.svm_land_use import SVMLandUseClassifier


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
    input_spec = BentoImage()
    output_spec = BentoImage()

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

    @bentoml.api
    def areasqmsvm(self,input_series: np.ndarray,retrain: bool) -> np.ndarray:
        # filePath = "D:/HighSpeedStorage/LVCHighSpd/MeenMookCoProject/POC/research/land-use-data/Landuse_bkk/กรุงเทพมหานคร2566/การใช้ที่ดิน/LU_BKK_2566.shp"
        filePath = "D:/HighSpeedStorage/LVCHighSpd/MeenMookCoProject/POC/research/land-use-data/Landuse_npt/นครปฐม2567/การใช้ที่ดิน/LU_NPT_2567.shp"
        data = gpd.read_file(filePath)
        area_sqm = data['Area_Sqm']
        shape_area = data['Shape_Area']
        target_if_more_than_10000 = (data['Area_Sqm'] <= 10000)
        if retrain:
            retrain_model = SVMLandUseClassifier()
            retrain_model.train()
        input_series = DataFrame({'area_sqm': area_sqm, 'shape_area': shape_area})
        area_sqm_pred_runner = bentoml.sklearn.load_model("area_sqm_svm_clf:latest")
        result = area_sqm_pred_runner.predict(input_series)
        print('my score of real pred ', metrics.accuracy_score(target_if_more_than_10000, result))
        # result = self.area_sqm_svm_clf.predict.run(input_series)
        # return {
        #     "prediction": return_result.tolist(),
        #     "model": "trend_transformer",
        #     "timestamp": datetime.datetime.now().isoformat()
        #     }
        return result

    # @bentoml.api(input_spec=input_spec, output_spec=output_spec)
    def map_marking_semantic_segment(self, input_image):
        with bentoml.importing():
            parking_lot_marker = bentoml.pytorch.load_model("parkinglot_unet:latest")
            model = parking_lot_marker.to('cuda')
            model.eval()

        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),  # Adjust size to match model's expected input
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0).to('cuda')
        with torch.no_grad():
            output = model(input_batch)

        if isinstance(output, torch.Tensor):
            output = output.squeeze(0)  # Remove batch dimension
            output = torch.argmax(output, dim=0)  # Get class with highest probability
            segmentation = output.byte().cpu().numpy()
        else:
            segmentation = output[0].argmax(dim=0).byte().cpu().numpy()

        result = Image.fromarray(segmentation)
        result = result.resize(input_image.size)
        return result