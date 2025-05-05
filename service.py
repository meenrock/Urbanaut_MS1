import datetime

import bentoml
import numpy as np
from bentoml._internal.io_descriptors import NumpyNdarray
from grpc import services
from pandas import DataFrame
import geopandas as gpd

from sklearn import svm
from sklearn import datasets

from src.data_processing.models.svm_land_use import SVMLandUseClassifier

# Load training data set
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Train the model
clf = svm.SVC(gamma='scale')
clf.fit(X, y)

my_model = SVMLandUseClassifier()
my_model.train()

# Save model to the BentoML local Model Store
saved_model = bentoml.sklearn.save_model("iris_clf", clf)

iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()
area_sqm_pred_runner = bentoml.sklearn.get("area_sqm_svm_clf:latest").to_runner()

svc = bentoml.Service("iris_classifier",
                      runners=[iris_clf_runner, area_sqm_pred_runner],
                      )


@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
    result = iris_clf_runner.predict.run(iris.data)
    return result

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def areasqmsvm(input_series: np.ndarray) -> np.ndarray:
    # input_series: np.ndarray
    filePath = "D:/HighSpeedStorage/LVCHighSpd/MeenMookCoProject/POC/research/land-use-data/Landuse_bkk/กรุงเทพมหานคร2566/การใช้ที่ดิน/LU_BKK_2566.shp"
    data = gpd.read_file(filePath)
    area_sqm = data['Area_Sqm']
    input_series = DataFrame({'area_sqm': area_sqm})
    result = area_sqm_pred_runner.predict.run(input_series)
    # result = self.area_sqm_svm_clf.predict.run(input_series)
    # return {
    #     "prediction": return_result.tolist(),
    #     "model": "trend_transformer",
    #     "timestamp": datetime.datetime.now().isoformat()
    #     }
    return result