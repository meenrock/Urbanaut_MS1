import bentoml
import geopandas as gpd
from pandas import DataFrame

from sklearn import svm

from sklearn import datasets

class SVMLandUseClassifier:
    def __init__(self):
        filePath = "D:/HighSpeedStorage/LVCHighSpd/MeenMookCoProject/POC/research/land-use-data/Landuse_bkk/กรุงเทพมหานคร2566/การใช้ที่ดิน/LU_BKK_2566.shp"
        data = gpd.read_file(filePath)
        self.data = data
        # print(data.shape,'shape of data')
        # print(data.columns)
        # print(data['Area_Sqm'])
        # target_if_more_than_10000 = (self.data['Area_Sqm'] <= 10000)
        # print(target_if_more_than_10000)

        # iris = datasets.load_iris()
        # X, y = iris.data, iris.target
        #
        # clf = svm.SVC(gamma='scale')
        # clf.fit(X, y)
        #
        # saved_model = bentoml.sklearn.save_model("iris_clf", clf)

    def train(self):
        geometry = self.data['geometry']
        lu_code = self.data['LU_CODE']
        shape_area = self.data['Shape_Area']
        area_sqm = self.data['Area_Sqm']

        target_if_more_than_10000 = (self.data['Area_Sqm'] <= 10000)

        # my_df_feature = DataFrame({'lu_code':lu_code,'shape_area':shape_area,'area_sqm':area_sqm})
        my_df_feature = DataFrame({'area_sqm': area_sqm})
        clf = svm.SVC(gamma='scale',kernel='rbf')
        clf.fit(my_df_feature, target_if_more_than_10000)
        print('my score ',clf.score(my_df_feature, target_if_more_than_10000))

        saved_model = bentoml.sklearn.save_model("area_sqm_svm_clf", clf)






