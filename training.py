import bentoml
import geopandas as gpd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

from src.data_input.repositories import shape_data_prepare
from src.data_input.repositories.Feature_Converter import FeatureConverter
from src.data_processing.transformer import Transformer

filePath = "D:/HighSpeedStorage/LVCHighSpd/MeenMookCoProject/POC/research/land-use-data/Landuse_bkk/กรุงเทพมหานคร2566/การใช้ที่ดิน/LU_BKK_2566.shp"

data = gpd.read_file(filePath)

feature_converter = FeatureConverter(['Area_Rai','Shape_Area'])
new_data = feature_converter.transform(data)

# prepare_data = shape_data_prepare.prepare_data(new_data,['Area_Rai','Shape_Area'],['LU_DES_EN'])

lu_code = data['LU_CODE']
lu_des_en = data['LU_DES_EN']
area_rai = data['Area_Rai']

model = Transformer()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print(data.columns)
print(lu_code)
print(lu_des_en)
print(area_rai)

def generate_batch(batch_size=95, seq_len=100):
    time = torch.linspace(0, 10, seq_len)
    trends = torch.stack([torch.sin(time) + 0.1*torch.randn(seq_len) + 0.05*time for _ in range(batch_size)])
    return trends[:, :-5], trends[:, -5:]  # Input, target

for epoch in range(1000):
    inputs, targets = generate_batch()
    optimizer.zero_grad()
    outputs = model(inputs)  # [seq_len, batch_size]
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

bentoml.pytorch.save(
    "trend_transformer",
    model,
    signatures={
        "predict": {"batchable": True}
    }
)




## Assumption of land usage, find out which integrated farm
# is the correct one that people used for masking their true land usage
# for arbitrage
## Smaller Integrated farm could be a plot for future sells
### Determine the LU_CODE from Area and Shape and geometry of plot, Classification problem

