import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd

from pathlib import Path

from model import GeoSIR
from utils import InfecStatus

data_path = Path(__file__).resolve().parent / 'data'
images_path = Path(__file__).resolve().parent / 'images'
images_path.mkdir(exist_ok=True)
flow_path = data_path / 'flow'

# Data
tracts_df = gpd.read_file(data_path / 'tracts.shp').to_crs(epsg=3857)
agents_tract_df = pd.read_csv(data_path / 'agents_tract.csv')
prob_stay_at_home_data = (
  pd.read_csv(
    data_path / 'agents_home.csv',
    parse_dates=['date'],
    date_format='%Y-%m-%d'
  )
  .assign(date=lambda x: x['date'].dt.date)
  .set_index(['date', 'tract'])
  .squeeze()
)
percentage_time_at_home_data = (
  pd.read_csv(
    data_path / 'agents_percentage_home.csv',
    parse_dates=['date'],
    date_format='%Y-%m-%d'
  )
  .assign(
    date=lambda x: x['date'].dt.date,
    percentage_time_home=lambda x: x['percentage_time_home'] / 100
  )
  .set_index(['date', 'tract'])
  .squeeze()
)

# Model
infection_params = {
  'beta': 0.5,
  'gamma': 1/2
}
initial_condition = {
  InfecStatus.S: 0.99,
  InfecStatus.I: 0.01,
  InfecStatus.R: 0.00
}
exposure_distance = 100
model = GeoSIR(
  infection_params=infection_params,
  initial_condition=initial_condition,
  exposure_distance=exposure_distance,
  tracts_df=tracts_df,
  agents_tract_df=agents_tract_df,
  prob_stay_at_home_data=prob_stay_at_home_data,
  percentage_time_at_home_data=percentage_time_at_home_data,
  flow_path=flow_path,
  seed=42
)

for _ in range(30):
  model.step()

print(model.datacollector.get_model_vars_dataframe())