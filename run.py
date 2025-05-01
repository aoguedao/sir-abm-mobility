# %%
import numpy as np
import matplotlib.pyplot as plt
import mesa
import pandas as pd
import geopandas as gpd

from datetime import date
from pathlib import Path

from sir_abm_mobility.model import GeoSIR
from sir_abm_mobility.utils import InfecStatus


MAX_STEPS = 30

data_path = Path(__file__).resolve().parent / 'data'
images_path = Path(__file__).resolve().parent / 'images'
images_path.mkdir(exist_ok=True)
flow_path = data_path / 'flow'
tracts_filepath = data_path / 'tracts.shp'
agents_tract_filepath = data_path / 'agents_tract.csv'
prob_stay_at_home_filepath = data_path / 'agents_home.csv'
percentage_time_at_home_filepath = data_path / 'agents_percentage_home.csv'
epsg = 3857

# Model
infection_params = {
  'beta': 0.5,
  'gamma': 1 / 14
}
initial_condition = {
  InfecStatus.S: 0.99,
  InfecStatus.I: 0.01,
  InfecStatus.R: 0.00
}
exposure_distance = 100
avg_trips = 2.6

min_date = date(year=2020, month=3, day=1)
max_date = date(year=2020, month=5, day=31)

model = GeoSIR(
  infection_params=infection_params,
  initial_condition=initial_condition,
  exposure_distance=exposure_distance,
  avg_trips=avg_trips,
  tracts_filepath=tracts_filepath,
  agents_tract_filepath=agents_tract_filepath,
  prob_stay_at_home_filepath=prob_stay_at_home_filepath,
  percentage_time_at_home_filepath=percentage_time_at_home_filepath,
  flow_path=flow_path,
  min_date=min_date,
  max_date=max_date,
  epsg=epsg
)
# %%
MAX_STEPS = 2
for _ in range(MAX_STEPS):
  model.step()

result_df = model.datacollector.get_model_vars_dataframe()
print(result_df)