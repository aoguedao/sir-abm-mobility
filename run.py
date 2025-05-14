import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import mesa
import pandas as pd
import geopandas as gpd
from datetime import date
from pathlib import Path

from sir_abm_mobility.model import GeoSIR
from sir_abm_mobility.utils import InfecStatus


data_path = Path(__file__).resolve().parent / 'data'
images_path = Path(__file__).resolve().parent / 'images'
images_path.mkdir(exist_ok=True)
epsg = 3857

# Model
infection_params = {
  'beta': 0.5,
  'gamma': 1 / 14
}
initial_condition = {
  "S": 0.99,
  "I": 0.01,
  "R": 0.00
}
exposure_distance = 50
avg_trips = 2.6

min_date = "2020/03/01"
max_date = "2020/05/31"
population_percentage = 0.01

model = GeoSIR(
  data_path=str(data_path),
  infection_params=infection_params,
  initial_condition=initial_condition,
  exposure_distance=exposure_distance,
  avg_trips=avg_trips,
  min_date=min_date,
  max_date=max_date,
  population_percentage=population_percentage,
  epsg=epsg
)

MAX_STEPS = 273
for _ in range(MAX_STEPS):
  model.step()

result_df = model.datacollector.get_model_vars_dataframe()
print(result_df)

