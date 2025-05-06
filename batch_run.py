import numpy as np
import matplotlib.pyplot as plt
import mesa
import pandas as pd
import geopandas as gpd

from datetime import date
from pathlib import Path

from sir_abm_mobility.model import GeoSIR
from sir_abm_mobility.utils import InfecStatus


if __name__ == '__main__':

  data_path = Path(__file__).resolve().parent / 'data'
  images_path = Path(__file__).resolve().parent / 'images'
  images_path.mkdir(exist_ok=True)
  output_path = Path(__file__).resolve().parent / 'output'
  output_path.mkdir(exist_ok=True)
  flow_path = data_path / 'flow'

  tracts_filepath = data_path / 'tracts.shp'
  agents_tract_filepath = data_path / 'agents_tract.csv'
  prob_stay_at_home_filepath = data_path / 'agents_home.csv'
  percentage_time_at_home_filepath = data_path / 'agents_percentage_home.csv'
  epsg = 3857

  # Model
  infection_params = {
    'beta': 0.5,
    'gamma': 1/2
  }
  initial_condition = {
    "S": 0.99,
    "I": 0.01,
    "R": 0.00
  }
  exposure_distance = 100
  avg_trips = 2.6
  min_date = "2020/03/01"
  max_date = "2020/05/31"

  parameters = {
    "data_path": str(data_path),
    "infection_params": [infection_params],
    "initial_condition": [initial_condition],
    "exposure_distance": exposure_distance,
    "avg_trips": avg_trips,
    "epsg": epsg,
    "min_date": min_date,
    "max_date": max_date,
    "population_percentage": 1
  }

  MAX_STEPS = 200

  results = mesa.batch_run(
    GeoSIR,
    parameters,
    iterations=5,
    max_steps=MAX_STEPS,
    data_collection_period=1,
    number_processes=5,
  )
  today_str = date.today().strftime('%Y%m%d')
  results_df = pd.DataFrame(results)
  print(results_df.head())
  results_df.to_csv(output_path / f"results_{today_str}.csv")