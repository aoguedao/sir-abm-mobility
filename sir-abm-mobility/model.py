import geopandas as gpd
import mesa
import mesa_geo as mg
import numpy as np
import pandas as pd

from datetime import date, datetime, timedelta
from pathlib import Path

from agents import PersonAgent, TractAgent
from utils import InfecStatus, TimeBlock, Decision


class GeoSIR(mesa.Model):

  def __init__(
    self,
    infection_params: dict,
    initial_condition: dict,
    exposure_distance: float,
    tracts_df: gpd.GeoDataFrame,
    agents_tract_df: pd.DataFrame,
    prob_stay_at_home_data: pd.Series,
    percentage_time_at_home_data: pd.Series,
    flow_path: Path,
    min_date: date = None,
    max_date: date = None,
    seed: int = None,
  ):
    super().__init__(seed=seed)
    self.infect_params = infection_params
    self.initial_condition = initial_condition
    self.exposure_distance = exposure_distance
    self.tracts_df = tracts_df
    self.agents_tract_df = agents_tract_df
    self.prob_stay_at_home_data = prob_stay_at_home_data
    self.percentage_time_at_home_data = percentage_time_at_home_data
    self.flow_path = flow_path
    self.min_date = min_date if min_date is not None else date.min
    self.max_date = max_date if max_date is not None else date.max
    self.start_date = None
    self.end_date = None
    self.today = None
    self.time_block = None
    self.current_flow_date = None
    self.next_flow_date = None
    self.flow = None
    self.counts = {}
    self.running = True

    self.space = mg.GeoSpace(crs=tracts_df.crs)
    self.datacollector = mesa.DataCollector(
      model_reporters={
        'date': 'today',
        'time_block': 'time_block',
        'S': lambda m: m.counts[InfecStatus.S],
        'I': lambda m: m.counts[InfecStatus.I],
        'R': lambda m: m.counts[InfecStatus.R],
      }
    )
    self.init_time()
    self.init_tracts()
    self.init_population()
    self.reset_counts()
    print("Model ready!")

  def step(self):
    print(f'step {self.steps}')
    # Pre-step
    self.reset_counts()
    self.agents_by_type[TractAgent].do('pre_step')

    # People steps
    self.agents_by_type[PersonAgent].shuffle_do('step')

    # Post-step
    match self.time_block:
      case TimeBlock.MORNING:
        pass
      case TimeBlock.AFTERNOON:
        pass
      case TimeBlock.EVENING:
        self.agents_by_type[PersonAgent].shuffle_do('move_to_home')

    self.datacollector.collect(self)
    self.update_date_and_timeblock()
    if (self.today == self.end_date) and (self.time_block is TimeBlock.EVENING):
      self.running = False  # Stop


  def init_time(self):
    print("Initializing Time")
    self.flow_dates = []  #  List of flow dates
    self.flow_filepaths = {}  # Dict. of flow dates and filepath
    for filepath in self.flow_path.glob('agents_flow_????-??-??.csv'):
      flow_date = datetime.strptime(filepath.stem.split('_')[-1], "%Y-%m-%d").date()
      if self.min_date <= flow_date <= self.max_date:
        self.flow_dates.append(flow_date)
        self.flow_filepaths[flow_date] = filepath
    self.flow_dates.sort()  # Sort dates

    # Initializations
    self.start_date = self.flow_dates[0]  # First flow date
    self.end_date = self.flow_dates[-1]  + timedelta(days=6)  # Last flow date + 1 week
    self.today = self.flow_dates[0]  # Init today
    self.time_block = TimeBlock.MORNING  # Init at morning
    self.current_next_flow_date_iter = iter(zip(self.flow_dates, self.flow_dates[1:]))
    self.current_flow_date, self.next_flow_date = next(self.current_next_flow_date_iter)
    self.flow = self._read_flow(self.current_flow_date)


  def init_tracts(self):
    print("Initializing Tracts")
    tract_creator = mg.AgentCreator(TractAgent, model=self)
    tracts_and_pop = (
      self.tracts_df.merge(
        self.agents_tract_df,
        how='left',
        on='tract',
        validate='1:1'
      )
      .rename(columns={'tract': 'code', 'n_agents': 'population'})
    )
    tract_agents = tract_creator.from_GeoDataFrame(tracts_and_pop)
    self.space.add_agents(tract_agents)
    self.code_tract_dict = {t.code: t for t in self.agents_by_type[TractAgent]}
    self.agents_by_type[TractAgent].do('update_data')


  def init_population(self):
    print("Initializing Population")
    for tract in self._agents_by_type[TractAgent]:
      population = int(tract.population / 10) # TODO: CHANGE THIS IN THE FINAL VERSION!
      coords = (
        gpd.GeoSeries(tract.geometry)
        .sample_points(population)
        .explode()
        .to_numpy()
      )
      statuses = self.random.choices(
        list(self.initial_condition.keys()),
        k=population,
        weights=list(self.initial_condition.values())
      )
      person_agents = PersonAgent.create_agents(
        self,
        n=population,
        geometry=coords,
        crs=self.space.crs,
        home_tract=tract,
        status=statuses
      )
      self.space.add_agents(person_agents)
      tract.people = person_agents


  def _read_flow(self, flow_date):
    flow_df = (
      pd.read_csv(self.flow_filepaths[flow_date])
      .pivot(index='origin', columns='destination', values='flow_ratio')
    )
    return flow_df


  def reset_counts(self):
    self.counts = {status: 0 for status in InfecStatus}


  def update_disease(self):
    pass


  def update_date_and_timeblock(self):
    if self.time_block is TimeBlock.EVENING:
      self.today += timedelta(days=1)  # If evening, update to next day
      if self.today == self.next_flow_date:  # Check if next flow dataset
        self.current_flow_date, self.next_flow_date = next(self.current_next_flow_date_iter)
        self.flow = self._read_flow(self.current_flow_date)
        self.agents_by_type[TractAgent].do('update_data')
    self.time_block = self.time_block.next()  # Next TimeBlock


  def get_tract_id(self, code):
    return self.tract_code_to_id_dict[code]