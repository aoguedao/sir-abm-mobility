import mesa
import mesa_geo as mg
import numpy as np
import pandas as pd
import geopandas as gpd

from pyproj import CRS
from shapely.geometry import Point
from typing import Any

from utils import InfecStatus, TimeBlock, Decision


class TractAgent(mg.GeoAgent):
  '''Tract Agent.'''

  def __init__(self, model, geometry, crs, code=None, population=None):
    super().__init__(model, geometry, crs)
    self.code = code
    self.population = population
    self._prob_stay_at_home = None  # Probability of stay at home
    self._perc_time_at_home = None  # Percentage of time staying at home
    self._prob_flow = None  # Flow probabilities
    self._people = None  # AgentSet with people living here
    self.n_stay_at_home = 0
    self._gpd_geometry = gpd.GeoSeries(self.geometry)

  def __repr__(self):
    return f'Tract Code {self.code} (ID {self.unique_id})'

  def pre_step(self):
    match self.model.time_block:
      case TimeBlock.MORNING:
        self.daily_reset()
      case TimeBlock.AFTERNOON:
        pass
      case TimeBlock.EVENING:
        pass

  def update_data(self):
    self.prob_stay_at_home = self.model.prob_stay_at_home_data.loc[(self.model.today, self.code)]
    self.perc_time_at_home = self.model.percentage_time_at_home_data.loc[(self.model.today, self.code)]
    self.prob_flow = self.model.flow.loc[self.code, :].to_dict()

  def choose_next_tract_code(self):
    next_tract_id = self.random.choices(
      list(self.prob_flow.keys()),
      weights=list(self.prob_flow.values())
    )
    return next_tract_id[0]

  def sample_points(self, n=1, method='uniform', **kwargs):
    return self._gpd_geometry.sample_points(size=n, method=method, **kwargs)

  def daily_reset(self):
    self.n_stay_at_home = 0

  @property
  def prob_stay_at_home(self):
    return self._prob_stay_at_home

  @prob_stay_at_home.setter
  def prob_stay_at_home(self, value):
    if not isinstance(value, float) and (0 <= value <= 1):
      raise TypeError('Probability of Stay at Home must be a float between 0 and 1.')
    self._prob_stay_at_home = value

  @property
  def perc_time_at_home(self):
    return self._perc_time_at_home

  @perc_time_at_home.setter
  def perc_time_at_home(self, value):
    if not isinstance(value, float) and (0 <= value <= 100):
      raise TypeError('Percentage of time at home must be a float between 0 and 100.')
    self._perc_time_at_home = value

  @property
  def prob_flow(self):
    return self._prob_flow

  @prob_flow.setter
  def prob_flow(self, value):
    if not isinstance(value, dict):
      raise TypeError('Probability flow must be a dictionary')
    self._prob_flow = value

  @property
  def people(self):
    return self._people

  @people.setter
  def people(self, value):
    if not isinstance(value, mesa.agent.AgentSet):
      raise TypeError('People must be an AgenSet.')
    self._people = value


class PersonAgent(mg.GeoAgent):
  '''Person Agent.'''

  def __init__(
    self,
    model,
    geometry: Point,
    crs: Any,
    home_tract: TractAgent,
    status: InfecStatus
  ):
    if status not in InfecStatus:
      raise ValueError(f'Status must be one of {InfecStatus.__members__}')

    super().__init__(model, geometry, crs)
    self.home_tract = home_tract
    self.home_pos = self.geometry  # Initial position is at home
    self._tract = home_tract  # Current TractAgent, it will change
    self.tract = self._tract
    self._status = status
    self._decision = None
    self.steps_in_status = 0  # Since the last time it changed status


  def __repr__(self):
    return f'Person {self.unique_id}'

  def step(self):
    match self.model.time_block:
      case TimeBlock.MORNING:
        self.morning_step()
      case TimeBlock.AFTERNOON:
        self.afternoon_step()
      case TimeBlock.EVENING:
        self.evening_step()
    self.update_infection_status()
    self.model.counts[self.status] += 1  # Count agent type

  def morning_step(self):
    # Take decision
    if self.random.random() < self.tract.prob_stay_at_home:
      self.decision = Decision.STAY_HOME
      self.tract.n_stay_at_home += 1
    else:
      self.decision = Decision.GO_OUT
      self.move_to_next_tract()

  def afternoon_step(self):
    match self.decision:
      case Decision.STAY_HOME:
        pass
      case Decision.GO_OUT:
        self.move_to_next_tract()

  def evening_step(self):
    match self.decision:
      case Decision.STAY_HOME:
        pass
      case Decision.GO_OUT:
        self.move_to_next_tract()

  def move_to_next_tract(self, method='uniform'):
    next_tract_code = self.tract.choose_next_tract_code()
    self.tract = self.model.code_tract_dict[next_tract_code]
    next_pos = self.tract.sample_points(n=1, method=method).iat[0]
    self.geometry = next_pos

  def move_to_home(self):
    self.geometry = self.home_pos
    self.update_infection_status()

  def update_infection_status(self):
    match self.status:
      # Susceptible
      case InfecStatus.S:
        neighbors = self.model.space.get_neighbors_within_distance(
          self, self.model.exposure_distance
        )
        for neighbor in neighbors:
          if (
            isinstance(neighbor, PersonAgent)
            and neighbor.status is InfecStatus.I
            and self.random.random() < self.model.infect_params['beta']
          ):
            self.status = InfecStatus.I
            self.steps_in_status = 0
            break  # stop process if agent becomes infected
        if self.status is InfecStatus.S:
          self.steps_in_status += 1
      # Infected
      case InfecStatus.I:
        recovery_steps = 1 / self.model.infect_params['gamma'] * len(InfecStatus)  # Each day
        if self.steps_in_status >= recovery_steps:
          self.status = InfecStatus.R
          self.steps_in_status = 0
        else:
          self.steps_in_status += 1
      # Recovered
      case InfecStatus.R:
        self.steps_in_status += 1

  @property
  def status(self):
    return self._status

  @status.setter
  def status(self, value):
    if value not in InfecStatus:
      raise TypeError(f'Status must be InfecStatus member: {InfecStatus._member_names_}')
    self._status = value

  @property
  def tract(self):
    return self._tract

  @tract.setter
  def tract(self, value):
    if value not in self.model.agents_by_type[TractAgent]:
      raise TypeError(f'Tract must be a TractAgent instance registered in the model')
    self._tract = value

  @property
  def decision(self):
    return self._decision

  @decision.setter
  def decision(self, value):
    if value not in Decision:
      raise TypeError(f'Decision must be a Decision member: {Decision._member_names_}')
    self._decision = value
