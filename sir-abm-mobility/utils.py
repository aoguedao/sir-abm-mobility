from enum import Enum, Flag, IntEnum, auto

class InfecStatus(Flag):
  '''
  Infection Status
  '''
  S = auto()  # Susceptible
  I = auto()  # Infected
  R = auto()  # Recovered


class TimeBlock(IntEnum):
  '''
  Time Blocks
  '''
  MORNING = 1
  AFTERNOON = 2
  EVENING = 3

  def next(self):
    cls = self.__class__
    members = list(cls)
    index = members.index(self) + 1
    if index >= len(members):
      index = 0
    return members[index]


class Decision(Flag):
  '''
  People's decision
  '''
  STAY_HOME = auto()
  GO_OUT = auto()
