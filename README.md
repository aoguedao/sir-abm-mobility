# sir-abm-mobility

Agent-Based Model(ABM) for a SIR infectious diseases model using mobility patterns.

This model uses `mesa` and its extension `mesa-geo` as ABM framework.

There are two major agents that interacts in the simulation:

* `TractAgent`: Represents census tracts and contains information about the population, the probability of stay at home and origin-destination (flow) probabilities of mobility among other census tracts. Both probabilities change weekly (since the data has been gathered weekly).
* `PersonAgent`: Represents a person who interacts with the census tracts and other people. Each one has an infection status (S, I, R) that can be changed over the simulation.

Each day has been modeled as a series of Time-Blocks, for this first iteration we are using morning, afternoon and evening time blocks. Each one affects how agents interact with each other. For example, first time in the morning the person agent needs to decide if it will stay at home or will go out. On the other hand, at the end of evening each person agent goes back to their home.

_Disclaimer:_ This is a work-in-progress, however the modularized coding philoshophy behind this project allow to extend the model to more time blocks, infectious statuses or even people decisions.