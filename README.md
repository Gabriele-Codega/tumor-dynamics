# Tumor dynamcis
This code implements some utilities to study the stochastic dynamics of the interaction between a tumor and immune system in presence of chemotherapy.

The model is implemented according to what proposed in [this paper](https://www.sciencedirect.com/science/article/pii/S0378437123003904?via%3Dihub) (which in turn is an extension of the model proposed by [Kuznetsov et al.](https://doi.org/10.1016/S0092-8240(05)80260-5)).


## TumorDynamics
The files are organised as follows

### immune_system.py
---
Implements a class (`DeterministicImmuneSystem`) for the representation of the immune system. Its main features are:
- a function that defines the rhs of the dynamical system describing the evolution of the system
- a method to integrate the system and get its evolution in time
- methods to compute the period of limit cycles
- a method to plot the phase space trajectory

### stochastic_system.py
---
Implements a `StochasticImmuneSystem`, that inherits from `DeterministicImmuneSystem` and represents an immune system in presence of a disturbance in the intensity of chemotherapy. Its main features are:
- a function that describes the stochastic part of the system
- methods to integrate the equations and simulate random trajectories
- methods to compute the stochastic sensitivity matrix and function
- a method to plot the phase space trajectory

### plotting_utils.py
---
Utilities to build confidence bands and ellipses, using the method of stochastic sensitivity function (see for instance [this paper](http://dx.doi.org/10.1103/PhysRevE.87.052711)).