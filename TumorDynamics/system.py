import numpy as np
from scipy.integrate import odeint
from abc import ABC, abstractmethod

class DynamicalSystem(ABC):
    @abstractmethod
    def _f(self,v,t):
        pass

    @property
    def v(self)->np.ndarray:
        return self._v

    @abstractmethod
    def integrate(self):
        pass
