from typing import Optional, Callable
from dataclasses import dataclass
import numpy as np


@dataclass
class SimpleHarmonicSystem:
    theta_0: float
    phi: float
    m: float
    L: float
    g: float

    def __post_init__(self):
        self.omega = np.sqrt(self.g / self.L)
        self._t = 0
        self._theta = None

    @property
    def t(self) -> float:
        return self._t

    @t.setter
    def t(self, value: float) -> None:
        self._t = value

    @property
    def theta(self) -> float:
        if self._theta is not None:
            return self._theta
        self._theta = self.theta_0 * np.cos(self.omega * self.t + self.phi)
        return self._theta

    @theta.setter
    def theta(self, value: float) -> None:
        self._theta = value

    def velocity(self) -> float:
        return -self.theta_0 * self.omega * np.sin(self.omega * self.t + self.phi)

    def acceleration(self) -> float:
        result = -self.omega ** 2 * self.theta
        return result

    def kinetic_energy(self) -> float:
        return (1 / 2.0) * self.m * self.velocity() ** 2

    def potential_energy(self) -> float:
        return self.m * self.g * self.L * (1 - np.cos(self.theta))

    def mechanical_energy(self) -> float:
        return self.potential_energy() + self.kinetic_energy()
