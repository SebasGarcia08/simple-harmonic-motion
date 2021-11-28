from dataclasses import dataclass
import numpy as np


@dataclass
class SimpleHarmonicSystem:
    theta_0: float
    omega: float
    phi: float
    m: float
    L: float
    g: float

    def theta(self, t: float) -> float:
        return self.theta_0 * np.cos(self.omega * t + self.phi)

    def velocity(self, t: float) -> float:
        return -self.theta_0 * self.omega * np.sin(self.omega * t + self.phi)

    def acceleration(self, t: float) -> float:
        return -self.omega ** 2 * self.theta(t)

    def ek(self, t: float) -> float:
        return (
            (1 / 1)
            * self.m
            * (self.theta_0 * self.omega * np.sin(self.omega * t + self.phi))
        )

    def eu(self, t: float) -> float:
        return self.m * self.g * self.L * (1 - np.cos(self.theta(t)))
