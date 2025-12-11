import orbits
import numpy as np
import scipy as sp
from abc import ABC, abstractmethod


class Propagator(ABC):
    def __init__(self, orbit, final_time, step_size):
        self.orbit = orbit
        self.step_size = step_size

        self.initial_time = orbit.time
        self.final_time = final_time

        self.position_history = [orbit.position]
        self.velocity_history = [orbit.velocity]
        self.time_history = [orbit.time]

    @abstractmethod
    def propagate(self):
        pass


class Keplerian(Propagator):
    def __init__(self, orbit, initial_time, final_time, step_size, tol, fg_constraint=True):
        self.fg_constraint = fg_constraint
        self.tol = tol
        super().__init__(orbit, final_time, step_size)

    def propagate(self):
        for _ in range(self.initial_time, self.final_time, self.step_size):
            position_old = self.orbit.position.copy()
            velocity_old = self.orbit.velocity.copy()

            eccentric_anomaly_old = self.gauss_equation()
            eccentric_anomaly = self.kepler_equation()

            self.orbit.time += self.step_size

            f_func = (
                    1 - self.orbit.sm_axis / np.linalg.norm(position_old)
                        * (1 - np.cos(eccentric_anomaly - eccentric_anomaly_old))
            )
            g_func = (
                    self.step_size - 1 / self.orbit.mean_motion
                        * (eccentric_anomaly - eccentric_anomaly_old - np.sin(eccentric_anomaly - eccentric_anomaly_old))
            )

            self.orbit.position = np.array([f_func, g_func]) @ position_old

            fdot_func = (
                -np.sqrt(self.orbit.grav_param * self.orbit.sm_axis)
                    / (np.linalg.norm(position_old) * np.linalg.norm(self.orbit.position))
                    * np.sin(eccentric_anomaly - eccentric_anomaly_old)
            )
            if self.fg_constraint:
                gdot_func = (g_func * fdot_func + 1) / f_func
            else:
                gdot_func = (
                        1 - self.orbit.sm_axis / np.linalg.norm(self.orbit.position)
                            * (1 - np.cos(eccentric_anomaly - eccentric_anomaly_old))
                )

            self.orbit.velocity = np.array([fdot_func, gdot_func]) @ velocity_old

            self.time_history.append(self.time_history[-1] + self.step_size)
            self.position_history.append(self.orbit.position)
            self.velocity_history.append(self.orbit.velocity)

    def gauss_equation(self):
        return (
                2 * np.arctan(np.sqrt((1 - self.orbit.eccentricity) / (1 + self.orbit.eccentricity)))
                    * np.tan(self.orbit.true_anomaly / 2)
        )

    def kepler_equation(self):
        eccentric_anomaly = self.gauss_equation()
        eq = lambda x: (
                self.orbit.mean_motion * (-self.step_size)
                    -  (eccentric_anomaly - x)
                    - self.orbit.eccentricity * (np.sin(eccentric_anomaly) - np.sin(x))
        )

        return sp.optimize.minimize_scalar(eq, tol=self.tol).x
