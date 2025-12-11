import numpy as np
import util


class Orbit:
    def __init__(
            self,
            position,
            velocity,
            time,
            grav_param=3.986004418e14  # Default to Earth in units of m^3/s^2.
    ):
        self.position = position
        self.velocity = velocity
        self.time = time
        self.grav_param = grav_param

    @classmethod
    def from_state(
            cls,
            position,
            velocity,
            grav_param=3.986004418e14  # Default to Earth in units of m^3/s^2.
    ):
        return cls(position, velocity, grav_param)

    @classmethod
    def from_orbital_elements(
            cls,
            sm_axis,
            eccentricity,
            raan,
            inclination,
            argp,
            true_anomaly,
            grav_param=3.986004418e14  # Default to Earth in units of m^3/s^2.
    ):
        position, velocity = util.elements_2_state(
            sm_axis=sm_axis,
            eccentricity=eccentricity,
            raan=raan,
            inclination=inclination,
            argp=argp,
            true_anomaly=true_anomaly,
            grav_param=grav_param
        )
        return cls(position, velocity, grav_param)

    @property
    def spf_angular_momentum(self):
        return np.cross(self.position, self.velocity)

    @property
    def eccentricity(self):
        eccentricity_vec = (
                np.cross(self.velocity, self.spf_angular_momentum) / self.grav_param
                    - self.position / np.linalg.norm(self.position)
        )
        return np.linalg.norm(eccentricity_vec)

    @property
    def sm_axis(self):
        return np.linalg.norm(self.spf_angular_momentum) ** 2 / self.grav_param / (1 - self.eccentricity ** 2)

    @property
    def raan(self):
        unit_vec_1 = np.array([[1], [0], [0]])
        unit_vec_2 = np.array([[0], [1], [0]])
        unit_vec_3 = np.array([[0], [0], [1]])

        node_vec = np.cross(unit_vec_3, self.spf_angular_momentum)

        raan = np.arctan2(
            np.dot(node_vec, unit_vec_2),
            np.dot(node_vec, unit_vec_1)
        )
        if raan < 0:
            raan += 2 * np.pi

        return raan

    @property
    def inclination(self):
        unit_vec_3 = np.array([[0], [0], [1]])

        node_vec = np.cross(unit_vec_3, self.spf_angular_momentum)

        inclination = np.arctan2(
            np.dot(self.spf_angular_momentum, np.dot(node_vec, unit_vec_3)),
            np.linalg.norm(node_vec) * np.dot(self.spf_angular_momentum, unit_vec_3)
        )
        if inclination < 0:
            inclination += 2 * np.pi

        return inclination

    @property
    def argp(self):
        unit_vec_3 = np.array([[0], [0], [1]])

        eccentricity_vec = (
                np.cross(self.velocity, self.spf_angular_momentum) / self.grav_param
                - self.position / np.linalg.norm(self.position)
        )
        node_vec = np.cross(unit_vec_3, self.spf_angular_momentum)

        argp = np.arctan2(
            np.linalg.norm(node_vec) * np.dot(eccentricity_vec, np.cross(self.spf_angular_momentum, node_vec)),
            np.linalg.norm(np.cross(self.spf_angular_momentum, node_vec)) * np.dot(eccentricity_vec, node_vec)
        )
        if argp < 0:
            argp += 2 * np.pi

        return argp

    @property
    def true_anomaly(self):
        eccentricity_vec = (
                np.cross(self.velocity, self.spf_angular_momentum) / self.grav_param
                - self.position / np.linalg.norm(self.position)
        )

        true_anomaly = np.arctan2(
            np.dot(self.position, np.cross(self.spf_angular_momentum, eccentricity_vec)),
            np.linalg.norm(self.spf_angular_momentum) * np.dot(self.position, eccentricity_vec)
        )
        if true_anomaly < 0:
            true_anomaly += 2 * np.pi

        return true_anomaly

    @property
    def mean_motion(self):
        return np.sqrt(self.grav_param / self.sm_axis ** 3)
