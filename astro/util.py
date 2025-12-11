import numpy as np
from dynamics import dcms


def elements_2_state(
        sm_axis,
        eccentricity,
        raan,
        argp,
        inclination,
        true_anomaly,
        grav_param=3.986004418e14  # Default to Earth in units of m^3/s^2.
):
    parameter = sm_axis * (1 - eccentricity ** 2)
    pos_magnitude = parameter / (1 + eccentricity * np.cos(true_anomaly))
    pos_magnitude_dt = np.sqrt(grav_param * parameter) * np.sin(true_anomaly) / parameter
    true_anomaly_dt = np.sqrt(grav_param * parameter) * np.cos(true_anomaly)

    local_2_perifocal_dcm = dcms.euler_2_dcm(true_anomaly, 3).T
    perifocal_2_inertial_dcm = (
        dcms.euler_2_dcm(raan, 3).T
            @ dcms.euler_2_dcm(inclination, 1).T
            @ dcms.euler_2_dcm(argp, 3).T
    )

    position = np.array([[pos_magnitude], [0], [0]])
    velocity = np.array([[pos_magnitude_dt], [pos_magnitude * true_anomaly_dt], [0]])

    position = perifocal_2_inertial_dcm @ local_2_perifocal_dcm @ position
    velocity = perifocal_2_inertial_dcm @ local_2_perifocal_dcm @ velocity

    return position, velocity


def state_2_elements(
        position,
        velocity,
        grav_param=3.986004418e14  # Default to Earth in units of m^3/s^2.
):
    unit_vec_1 = np.array([[1], [0], [0]])
    unit_vec_2 = np.array([[0], [1], [0]])
    unit_vec_3 = np.array([[0], [0], [1]])

    spf_ang_momentum = np.cross(position, velocity)
    eccentricity_vec = np.cross(velocity, spf_ang_momentum) / grav_param - position / np.linalg.norm(position)
    node_vec = np.cross(unit_vec_3, spf_ang_momentum)

    parameter = np.linalg.norm(spf_ang_momentum) ** 2 / grav_param

    eccentricity = np.linalg.norm(eccentricity_vec)
    sm_axis = parameter / (1 - eccentricity ** 2)
    raan = np.arctan2(
        np.dot(node_vec, unit_vec_2),
        np.dot(node_vec, unit_vec_1)
    )
    inclination = np.arctan2(
        np.dot(spf_ang_momentum, np.dot(node_vec, unit_vec_3)),
        np.linalg.norm(node_vec) * np.dot(spf_ang_momentum, unit_vec_3)
    )
    argp = np.arctan2(
        np.linalg.norm(node_vec) * np.dot(eccentricity_vec, np.cross(spf_ang_momentum, node_vec)),
        np.linalg.norm(np.cross(spf_ang_momentum, node_vec)) * np.dot(eccentricity_vec, node_vec)
    )
    true_anomaly = np.arctan2(
        np.dot(position, np.cross(spf_ang_momentum, eccentricity_vec)),
        np.linalg.norm(spf_ang_momentum) * np.dot(position, eccentricity_vec)
    )

    if raan < 0:
        raan += 2 * np.pi
    if inclination < 0:
        inclination += 2 * np.pi
    if argp < 0:
        argp += 2 * np.pi
    if true_anomaly < 0:
        true_anomaly += 2 * np.pi

    return sm_axis, eccentricity, raan, inclination, argp, true_anomaly
