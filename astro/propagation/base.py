from abc import ABC, abstractmethod
import numpy as np
from .. import orbits


class Propagator(ABC):
    """
    Base class for all propagators. All derivatives revolve around the method propagate() which takes in the initial
    orbital parameters (whatever those might be) and propagates them along the orbit up to the final time.
        This class effectively has two initializations. The first, the default __init__() is exposed to the user and is
    used to set solver settings. The second is the setup() method which fills in all the attributes needed to actually
    call propagate(). This is kept separate from __init__() because we want the former to only be for the user to set
    propagator settings before passing it into a Mission object. setup() can then be called internally by
    Mission.simulate() without exposing a bunch of extra code to the user.

    :ivar orbit: Orbit to perform propagation on. The position, and velocity, and time attributes of this object
        when it is passed in serve as the initial conditions of the orbit.
    :ivar final time: When to stop orbit propagation.
    :ivar step_size: Time step size for propagation.
    :ivar position_history: List of orbital positions (3, ) as a (3, length of propagation) array.
    :ivar velocity_history: List of orbital velocities (3, ) as a (3, length of propagation) array.
    :ivar time_history: List of time steps as a (1, length of propagation) array.
    """

    def __init__(self, step_size: float = None):
        """
        Pre-initialization, all these attributes (excluding step_size) are not filled in till setup() is called.
        """

        self.step_size = step_size

        self.orbit = None
        self.final_time = None
        self.position_history = None
        self.velocity_history = None
        self.time_history = None

    @abstractmethod
    def propagate(self):
        """
        Actual propagation is implemented here for child classes.

        NOTE: setup() MUST be called before this function in all circumstances.
        """

        pass

    def setup(
            self,
            orbit: orbits.Orbit,
            final_time: float,
    ):
        """
        Perform all the behind-the-scenes bookkeeping necessary to set up this object before propagating.
        """

        self.orbit = orbit
        self.final_time = final_time
        if self.step_size is None:  # Default to 10000 steps.
            self.step_size = (final_time - self.orbit.time) / 10000

        # Initialize history arrays. During propagation there will be N timesteps plus the initial timestep so the
        # arrays need to be of size N+1.
        timesteps = int(np.floor((self.final_time - orbit.time) / self.step_size))
        self.position_history = np.zeros([3, timesteps + 1])
        self.velocity_history = np.zeros([3, timesteps + 1])
        self.time_history = np.zeros([1, timesteps + 1])

        # Assign initial conditions to the history arrays.
        self.position_history[:, 0] = orbit.position
        self.velocity_history[:, 0] = orbit.velocity
        self.time_history[0, 0] = orbit.time
