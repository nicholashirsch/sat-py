from . import propagation, orbits
from ui import rendering


class Mission:
    def __init__(
            self,
            starting_orbit: orbits.Orbit,
            maneuvers,
            initial_global_time: float,
            final_global_time: float,
            propagator: propagation.base.Propagator = None,
            satellite=None,
    ):
        # Instantiate all the passed-in attributes.
        self.starting_orbit = starting_orbit
        self.global_time = initial_global_time
        self.initial_global_time = initial_global_time
        self.final_global_time = final_global_time

        # For both the propagator and satellite a default option exists if the user does not input one, if they did
        # ignore and simply instantiate as normal.
        if satellite is None:
            self.satellite = ...  # TODO: Add a generic cube-sat.
        else:
            self.satellite = satellite

        if propagator is None:
            self.propagator = propagation.universal_variable.UniversalVariable(solver_tol=1e-8)
        else:
            self.propagator = propagator

        # TODO: Add maneuvers.
        self.maneuvers = maneuvers

        # Pre-instantiate attributes to be assigned later.
        self.traj = ...

    def simulate(self):
        """
        Call the propagator's propagate() function to generate the orbital trajectory and then log it.
        """

        self.propagator.setup(
            orbit=self.starting_orbit,
            final_time=self.final_global_time
        )
        self.propagator.propagate()
        self.traj = self.propagator.position_history

    def display(self):
        """
        Use pygfx to display the resulting trajectory.
        """

        # TODO: Add an error here if simulate() has not yet been called.
        engine = rendering.TempRenderEngine()
        engine.draw_orbit(self.traj)
        engine.render()
