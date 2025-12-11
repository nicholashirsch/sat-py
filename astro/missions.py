import propagation

class Mission:
    def __init__(
            self,
            starting_orbit,
            maneuvers,
            initial_global_time,
            final_global_time,
            step_size,
            tol=1e-8,
            propagator="Keplerian",
            satellite=None,
    ):
        self.satellite = satellite
        self.propagator = propagator
        self.starting_orbit = starting_orbit
        self.maneuvers = maneuvers
        self.tol=tol
        self.step_size = step_size  # TODO: Add some logic to make this automatic if not set by the user.

        self.global_time = initial_global_time
        self.initial_global_time = initial_global_time
        self.final_global_time = final_global_time

        self.traj = ...

    def simulate(self):
        match self.propagator:
            case "Keplerian":
                self.traj = propagation.Keplerian(
                    self.starting_orbit,
                    initial_time=0,
                    final_time=self.final_global_time - self.initial_global_time,
                    step_size=self.step_size,
                    tol=self.tol,
                )

    def display(self):
        pass
