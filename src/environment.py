import random

from gym import Env
from gym.spaces import *

from edge_server import *
from user_equipment import *
from utility import *


class VehicleEnvironment(Env):
    TIME_SLOT_MAX = 40

    def __init__(self):
        self.small_cell_total = random.randint(5, 50)
        self.vehicle_total = random.randint(1, 10)  # Number of vehicles in a small-cell

        self.ue_total = random.randint(4, 10)  # Number of UE connect to VEC operator
        self.ves_total = self.vehicle_total  # Number of total VES = K
        self.fes_total = self.small_cell_total  # Number of total FES = N

        self.current_time_slot = 0
        self.current_network_utility = 0
        self.state = None

        # List of UEs, VESs, FESs
        self.UEs = [UserEquipment() for _ in range(self.ue_total)]
        self.VESs = [VehicularEdgeServer(self.UEs) for _ in range(self.ves_total)]
        self.FESs = [FixedEdgeServer(self.UEs) for _ in range(self.ves_total)]

        # Model
        self.comm_model = CommunicationModel(self.UEs, self.VESs, self.FESs)
        self.comp_model = ComputationModel(self.UEs, self.VESs, self.FESs)

        # Utility
        self.utility = Utility(self.UEs, self.VESs, self.FESs)

        # State space
        self.observation_space = Dict({"ves_count": MultiDiscrete([self.ves_total] * self.ue_total),
                                       "ves_data_rate": Box(low=0.0, high=5, shape=[self.ue_total, self.ves_total]),
                                       "ves_resource": Box(low=0, high=20, shape=[self.ue_total, self.ves_total]),
                                       "fes_data_rate": Box(low=0.0, high=10, shape=[self.ue_total, self.fes_total]),
                                       "fes_resource": Box(low=0, high=100, shape=[self.ue_total, self.fes_total])})

        # Action we can take for each UE: Compute task locally, offload task to VES, offload task to FES
        self.action_space = Dict({"transition_state": MultiDiscrete([3] * self.ue_total),
                                  "ves_spectrum_rate": Box(low=0.0, high=1.0, shape=[self.ue_total, self.ves_total]),
                                  "fes_spectrum_rate": Box(low=0.0, high=1.0, shape=[self.ue_total, self.fes_total]),
                                  "ves_resource_rate": Box(low=0.0, high=1.0, shape=[self.ue_total, self.ves_total]),
                                  "fes_resource_rate": Box(low=0.0, high=1.0, shape=[self.ue_total, self.fes_total])})

    def _get_state(self):
        return self.state

    # def _get_obs(self):
    #     return {"ves_count": self._ves_count,
    #             "ves_data_rate": self._ves_data_rate, "ves_resource": self._ves_computation_resource,
    #             "fes_data_rate": self._fes_data_rate, "fes_resource": self._fes_computation_resource}

    def _get_info(self):
        pass

    def reset(self, **kwargs):
        # Reset time
        self.current_time_slot = 0

        # Reset cumulative utility
        self.current_network_utility = 0

        # Reset state
        self.state = self.observation_space.sample()

        return self.state

    def step(self, action):
        # Increase time by 1 slot
        self.current_time_slot += 1

        # Update current state
        observation = self.comm_model.get_state(action)
        self.state = observation

        # Calculate reward
        if self.utility.get_utility(action, self.state) >= self.current_network_utility:
            reward = 1
        else:
            reward = -1

        # Check if this is the last time slot to consider
        if self.current_time_slot >= self.TIME_SLOT_MAX:
            done = True
        else:
            done = False

        # Set placeholder for info
        info = {}

        return [observation, reward, done, info]

    def render(self, **kwargs):
        # Implement viz
        pass
