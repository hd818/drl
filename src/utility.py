from model import *


class Utility(Model):
    def __init__(self, UEs, VESs, FESs):
        super().__init__(UEs, VESs, FESs)

        # Computation task from UEs
        self.tasks = [ue.generate_task() for ue in self.UEs]

    def calc_communication_utility(self, action, state, ue_index):
        transition_state = action["transition_state"][ue_index]

        ves_price = np.array([self.VESs[ves_index].spectrum_price[ue_index] for ves_index in range(self.ves_total)])
        fes_price = np.array([self.FESs[fes_index].spectrum_price[ue_index] for fes_index in range(self.fes_total)])

        utility = self.UEs[ue_index].spectrum_price * self.tasks[ue_index][0] \
                  - (transition_state == 1) * np.sum(np.multiply(ves_price, state["ves_data_rate"][ue_index])) \
                  - (transition_state == 2) * np.sum(np.multiply(fes_price, state["fes_data_rate"][ue_index]))
        return utility

    def calc_computation_utility(self, action, state, ue_index):
        transition_state = action["transition_state"][ue_index]

        ves_price = np.array([self.VESs[ves_index].resource_price[ue_index] for ves_index in range(self.ves_total)])
        fes_price = np.array([self.FESs[fes_index].resource_price[ue_index] for fes_index in range(self.fes_total)])

        utility = self.UEs[ue_index].resource_price * self.tasks[ue_index][1] \
                  - (transition_state == 1) * np.sum(np.multiply(ves_price, state["ves_resource"][ue_index])) \
                  - (transition_state == 2) * np.sum(np.multiply(fes_price, state["fes_resource"][ue_index]))
        return utility

    def calc_total_utility(self, action, state, ue_index):
        utility = self.calc_communication_utility(action, state, ue_index) \
                  + self.calc_computation_utility(action, state, ue_index)
        return utility

    def get_utility(self, action, state):
        utility = sum([self.calc_total_utility(action, state, ue_index) for ue_index in range(self.ue_total)])
        return utility
