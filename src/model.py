from math import log2
import numpy as np


class Model(object):
    def __init__(self, UEs, VESs, FESs):
        self.UEs = UEs
        self.VESs = VESs
        self.FESs = FESs

        self.ue_total = len(self.UEs)
        self.ves_total = len(self.VESs)
        self.fes_total = len(self.FESs)


class CommunicationModel(Model):
    def __init__(self, UEs, VESs, FESs):
        super().__init__(UEs, VESs, FESs)

        # Matrix of spectrum efficiency for the communication link between VESs and UEs
        self.ves_spectrum_efficiency = np.full((self.ves_total, self.ue_total), 0.0)

        # Matrix of spectrum efficiency for the communication link between FESs and UEs
        self.fes_spectrum_efficiency = np.full((self.fes_total, self.ue_total), 0.0)

    def calc_spectrum_efficiency(self, ue_index, es_index, es_type):
        """
        Calculate the spectrum efficiency for the communication link between an ES and an UE
        :param ue_index: Index of user equipment
        :param es_index: Index of edge server
        :param es_type: Type of edge server (0: VES, 1: FES)
        :return: The spectrum efficiency
        """
        if es_type == 0:
            noise = -100  # Power of additive white Gaussian noise in the VES communication case

            # List of VES indexes except es_index
            fes_indexes = list(range(self.ves_total))
            fes_indexes.pop(es_index)

            numerator = self.UEs[ue_index].transmitting_power * self.VESs[es_index].channel_gain[ue_index]
            denominator = sum([self.UEs[j].transmitting_power * self.VESs[l].channel_gain[j]
                               for j in range(self.ue_total) for l in fes_indexes]) + noise

            spectrum_efficiency = log2(1 + numerator / denominator)
            return spectrum_efficiency
        elif es_type == 1:
            noise = -100  # Power of additive white Gaussian noise in the FES communication case

            # List of FES indexes except es_index
            fes_indexes = list(range(self.fes_total))
            fes_indexes.pop(es_index)

            numerator = self.UEs[ue_index].transmitting_power * self.FESs[es_index].channel_gain[ue_index]
            denominator = sum([self.UEs[j].transmitting_power * self.FESs[l].channel_gain[j]
                               for j in range(self.ue_total) for l in fes_indexes]) + noise

            spectrum_efficiency = log2(1 + numerator / denominator)
            return spectrum_efficiency

    def calc_data_rate(self, action, ue_index, es_index, es_type):
        """
        Calculate the data rate of an UE served by an edge server
        :param action: Action taken by agent in VEC operator
        :param ue_index: Index of user equipment
        :param es_index: Index of edge server
        :param es_type: Type of edge server (0: VES, 1: FES)
        :return: The data rate of UE i served by an edge server
        """
        if es_type == 0:
            if self.ves_spectrum_efficiency[es_index][ue_index] == 0:
                self.build_matrix()

            data_rate = action["ves_spectrum_rate"][ue_index][es_index] * self.VESs[es_index].bandwidth * \
                        self.ves_spectrum_efficiency[es_index][ue_index]
            return data_rate
        elif es_type == 1:
            if self.fes_spectrum_efficiency[es_index][ue_index] == 0:
                self.build_matrix()

            data_rate = action["fes_spectrum_rate"][ue_index][es_index] * self.FESs[es_index].bandwidth * \
                        self.fes_spectrum_efficiency[es_index][ue_index]
            return data_rate

    def build_matrix(self):
        for ue_index in range(self.ue_total):
            for ves_index in range(self.ves_total):
                self.ves_spectrum_efficiency[ves_index][ue_index] = \
                    self.calc_spectrum_efficiency(ue_index, ves_index, 0)

            for fes_index in range(self.fes_total):
                self.fes_spectrum_efficiency[fes_index][ue_index] = \
                    self.calc_spectrum_efficiency(ue_index, fes_index, 1)

    def get_state(self, action):
        # Create a state vector for all UEs
        state = {"ves_count": np.full(self.ue_total, 0),
                 "ves_data_rate": np.full((self.ue_total, self.ves_total), 0.0),
                 "ves_resource": np.full((self.ue_total, self.ves_total), 0.0),
                 "fes_data_rate": np.full((self.ue_total, self.fes_total), 0.0),
                 "fes_resource": np.full((self.ue_total, self.fes_total), 0.0)}

        # Computation capability of all VESs and FESs
        ves_capability = np.array([ves.computation_capability for ves in self.VESs])
        fes_capability = np.array([fes.computation_capability for fes in self.FESs])

        for ue_index in range(self.ue_total):
            state["ves_count"][ue_index] = self.ves_total
            state["ves_data_rate"][ue_index] = (action["transition_state"][ue_index] == 1) * \
                                               np.array([self.calc_data_rate(action, ue_index, es_index, 0)
                                                         for es_index in range(self.ves_total)])
            state["ves_resource"][ue_index] = (action["transition_state"][ue_index] == 1) * \
                                              np.multiply(action["ves_resource_rate"][ue_index], ves_capability)
            state["fes_data_rate"][ue_index] = (action["transition_state"][ue_index] == 2) * \
                                               np.array([self.calc_data_rate(action, ue_index, es_index, 1)
                                                         for es_index in range(self.fes_total)])
            state["fes_resource"][ue_index] = (action["transition_state"][ue_index] == 2) * \
                                              np.multiply(action["fes_resource_rate"][ue_index], fes_capability)
        return state


class ComputationModel(Model):
    def __init__(self, UEs, VESs, FESs):
        super().__init__(UEs, VESs, FESs)

        # Computation task from UEs
        self.tasks = [ue.generate_task() for ue in self.UEs]

    def calc_local_time(self, ue_index):
        time = self.tasks[ue_index][1] / self.UEs[ue_index].computation_capability
        return time

    def calc_communication_time(self, state, ue_index, es_index, es_type):
        """
        Calculate the communication time costs for task of an UE by an ES
        :param state: State of environment after an action
        :param ue_index: Index of user equipment
        :param es_index: Index of edge server
        :param es_type: Type of edge server (0: VES, 1: FES)
        :return: The communication time
        """
        if es_type == 0:
            time = self.tasks[ue_index][0] / state["ves_data_rate"][ue_index][es_index]
            return time
        elif es_type == 1:
            time = self.tasks[ue_index][0] / state["fes_data_rate"][ue_index][es_index]
            return time

    def calc_computation_time(self, state, ue_index, es_index, es_type):
        """
        Calculate the computation time costs for task of an UE by an ES
        :param state: State of environment after an action
        :param ue_index: Index of user equipment
        :param es_index: Index of edge server
        :param es_type: Type of edge server (0: VES, 1: FES)
        :return: The computation time
        """
        if es_type == 0:
            time = self.tasks[ue_index][1] / state["ves_resource"][ue_index][es_index]
            return time
        elif es_type == 1:
            time = self.tasks[ue_index][1] / state["fes_resource"][ue_index][es_index]
            return time

    def calc_execution_time(self, state, ue_index, es_index, es_type):
        """
        Calculate the total execution time costs for task of an UE by an ES
        :param state: State of environment after an action
        :param ue_index: Index of user equipment
        :param es_index: Index of edge server
        :param es_type: Type of edge server (0: VES, 1: FES)
        :return: The total execution time
        """
        time = self.calc_communication_time(state, ue_index, es_index, es_type) + \
               self.calc_computation_time(state, ue_index, es_index, es_type)
        return time
