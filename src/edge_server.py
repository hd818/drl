import numpy as np


class EdgeServer(object):
    def __init__(self, user_equipments):
        self.user_equipments = user_equipments

        self.computation_capability = 0  # The computation capability of an ES
        self.bandwidth = 0  # The spectrum bandwidth between UEs and ES
        self.spectrum_price = None  # Unit price of leasing spectrum from ES to UEs
        self.resource_price = None  # Unit price of leasing computation resource from ES to UEs

        self.channel_gain = None


class VehicularEdgeServer(EdgeServer):
    def __init__(self, user_equipments):
        super().__init__(user_equipments)

        ue_total = len(user_equipments)

        self.computation_capability = 20  # The computation capability of an VES
        self.bandwidth = 5  # The spectrum bandwidth between UEs and VES
        self.spectrum_price = np.full(ue_total, 1)  # Unit price of leasing spectrum from VES to UEs
        self.resource_price = np.full(ue_total, 1)  # Unit price of leasing computation resource from VES to UEs

        self.channel_gain = np.full(ue_total, 1)


class FixedEdgeServer(EdgeServer):
    def __init__(self, user_equipments):
        super().__init__(user_equipments)

        ue_total = len(user_equipments)

        self.computation_capability = 100  # The computation capability of an FES
        self.bandwidth = 10  # The spectrum bandwidth between UEs and FES
        self.spectrum_price = np.full(ue_total, 1)  # Unit price of leasing spectrum from FES to UEs
        self.resource_price = np.full(ue_total, 1)  # Unit price of leasing computation resource from FES to UEs

        self.channel_gain = np.full(ue_total, 1)

