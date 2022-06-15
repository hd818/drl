class UserEquipment(object):
    def __init__(self):
        self.task = []
        self.transmitting_power = 100  # The transmitting power of an UE
        self.computation_capability = 0.5  # The computation capability of an UE
        self.spectrum_price = 1  # The unit price that VEC operator charges UE per spectrum
        self.resource_price = 1  # The unit price that VEC operator charges UE per computation resource

    def generate_task(self):
        data_size = 1  # The data size of computation task for an user equipment
        resource_amount = 2500  # The number of CPU cycles of computation task for UE i
        maximum_latency = 10  # The maximal time of task execution

        self.task = [data_size, resource_amount, maximum_latency]
        return self.task
