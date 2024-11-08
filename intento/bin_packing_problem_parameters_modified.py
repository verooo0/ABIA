
from typing import List

class AzamonParameters(object):
    def __init__(self, max_weight: int, package_weights: List[float], offer_capacities: List[float], days_limits: List[int]):
        self.max_weight = max_weight  # Maximum weight each offer can handle
        self.package_weights = package_weights  # Weight of each package
        self.offer_capacities = offer_capacities  # Maximum capacities of each offer
        self.days_limits = days_limits  # Delivery days allowed for each offer

    def __repr__(self):
        return f"Params(max_weight={self.max_weight}, package_weights={self.package_weights}, offer_capacities={self.offer_capacities}, days_limits={self.days_limits})"
