from typing import List

class AzamonParameters(object):
    def __init__(self, max_weight: int, package_weights: List[float], priority_packages: List[int], offer_capacities: List[float], days_limits: List[int], price_kg: List[float]):
        self.max_weight = max_weight  # Maximum weight each offer can handle
        self.package_weights = package_weights  # Weight of each package
        self.priority_packages = priority_packages  #Priority of each package
        self.offer_capacities = offer_capacities  # Maximum capacities of each offer
        self.days_limits = days_limits  # Delivery days allowed for each offer
        self.price_kg = price_kg        #Price of kg for each offer


    def __repr__(self):
        return f"Params(max_weight={self.max_weight}, package_weights={self.package_weights}, priority_packages={self.priority_packages} offer_capacities={self.offer_capacities}, days_limits={self.days_limits}, price_kg={self.price_kg})"