from typing import List

class AzamonParameters(object):
    def __init__(self, max_weight: int, package_weights: List[float], priority_packages: List[int], max_delivery_days_per_package: List[int], offer_capacities: List[float], 
                days_limits: List[int], price_kg: List[float]):
        
        self.max_weight = max_weight  # Peso máximo de cada oferta 
        self.package_weights = package_weights  # Peso de cada paquete
        self.priority_packages = priority_packages  # Prioridad de cada paquete
        self.max_delivery_days_per_package = max_delivery_days_per_package  # Días máximos para entregar un paquete
        self.offer_capacities = offer_capacities  # Capacidad máxima de cada oferta
        self.days_limits = days_limits  # Días permitidos para la entrega en cada oferta
        self.price_kg = price_kg  # Precio por kg para cada oferta



    def __repr__(self):
        return f"Params(max_weight={self.max_weight}, package_weights={self.package_weights}, priority_packages={self.priority_packages}, max_delivery_days_per_package={self.max_delivery_days_per_package} ,offer_capacities={self.offer_capacities}, days_limits={self.days_limits}, price_kg={self.price_kg})"