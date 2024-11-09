from typing import List, Dict, Generator
from bin_packing_operators_modified import AzamonOperator, AssignPackage, SwapAssignments, RemovePackage
from bin_packing_problem_parameters_modified import AzamonParameters

class StateRepresentation(object):
    def __init__(self, params: AzamonParameters, assignments: List[int], happiness: Dict[int, int] = None):
        self.params = params
        self.assignments = assignments
        self.v_p = assignments
        self.happiness = happiness or {}
        self.update_happiness()
        self.falta = []

    def update_happiness(self):
        for pkg_id, offer_id in enumerate(self.assignments):
            package_priority = self.params.priority_packages[pkg_id]
            max_delivery_days = (1 if package_priority == 0 else
                             3 if package_priority == 1 else
                             5)
            self.happiness[pkg_id] = max(0, max_delivery_days - self.params.days_limits[offer_id])

    def copy(self):
        return StateRepresentation(self.params, self.assignments.copy(), self.happiness.copy())

    def generate_actions(self) -> Generator[AzamonOperator, None, None]:
        for pkg_id, offer_id in enumerate(self.assignments):
            package_priority = self.params.priority_packages[pkg_id]

            if package_priority == 0:  # Prioridad de entrega al día siguiente
                max_delivery_days = 1
            elif package_priority == 1:  # Prioridad de entrega en 2-3 días
                max_delivery_days = 3
            elif package_priority == 2:  # Prioridad de entrega en 4-5 días
                max_delivery_days = 5

            for new_offer_id in range(len(self.params.offer_capacities)):

                if new_offer_id != offer_id and self.params.offer_capacities[new_offer_id] >= self.params.package_weights[pkg_id] and self.params.days_limits[new_offer_id] <= max_delivery_days:
                    yield AssignPackage(pkg_id, new_offer_id)

        for pkg_id_1 in range(len(self.assignments)):
            for pkg_id_2 in range(len(self.assignments)):
                if pkg_id_1 != pkg_id_2 and self.assignments[pkg_id_1] != self.assignments[pkg_id_2] :
                    yield SwapAssignments(pkg_id_1, pkg_id_2)

########################################


        for pkg_id, offer_id in enumerate(self.assignments):
            if pkg_id not in self.falta:
                yield RemovePackage(pkg_id, offer_id)

            

    
        
    def apply_action(self, action: AzamonOperator):
        new_state = self.copy()
        if isinstance(action, AssignPackage):
            pkg_id = action.package_id
            new_offer_id = action.offer_id
            new_state.assignments[pkg_id] = new_offer_id
            new_state.update_happiness()
        elif isinstance(action, SwapAssignments):
            pkg_id_1 = action.package_id_1
            pkg_id_2 = action.package_id_2
            new_state.assignments[pkg_id_1], new_state.assignments[pkg_id_2] = new_state.assignments[pkg_id_2], new_state.assignments[pkg_id_1]
            new_state.update_happiness()
        elif isinstance(action, RemovePackage):
            pkg_id = action.package_id
            new_offer_id = action.offer_id
            new_state.assignments[pkg_id] = []
            self.falta.append(pkg_id)
        return new_state

    def heuristic_cost(self) -> float:
        total_transport_cost, total_storage_cost= 0.0, 0.0
        #send_cost = sum(self.params.price_kg[offer_id] * self.params.package_weights[pkg_id] for pkg_id, offer_id in enumerate(self.assignments))
        for pkg_id, offer_id in enumerate(self.assignments):

            #PROB xyz
            if self.assignments[pkg_id] != []:
                package_weight = self.params.package_weights[pkg_id]
                days_limit = self.params.days_limits[offer_id]
                price_per_kg = self.params.price_kg[offer_id]

                # 1. Coste de transporte
                transport_cost = price_per_kg * package_weight
                total_transport_cost += transport_cost

            # 2. Coste de almacenamiento (solo para ofertas con 3 días o más)
            # Si el paquete está asignado a una oferta de 3 días o más, calculamos el coste de almacenamiento.
                if days_limit >= 3:
                    if days_limit == 3 or days_limit == 4:
                        storage_days = 1  # Paquetes de 3-4 días se almacenan un día
                    elif days_limit == 5:
                        storage_days = 2  # Paquetes de 5 días se almacenan dos días

                    # Calculo del coste de almacenamiento basado en el peso del paquete y los días de almacenamiento
                    storage_cost = storage_days * 0.25 * package_weight
                    total_storage_cost += storage_cost
            #return sum(self.params.offer_capacities[offer_id] * self.params.package_weights[pkg_id] for pkg_id, offer_id in enumerate(self.assignments))
            #return sum(self.params.price_kg[offer_id] * self.params.package_weights[pkg_id] for pkg_id, offer_id in enumerate(self.assignments))
            
        return total_transport_cost + total_storage_cost
    
    def heuristic_happiness(self) -> float:
        self.update_happiness()
        
        return sum(self.happiness.values())

    def is_goal(self) -> bool:
        
        total_weights_per_offer = [0.0] * len(self.params.offer_capacities)

    
        for pkg_id, offer_id in enumerate(self.assignments):
            package_weight = self.params.package_weights[pkg_id]
            package_priority = self.params.package_priorities[pkg_id]
            
            
            total_weights_per_offer[offer_id] += package_weight

            
            max_delivery_days = (1 if package_priority == 0 else
                                3 if package_priority == 1 else
                                5)

            
            if self.params.days_limits[offer_id] > max_delivery_days:
                return False  

        
        for offer_id, total_weight in enumerate(total_weights_per_offer):
            if total_weight > self.params.offer_capacities[offer_id]:
                return False  

        
        return True

         
        #return all(self.params.offer_capacities[offer_id] >= self.params.package_weights[pkg_id] for pkg_id, offer_id in enumerate(self.assignments))
        