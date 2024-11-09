from typing import List, Dict, Generator
from bin_packing_operators_modified import AzamonOperator, AssignPackage, SwapAssignments, EliminatePackage
from bin_packing_problem_parameters_modified import AzamonParameters

class StateRepresentation(object):
    def __init__(self, params: AzamonParameters, assignments: List[int], happiness: Dict[int, int] = None):
        self.params = params
        self.assignments = assignments
        self.v_p = assignments
        self.happiness = happiness or {}
        self.update_happiness()

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
        total_weights_per_offer = [0.0] * len(self.params.offer_capacities)

        for pkg_id, offer_id in enumerate(self.assignments):
            total_weights_per_offer[offer_id] += self.params.package_weights[pkg_id]

        for pkg_id, offer_id in enumerate(self.assignments):
            max_delivery_days = self.params.max_delivery_days_per_package[pkg_id]
            
            for new_offer_id in range(len(self.params.offer_capacities)):

                weight_new_offer = total_weights_per_offer[new_offer_id] + self.params.package_weights[pkg_id]

                if (new_offer_id != offer_id and self.params.offer_capacities[new_offer_id] >= weight_new_offer 
                    and self.params.days_limits[new_offer_id] <= max_delivery_days):
                    
                    yield AssignPackage(pkg_id, new_offer_id)

        for pkg_id_1 in range(len(self.assignments)):
            for pkg_id_2 in range(len(self.assignments)):
                if pkg_id_1 != pkg_id_2:
                    yield SwapAssignments(pkg_id_1, pkg_id_2)

        # for pkg_id, offer_id in enumerate(self.assignments):
        #     for new_offer_id in range(len(self.params.offer_capacities)):
        #         if new_offer_id != offer_id and self.params.offer_capacities[new_offer_id] >= self.params.package_weights[pkg_id]:
        #             yield EliminatePackage(pkg_id, new_offer_id)
        
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
        return new_state

    def heuristic_cost(self) -> float:
        total_transport_cost, total_storage_cost, penalty= 0.0, 0.0, 0.0
        total_weights_per_offer = [0.0] * len(self.params.offer_capacities)
        #send_cost = sum(self.params.price_kg[offer_id] * self.params.package_weights[pkg_id] for pkg_id, offer_id in enumerate(self.assignments))
        for pkg_id, offer_id in enumerate(self.assignments):
            package_weight = self.params.package_weights[pkg_id]
            days_limit = self.params.days_limits[offer_id]
            price_per_kg = self.params.price_kg[offer_id]
            max_delivery_days = self.params.max_delivery_days_per_package[pkg_id]

            # 1. Coste de transporte
            transport_cost = price_per_kg * package_weight
            total_transport_cost += transport_cost

        # 2. Coste de almacenamiento (solo para ofertas con 3 días o más)
            if days_limit >= 3:
                if days_limit == 3 or days_limit == 4:
                    storage_days = 1  # Paquetes de 3-4 días se almacenan un día
                elif days_limit == 5:
                    storage_days = 2  # Paquetes de 5 días se almacenan dos días

            # Calculo del coste de almacenamiento basado en el peso del paquete y los días de almacenamiento
                storage_cost = storage_days * 0.25 * package_weight
                total_storage_cost += storage_cost

        #Penalización entrega tarde
            if days_limit > max_delivery_days:
                penalty+= 1000 

        #Penalización supera peso oferta
            total_weights_per_offer[offer_id] += package_weight

        for offer_id, total_weight in enumerate(total_weights_per_offer):
            if total_weight > self.params.offer_capacities[offer_id]:
                penalty += 500
                
        #return sum(self.params.offer_capacities[offer_id] * self.params.package_weights[pkg_id] for pkg_id, offer_id in enumerate(self.assignments))
        #return sum(self.params.price_kg[offer_id] * self.params.package_weights[pkg_id] for pkg_id, offer_id in enumerate(self.assignments))
        
        return total_transport_cost + total_storage_cost + penalty
    
    def heuristic_happiness(self) -> float:
        self.update_happiness()

        return sum(self.happiness.values())

    def is_goal(self) -> bool:
        
        total_weights_per_offer = [0.0] * len(self.params.offer_capacities)

    
        for pkg_id, offer_id in enumerate(self.assignments):
            package_weight = self.params.package_weights[pkg_id]
            max_delivery_days = self.params.max_delivery_days_per_package[pkg_id] 
            
            total_weights_per_offer[offer_id] += package_weight
          
            if self.params.days_limits[offer_id] > max_delivery_days:
                return False  

        
        for offer_id, total_weight in enumerate(total_weights_per_offer):
            if total_weight > self.params.offer_capacities[offer_id]:
                return False  

        
        return True 

         
        #return all(self.params.offer_capacities[offer_id] >= self.params.package_weights[pkg_id] for pkg_id, offer_id in enumerate(self.assignments))
        
