
from typing import List, Dict, Generator
from bin_packing_operators_modified import AzamonOperator, AssignPackage, SwapAssignments, RemovePackage, InsertPackage
from bin_packing_problem_parameters_modified import AzamonParameters
import random

class StateRepresentation(object):
    def __init__(self, params: AzamonParameters, assignments: List[int], happiness: Dict[int, int] = None, falta: List[int] = None):
        self.params = params
        self.assignments = assignments
        self.v_p = assignments
        self.happiness = happiness or {}
        self.update_happiness()
        self.falta = falta or []
        self.update_falta()

    def update_happiness(self):
        for pkg_id, offer_id in enumerate(self.assignments):
            package_priority = self.params.priority_packages[pkg_id]
            #max_delivery_days = self.params.max_delivery_days_per_package[pkg_id]
            min_delivery_days = (1 if package_priority == 0 else
                                2 if package_priority == 1 else
                                4)
            self.happiness[pkg_id] = max(0, min_delivery_days - self.params.days_limits[offer_id])

    def update_falta(self):
        self.falta = []
        for pkg_id, offer_id in enumerate(self.assignments):
            if self.assignments[pkg_id]== -1:
                self.falta.append(pkg_id)

    def copy(self):
        return StateRepresentation(self.params, self.assignments.copy(), self.happiness.copy(), self.falta.copy())

    def generate_actions(self, mode_simulated_annealing: bool = False) -> Generator[AzamonOperator, None, None]:
        total_weights_per_offer = [0.0] * len(self.params.offer_capacities)

        lista_actions_SA: list[AzamonOperator] = [] # Lista de acciones posibles para el algoritmo Simulated Annealing

        ##---AssignPackage---
        for pkg_id, offer_id in enumerate(self.assignments):
            total_weights_per_offer[offer_id] += self.params.package_weights[pkg_id]

        for pkg_id, offer_id in enumerate(self.assignments):
            max_delivery_days = self.params.max_delivery_days_per_package[pkg_id]
            
            for new_offer_id in range(len(self.params.offer_capacities)):

                weight_new_offer = total_weights_per_offer[new_offer_id] + self.params.package_weights[pkg_id]

                if (new_offer_id != offer_id and self.params.offer_capacities[new_offer_id] >= weight_new_offer 
                    and self.params.days_limits[new_offer_id] <= max_delivery_days):

                    if mode_simulated_annealing:
                        lista_actions_SA.append(AssignPackage(pkg_id, new_offer_id))
                    else:
                        yield AssignPackage(pkg_id, new_offer_id)

        ##-----SwapAssignments-----
        for pkg_id_1 in range(len(self.assignments)):
            for pkg_id_2 in range(len(self.assignments)):
                if pkg_id_1 != pkg_id_2:
                    offer_id_1 = self.assignments[pkg_id_1]
                    offer_id_2 = self.assignments[pkg_id_2]
                    if offer_id_1 != offer_id_2:

                        weight_pkg_1 = self.params.package_weights[pkg_id_1]
                        weight_pkg_2 = self.params.package_weights[pkg_id_2]

                        max_delivery_days_1 = self.params.max_delivery_days_per_package[pkg_id_1]
                        max_delivery_days_2 = self.params.max_delivery_days_per_package[pkg_id_2]

                        new_weight_offer_1 = (total_weights_per_offer[offer_id_1] - weight_pkg_1 + weight_pkg_2)
                        new_weight_offer_2 = (total_weights_per_offer[offer_id_2] - weight_pkg_2 + weight_pkg_1)

                        if (self.params.offer_capacities[offer_id_1] >= new_weight_offer_1 and 
                            self.params.offer_capacities[offer_id_2] >= new_weight_offer_2 and
                            self.params.days_limits[offer_id_1] <= max_delivery_days_2 and
                            self.params.days_limits[offer_id_2] <= max_delivery_days_1):
                            
                            if mode_simulated_annealing:
                                lista_actions_SA.append(SwapAssignments(pkg_id_1, pkg_id_2))
                            else:
                                yield SwapAssignments(pkg_id_1, pkg_id_2)

        # ##-----RemovePackage-----
        # for pkg_id, offer_id in enumerate(self.assignments):
        #     if pkg_id not in self.falta and self.assignments[pkg_id] != -1:
        #         if mode_simulated_annealing:
        #             lista_actions_SA.append(RemovePackage(pkg_id, offer_id))
        #         else:
        #             yield RemovePackage(pkg_id, offer_id)

        # ##-----InsertPackage-----
        # for pkg_id in self.falta:
        
        #     max_delivery_days = self.params.max_delivery_days_per_package[pkg_id]

        #     for new_offer_id in range(len(self.params.offer_capacities)):

        #         weight_new_offer = total_weights_per_offer[new_offer_id] + self.params.package_weights[pkg_id]

        #         if self.params.offer_capacities[new_offer_id] >= weight_new_offer and self.params.days_limits[new_offer_id] <= max_delivery_days:
        #             if mode_simulated_annealing:
        #                 lista_actions_SA.append(InsertPackage(pkg_id, new_offer_id))
        #             else:
        #                 yield InsertPackage(pkg_id, new_offer_id)

        if mode_simulated_annealing:
            # Return random action for SA
            if lista_actions_SA:
                index_action = random.randint(0, len(lista_actions_SA) - 1)
                yield lista_actions_SA[index_action]

            

    
        
    def apply_action(self, action: AzamonOperator):
        new_state = self.copy()
        if isinstance(action, AssignPackage):
            pkg_id = action.package_id
            new_offer_id = action.offer_id
            new_state.assignments[pkg_id] = new_offer_id
            new_state.update_happiness()
            new_state.update_falta()
        elif isinstance(action, SwapAssignments):
            pkg_id_1 = action.package_id_1
            pkg_id_2 = action.package_id_2
            new_state.assignments[pkg_id_1], new_state.assignments[pkg_id_2] = new_state.assignments[pkg_id_2], new_state.assignments[pkg_id_1]
            new_state.update_happiness()
            new_state.update_falta()
        elif isinstance(action, RemovePackage):
            pkg_id = action.package_id
            new_offer_id = action.offer_id
            new_state.assignments[pkg_id] = -1
            if pkg_id not in new_state.falta:
                new_state.falta.append(pkg_id)
            new_state.update_happiness()
            new_state.update_falta()
        elif isinstance(action, InsertPackage):
            pkg_id = action.package_id
            new_offer_id = action.offer_id
            new_state.assignments[pkg_id] = new_offer_id
            if pkg_id in new_state.falta:
                new_state.falta.remove(pkg_id)
            new_state.update_happiness()
            new_state.update_falta()
        return new_state


    def heuristic_cost(self, calc_penalty: bool = True) -> float:
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
                penalty+= 10000
                

        #Penalización supera peso oferta
            total_weights_per_offer[offer_id] += package_weight

        for offer_id, total_weight in enumerate(total_weights_per_offer):
            if total_weight > self.params.offer_capacities[offer_id]:
                penalty += 500
                
                
                

        self.update_falta()
        penalty += 1000*len(self.falta)
                
        #return sum(self.params.offer_capacities[offer_id] * self.params.package_weights[pkg_id] for pkg_id, offer_id in enumerate(self.assignments))
        #return sum(self.params.price_kg[offer_id] * self.params.package_weights[pkg_id] for pkg_id, offer_id in enumerate(self.assignments))
        if calc_penalty:
            return total_transport_cost + total_storage_cost + penalty
        else:
            return total_transport_cost + total_storage_cost
        
    def heuristic_happiness(self, calc_penalty: bool = True) -> float:

        self.update_happiness()
        penalty =0.0
        total_weights_per_offer = [0.0] * len(self.params.offer_capacities)
        #send_cost = sum(self.params.price_kg[offer_id] * self.params.package_weights[pkg_id] for pkg_id, offer_id in enumerate(self.assignments))
        for pkg_id, offer_id in enumerate(self.assignments):
            package_weight = self.params.package_weights[pkg_id]
            days_limit = self.params.days_limits[offer_id]
            max_delivery_days = self.params.max_delivery_days_per_package[pkg_id]

        #Penalización entrega tarde
            if days_limit > max_delivery_days:
                penalty+= 10
            
        #Penalización supera peso oferta
            total_weights_per_offer[offer_id] += package_weight

        for offer_id, total_weight in enumerate(total_weights_per_offer):
            if total_weight > self.params.offer_capacities[offer_id]:
                penalty += 10
                
        if calc_penalty:
            return sum(self.happiness.values()) - 10*len(self.falta) -penalty
        else:
            return sum(self.happiness.values())

    def heuristic_cost_happy(self) ->float:
        total_logistic_cost = 0.0
        total_happiness = 0.0
        penalty_cost = 0.0  # Penalización por violación de restricciones
        total_weights_per_offer = [0.0] * len(self.params.offer_capacities)
        # Valores máximos para normalización según las pruebas
        max_logistic_cost = 618.885
        max_happiness = 56

        # Ponderaciones
        weight_logistic = 0.7
        weight_happiness = 0.3
        #penalty_weight = 500  # Peso alto para penalizaciones de restricciones

        # Cálculo de los componentes y penalizaciones
        for pkg_id, offer_id in enumerate(self.assignments):
            package_weight = self.params.package_weights[pkg_id]
            days_limit = self.params.days_limits[offer_id]
            price_per_kg = self.params.price_kg[offer_id]
            max_delivery_days = self.params.max_delivery_days_per_package[pkg_id]

            # 1. Coste logístico (transporte + almacenamiento)
            transport_cost = price_per_kg * package_weight
            storage_cost = 0.0
            if days_limit >= 3:
                storage_days = 1 if days_limit in [3, 4] else 2
                storage_cost = storage_days * 0.25 * package_weight
            total_logistic_cost += transport_cost + storage_cost

            # 2. Felicidad (días de adelanto)
            self.update_happiness()
            total_happiness = sum(self.happiness.values())

            # 3. Restricciones: verificar capacidad y días máximos
            #Penalización entrega tarde
            if days_limit > max_delivery_days:
                penalty_cost+= 5
                #print('Penalty retraso')

        #Penalización supera peso oferta
            total_weights_per_offer[offer_id] += package_weight

        for offer_id, total_weight in enumerate(total_weights_per_offer):
            if total_weight > self.params.offer_capacities[offer_id]:
                penalty_cost += 2
                #print('Penalty peso')

        self.update_falta()
        penalty_cost += 1*len(self.falta)


        # Normalización de los costes para poder calcularlos juntos en la heurística
        normalized_logistic_cost = total_logistic_cost / max_logistic_cost
        normalized_happiness = total_happiness / max_happiness

        # Heurística combinada con penalizaciones
        heuristic_value = (weight_logistic * normalized_logistic_cost - 
                        weight_happiness * normalized_happiness + penalty_cost) #penalty_cost

        return heuristic_value

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

    def last_assigments(self):
        return self.assignments
            #return all(self.params.offer_capacities[offer_id] >= self.params.package_weights[pkg_id] for pkg_id, offer_id in enumerate(self.assignments))
        
