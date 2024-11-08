
from typing import List, Dict, Generator
from bin_packing_operators_modified import AzamonOperator, AssignPackage, SwapAssignments
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
            self.happiness[pkg_id] = max(0, self.params.days_limits[offer_id] - 1)

    def copy(self):
        return StateRepresentation(self.params, self.assignments.copy(), self.happiness.copy())

    def generate_actions(self) -> Generator[AzamonOperator, None, None]:
        for pkg_id, offer_id in enumerate(self.assignments):
            for new_offer_id in range(len(self.params.offer_capacities)):
                if new_offer_id != offer_id and self.params.offer_capacities[new_offer_id] >= self.params.package_weights[pkg_id]:
                    yield AssignPackage(pkg_id, new_offer_id)
        for pkg_id_1 in range(len(self.assignments)):
            for pkg_id_2 in range(len(self.assignments)):
                if pkg_id_1 != pkg_id_2:
                    yield SwapAssignments(pkg_id_1, pkg_id_2)

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
        return sum(self.params.offer_capacities[offer_id] * self.params.package_weights[pkg_id] for pkg_id, offer_id in enumerate(self.assignments))

    def heuristic_happiness(self) -> float:
        return sum(self.happiness.values())

    def is_goal(self) -> bool:
        return all(self.params.offer_capacities[offer_id] >= self.params.package_weights[pkg_id] for pkg_id, offer_id in enumerate(self.assignments))