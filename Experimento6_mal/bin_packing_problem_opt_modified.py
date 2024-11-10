

from typing import Generator
from aima.search import Problem
from bin_packing_operators_modified import AzamonOperator
from bin_packing_state_opt_modified import StateRepresentation

class AzamonProblem(Problem):
    def __init__(self, initial_state: StateRepresentation, maximize_happiness: bool = False, use_entropy=False, mode_simulated_annealing: bool = False, combine_heuristic: bool = False):
        self.maximize_happiness = maximize_happiness
        self.use_entropy = use_entropy
        self.mode_simulated_annealing = mode_simulated_annealing
        self.combine_heuristic = combine_heuristic
        super().__init__(initial_state)

    def actions(self, state: StateRepresentation) -> Generator[AzamonOperator, None, None]:
        return state.generate_actions(self.mode_simulated_annealing)

    def result(self, state: StateRepresentation, action: AzamonOperator) -> StateRepresentation:
        return state.apply_action(action)

    def value(self, state: StateRepresentation) -> float:
        # if self.maximize_happiness:
        #     return state.heuristic_happiness()
        # elif self.combine_heuristic:
        #     return 0*-state.heuristic_cost() + 1*state.heuristic_happiness()
        # else:
        #     return -state.heuristic_cost()

        # Calcular los valores de coste y felicidad del estado
        cost = state.heuristic_cost()
        happiness = state.heuristic_happiness()

        # Valores máximos empíricos
        max_cost = 618.885
        max_happiness = 56

        # Normalización
        normalized_cost = cost / max_cost if max_cost > 0 else 0
        normalized_happiness = happiness / max_happiness if max_happiness > 0 else 0

        # Heurística combinada con ponderaciones
        if self.maximize_happiness:
            return normalized_happiness  # Maximizar solo la felicidad
        elif self.combine_heuristic:
            return 0.1 * -normalized_cost + 0.9 * normalized_happiness
        else:
            return -normalized_cost  # Minimizar solo el coste
            

    def goal_test(self, state: StateRepresentation) -> bool:
        return False
        #return state.is_goal()
