
from typing import Generator
from aima.search import Problem
from bin_packing_operators_modified import AzamonOperator
from bin_packing_state_opt_modified import StateRepresentation

class AzamonProblem(Problem):
    def __init__(self, initial_state: StateRepresentation, maximize_happiness: bool = False, use_entropy=False):
        self.maximize_happiness = maximize_happiness
        self.use_entropy = use_entropy
        super().__init__(initial_state)

    def actions(self, state: StateRepresentation) -> Generator[AzamonOperator, None, None]:
        return state.generate_actions()

    def result(self, state: StateRepresentation, action: AzamonOperator) -> StateRepresentation:
        return state.apply_action(action)

    def value(self, state: StateRepresentation) -> float:
        if self.maximize_happiness:
            return state.heuristic_happiness()
        else:
            return -state.heuristic_cost()

    def goal_test(self, state: StateRepresentation) -> bool:
        return state.is_goal()
