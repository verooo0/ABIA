from __future__ import annotations
from typing import List, Set, Generator
from cont_par_op import BinPackingOperator, ProblemParameters, MoveParcel, SwapParcels
from aima.search import Problem
from aima . search import hill_climbing
import timeit

class StateRepresentation(object):
    def __init__(self, params: ProblemParameters, v_c: List[Set[int]]):
        self.params = params
        self.v_c = v_c

    def copy(self) -> StateRepresentation:
        # Afegim el copy per cada set!
        v_c_copy = [set_i.copy() for set_i in self.v_c]
        return StateRepresentation(self.params, v_c_copy)

    def __repr__(self) -> str:
        return f"v_c={str(self.v_c)} | {self.params}"

    def trobar_oferta(self, p_i: int) -> int: #??
        for c_i in range(len(self.v_c)):
            if p_i in self.v_c[c_i]:
                return c_i

    def generate_actions(self) -> Generator[BinPackingOperator, None, None]:
        free_spaces = []
        for c_i, parcels in enumerate(self.v_c):
            h_c_i = self.params.h_max
            for p_i in parcels:
                h_c_i = h_c_i - self.params.v_h[p_i]
            free_spaces.append(h_c_i)
        # Recorregut contenidor per contenidor per saber quins paquets podem moure
        for c_j, parcels in enumerate(self.v_c):
            for p_i in parcels:
                for c_k in range(len(self.v_c)):
                    # Condició: contenidor diferent i té espai lliure suficient
                    if c_j != c_k and free_spaces[c_k] >= self.params.v_h[p_i]:
                        yield MoveParcel(p_i, c_j, c_k)

        # Intercanviar paquets
        for p_i in range(self.params.p_max):
            for p_j in range(self.params.p_max):
                if p_i != p_j:
                    c_i = self.find_container(p_i)
                    c_j = self.find_container(p_j)

                    if c_i != c_j:
                        h_p_i = self.params.v_h[p_i]
                        h_p_j = self.params.v_h[p_j]

                        # Condició: hi ha espai lliure suficient per fer l'intercanvi
                        # (Espai lliure del contenidor + espai que deixa el paquet >= espai del nou paquet)
                        if free_spaces[c_i] + h_p_i >= h_p_j and free_spaces[c_j] + h_p_j >= h_p_i:
                            yield SwapParcels(p_i, p_j)

    def apply_action(self, action: BinPackingOperator) -> StateRepresentation:
        new_state = self.copy()
        if isinstance(action, MoveParcel):
            p_i = action.p_i
            c_j = action.c_j
            c_k = action.c_k

            new_state.v_c[c_k].add(p_i)
            new_state.v_c[c_j].remove(p_i)

            if len(new_state.v_c[c_j]) == 0:
                del new_state.v_c[c_j]

        elif isinstance(action, SwapParcels):
            p_i = action.p_i
            p_j = action.p_j

            c_i = new_state.find_container(p_i)
            c_j = new_state.find_container(p_j)

            new_state.v_c[c_i].add(p_j)
            new_state.v_c[c_i].remove(p_i)

            new_state.v_c[c_j].add(p_i)
            new_state.v_c[c_j].remove(p_j)

        return new_state

    def heuristic(self) -> float:
        return len(self.v_c)

class BinPackingProblem(Problem):
    def __init__(self, initial_state: StateRepresentation):
        super().__init__(initial_state)

    def actions(self, state: StateRepresentation) -> Generator[BinPackingOperator, None, None]:
        return state.generate_actions()

    def result(self, state: StateRepresentation, action: BinPackingOperator) -> StateRepresentation:
        return state.apply_action(action)

    def value(self, state: StateRepresentation) -> float:
        return -state.heuristic()

    def goal_test(self, state: StateRepresentation) -> bool:
        return False
