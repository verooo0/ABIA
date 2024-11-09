import timeit
from aima.search import hill_climbing
from bin_packing_problem_opt_modified import AzamonProblem 
from bin_packing_problem_parameters_modified import AzamonParameters
from bin_packing_state_opt_modified import StateRepresentation
from abia_azamon import random_ofertas, random_paquetes
import random

def generate_initial_state(n_paquetes: int, seed: int, proporcion: float) -> StateRepresentation:
    # Generamos paquetes y ofertas aleatorios
    paquetes = random_paquetes(n_paquetes, seed)
    ofertas = random_ofertas(paquetes, proporcion, seed)

    # Convertimos la lista de paquetes y ofertas en el formato esperado para el estado
    params = AzamonParameters(
        max_weight=max(oferta.pesomax for oferta in ofertas),
        package_weights=[p.peso for p in paquetes],
        priority_packages=[p.prioridad for p in paquetes],
        offer_capacities=[o.pesomax for o in ofertas],
        days_limits=[o.dias for o in ofertas],
        price_kg= [o.precio for o in ofertas]
    )

    # Diccionario de asignaciones basado en la prioridad
    asignaciones = {}

    # Asignar paquetes a ofertas basándose en la prioridad
    for prioridad in range(3):
        for paquete in paquetes:
            if paquete.prioridad == prioridad:
                for oferta in ofertas:
                    peso_total_asignado = sum(peso['peso'] for peso in asignaciones.get(tuple(oferta.__dict__.items()), []))
                    if peso_total_asignado + paquete.peso <= oferta.pesomax:
                        if tuple(oferta.__dict__.items()) not in asignaciones:
                            asignaciones[tuple(oferta.__dict__.items())] = []
                        asignaciones[tuple(oferta.__dict__.items())].append({'peso': paquete.peso, 'prioridad': paquete.prioridad})
                        break

    # Convertir asignaciones a representación de estado
    # Creamos un vector `v_p` que indica la oferta asignada a cada paquete
    v_p = []
    for paquete in paquetes:
        for idx, oferta in enumerate(ofertas):
            # Busca la oferta asignada para este paquete
            if {'peso': paquete.peso, 'prioridad': paquete.prioridad} in asignaciones.get(tuple(oferta.__dict__.items()), []):
                v_p.append(idx)
                break
        else:
            v_p.append(0)  # Asigna a la primera oferta si no se asignó

    return StateRepresentation(params, v_p)

# Pruebas aleatorias con asignación por prioridad
for i in range(5):
    initial_state = generate_initial_state(n_paquetes=10, seed=i, proporcion=1.5)
    result = hill_climbing(AzamonProblem(initial_state, use_entropy=False))

    print(f"Prueba {i + 1}")
    print(f"Heuristic cost: {result.heuristic_cost()} | Heuristic happiness: {result.heuristic_happiness()}")
    print()

