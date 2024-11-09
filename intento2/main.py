import timeit
import time
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
        max_delivery_days_per_package= [1 if p.prioridad == 0 else 3 if p.prioridad == 1 else 5
        for p in paquetes],
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

total_init_time = 0
total_hill_time = 0
total_cost = 0
total_happiness = 0

seeds=[1234,4321,6298,4015,8603,9914,7260,3281,2119,5673]

for i in range(10):
    print(f"\n=== Prueba {i + 1} ===")
    
    # Measure initial state generation time
    start_init = time.time()
    initial_state = generate_initial_state(n_paquetes=50, seed=seeds[i], proporcion=1.2)
    init_time = time.time() - start_init
    print(f"Initial state generation time: {init_time:.2f} seconds")
    total_init_time += init_time
    
    # Measure hill climbing time
    start_hill = time.time()
    result = hill_climbing(AzamonProblem(initial_state, use_entropy=False))
    hill_time = time.time() - start_hill
    total_hill_time += hill_time
    print(f"Hill climbing time: {hill_time:.2f} seconds")

    # Get heuristic values
    cost = result.heuristic_cost()
    happiness = result.heuristic_happiness()
    total_cost += cost
    total_happiness += happiness

    print(f"Time: {init_time + hill_time:.2f} seconds")
    print(f"Heuristic cost: {result.heuristic_cost()} | Heuristic happiness: {result.heuristic_happiness()} | Assignments: {result.last_assigments()}")
    print()

# Print averages at the end
print("\n=== Average Times ===")
print(f"Total execution time: {total_init_time + total_hill_time:.2f} seconds")
print(f"Average execution time: {(total_init_time + total_hill_time) / 10:.2f} seconds")
print(f"Average heuristic cost: {total_cost / 10:.2f}")
print(f"Average heuristic happiness: {total_happiness / 10:.2f}")

