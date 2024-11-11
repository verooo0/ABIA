import timeit
import time
from aima.search import hill_climbing
from bin_packing_problem_opt_modified import AzamonProblem 
from bin_packing_problem_parameters_modified import AzamonParameters
from bin_packing_state_opt_modified import StateRepresentation
from abia_azamon import random_ofertas, random_paquetes
import random

#Generador con prioridad
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
    v_p = []

    for prioridad in range(3):
        for i, paquete in enumerate(paquetes):
            if paquete.prioridad == prioridad:
                asignado = False  # Indicador para verificar si el paquete ha sido asignado

                for idx, oferta in enumerate(ofertas):
                    # Calcular el peso total si añadimos este paquete a la oferta actual
                    peso_total_asignado = sum(p['peso'] for p in asignaciones.get(tuple(oferta.__dict__.items()), []))

                    # Verificar si la oferta tiene suficiente capacidad para este paquete
                    if peso_total_asignado + paquete.peso <= oferta.pesomax and oferta.dias <= (1 if paquete.prioridad == 0 else 3 if paquete.prioridad == 1 else 5):
                        v_p.append(idx)
                        # Actualizar `asignaciones` con el paquete para esta oferta
                        if tuple(oferta.__dict__.items()) not in asignaciones:
                            asignaciones[tuple(oferta.__dict__.items())] = []
                        asignaciones[tuple(oferta.__dict__.items())].append({'peso': paquete.peso, 'prioridad': paquete.prioridad})

                        # Registrar en `v_p` el índice de la oferta asignada para este paquete
                        
                        asignado = True  # Marca el paquete como asignado
                        break # Puedes cambiar 0 por otro marcador si es necesario

                if not asignado:
                    v_p.append(-1) 

    # print(v_p)
    # print()
    # print(asignaciones)

    return StateRepresentation(params, v_p)



#Generador sin prioridad
def generate_initial_state_2(n_paquetes: int, seed: int, proporcion: float) -> StateRepresentation:
    # Generador aleatorio de paquetes y ofertas
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
    
    # Asignar paquetes a ofertas basándose
    v_p = []

    for paquete in paquetes:
        asignado = False  # Indicador para verificar si el paquete se ha asignado

        for idx, oferta in enumerate(ofertas):
            # Calcula el peso total actual de la oferta incluyendo el paquete
            peso_total_asignado = sum(p['peso'] for p in asignaciones.get(tuple(oferta.__dict__.items()), []))
            
            # Verifica si la oferta tiene suficiente capacidad para este paquete
            if peso_total_asignado + paquete.peso <= oferta.pesomax:
                # Registra el índice de la oferta en `v_p`
                v_p.append(idx)

                # Actualiza `asignaciones` con el paquete para esta oferta
                if tuple(oferta.__dict__.items()) not in asignaciones:
                    asignaciones[tuple(oferta.__dict__.items())] = []
                asignaciones[tuple(oferta.__dict__.items())].append({'peso': paquete.peso, 'prioridad': paquete.prioridad})

                asignado = True  # Marca el paquete como asignado
                break  # Sale del bucle ya que el paquete se ha asignado correctamente

        # Si el paquete no se pudo asignar a ninguna oferta
        if not asignado:
            v_p.append(-1)  # Asigna a la primera oferta si no se asignó

    # print(v_p)
    # print()
    # print(asignaciones)

    return StateRepresentation(params, v_p)

# Pruebas aleatorias con asignación por prioridad

def print_final_assignments(result_state):
    print("\nFINAL ASSIGNMENTS:")
    print("-" * 45)
    print(f"{'Package ID':^10} | {'Priority':^10} | "
          f"{'Delivery Days':^15} |")
    print("-" * 45)
    
    params = result_state.params
    
    # Calculate total weight per offer
    weights_per_offer = [0.0] * len(params.offer_capacities)
    for pkg_id, offer_id in enumerate(result_state.assignments):
        weights_per_offer[offer_id] += params.package_weights[pkg_id]
    
    for pkg_id, offer_id in enumerate(result_state.assignments):
        print(f"{pkg_id:^10} | "
              f"{params.priority_packages[pkg_id]:^10} | "
              f"{params.days_limits[offer_id]:^15} | ")
    
    print("-" * 45)
    print("\nOFFER SUMMARIES:")
    for offer_id in range(len(params.offer_capacities)):
        print(f"Offer {offer_id}:")
        print(f"  Total Weight: {weights_per_offer[offer_id]:.2f} / {params.offer_capacities[offer_id]:.2f}")
        print(f"  Delivery Days: {params.days_limits[offer_id]}")


total_init_time = 0
total_hill_time = 0
total_cost = 0
total_happiness = 0

seeds=[1234,4321,6298,4015,8603,9914,7260,3281,2119,5673]

# for i in range(1):
#     print(f"\n=== Prueba {i + 1} ===")
    
#     # Measure initial state generation time
#     start_init = time.time()
#     initial_state = generate_initial_state(n_paquetes=30, seed=seeds[i], proporcion=1.2)
#     init_time = time.time() - start_init
#     print(f"Initial state generation time: {init_time:.2f} seconds")
#     total_init_time += init_time
    
#     # Measure hill climbing time
#     start_hill = time.time()
#     result = hill_climbing(AzamonProblem(initial_state, use_entropy=False))
#     hill_time = time.time() - start_hill
#     total_hill_time += hill_time
#     print(f"Hill climbing time: {hill_time:.2f} seconds")

#     # Get heuristic values
#     cost = result.heuristic_cost()
#     happiness = result.heuristic_happiness()
#     total_cost += cost
#     total_happiness += happiness

#     print(f"Time: {init_time + hill_time:.2f} seconds")
#     print(f"Heuristic cost: {result.heuristic_cost()} | Heuristic happiness: {result.heuristic_happiness()} | Assignments: {result.last_assigments()}")
#     print()

#     # Print detailed final assignments
#     print_final_assignments(result)
#     print()


# # Print averages at the end
# print("\n=== Average Times ===")
# print(f"Total execution time: {total_init_time + total_hill_time:.2f} seconds")
# print(f"Average execution time: {(total_init_time + total_hill_time) / 10:.2f} seconds")
# print(f"Average heuristic cost: {total_cost / 10:.2f}")
# print(f"Average heuristic happiness: {total_happiness / 10:.2f}")

################################################################################################

import csv
import os


class CostTrackingProblem:
    def __init__(self, base_problem, experiment_num):
        self.base_problem = base_problem
        self.experiment_num = experiment_num
        self.iteration = 0
        self.states_data = []  # Store all states data
        self.initial = base_problem.initial
        
    def actions(self, state):
        return self.base_problem.actions(state)
        
    def result(self, state, action):
        new_state = self.base_problem.result(state, action)
        
        # Store state data
        self.states_data.append({
            'iteration': self.iteration,
            'cost': new_state.heuristic_cost(),
            'happiness': new_state.heuristic_happiness(),
            'assignments': new_state.assignments.copy()
        })
        
        self.iteration += 1
        return new_state
        
    def value(self, state):
        return self.base_problem.value(state)
        
    def path_cost(self, c, state1, action, state2):
        # Delegate to base problem
        return self.base_problem.path_cost(c, state1, action, state2)

def run_hill_climbing_with_tracking():
    # Create results directory
    if not os.path.exists('results'):
        os.makedirs('results')
        
    total_init_time = 0
    total_hill_time = 0
    total_cost = 0
    total_happiness = 0

    seeds=[1234,4321,6298,4015,8603,9914,7260,3281,2119,5673]

    for i in range(10):
        print(f"\n=== Prueba {i + 1} ===")
        
        # Measure initial state generation time
        start_init = time.time()
        initial_state = generate_initial_state_2(n_paquetes=50, seed=seeds[i], proporcion=1.2)
        init_time = time.time() - start_init
        print(f"Initial state generation time: {init_time:.2f} seconds")
        total_init_time += init_time
        
        # Create tracking problem
        tracking_problem = CostTrackingProblem(AzamonProblem(initial_state, use_entropy=False, combine_heuristic=True, alpha=0.1), i)
        
        # Measure hill climbing time
        start_hill = time.time()
        result = hill_climbing(tracking_problem)
        hill_time = time.time() - start_hill
        total_hill_time += hill_time
        print(f"Hill climbing time: {hill_time:.2f} seconds")

        # Get heuristic values
        cost = result.heuristic_cost()
        happiness = result.heuristic_happiness()
        total_cost += cost
        total_happiness += happiness

        # Save tracking data to CSV
        filename = f'results/hill_climbing_experiment_{i}.csv'
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Iteration', 'Cost', 'Happiness', 'Assignments'])
            for state_data in tracking_problem.states_data:
                writer.writerow([
                    state_data['iteration'],
                    state_data['cost'],
                    state_data['happiness'],
                    state_data['assignments']
                ])

        print(f"Time: {init_time + hill_time:.2f} seconds")
        print(f"Initial cost: {initial_state.heuristic_cost()}")
        print(f"Final cost: {cost}")
        print(f"Cost improvement: {initial_state.heuristic_cost() - cost:.2f}")
        print(f"Heuristic happiness: {happiness}")
        print(f"Number of iterations: {tracking_problem.iteration}")
        print(f"Data saved to {filename}")
    
    # Print averages at the end
    print("\n=== Average Times ===")
    print(f"Total execution time: {total_init_time + total_hill_time:.2f} seconds")
    print(f"Average execution time: {(total_init_time + total_hill_time) / 10:.2f} seconds")
    print(f"Average heuristic cost: {total_cost / 10:.2f}")
    print(f"Average heuristic happiness: {total_happiness / 10:.2f}")

if __name__ == "__main__":
    run_hill_climbing_with_tracking()



