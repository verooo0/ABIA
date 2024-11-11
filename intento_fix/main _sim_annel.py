
import timeit
import time
import math
from aima.search import hill_climbing
from aima.search import simulated_annealing

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

    #####BIEN####

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

total_init_time = 0
total_sim_time = 0
total_cost = 0
total_happiness = 0
total_combine=0

seeds=[1234,4321,6298,4015,8603,9914,7260,3281,2119,5673]
lista_k=[0.001,0.05,1,10,20]
lista_lam=[0.005,0.01,0.1,0.5]

def exp_schedule(k=20, lam=0.005, limit=1000):
    return lambda t: (k * math.exp(-lam * t) if t < limit else 0)

# for lam in lista_lam: #Descomentar para hacer experimentos con diferentes valores de lam y k
#     for k in lista_k:
#         print(f"======Test k={k} i lam={lam}======")
for alpha in range(1,10):
    total_init_time = 0
    total_sim_time = 0
    total_cost = 0
    total_happiness = 0
    total_combine=0
    for e in range(5):
        print(f"\n=== Replica {e + 1} ===")
        for i in range(10):
            print(f"\n=== Prueba {i + 1} ===")
            
            start_init = time.time()
            initial_state = generate_initial_state_2(n_paquetes=50, seed=i, proporcion=1.2)
            init_time = time.time() - start_init
            total_init_time += init_time
            
            start_sim = time.time()
            result = simulated_annealing(AzamonProblem(initial_state, use_entropy=False, mode_simulated_annealing=True, combine_heuristic=True, alpha=alpha/10), schedule=exp_schedule())
            sim_time = time.time() - start_sim
            total_sim_time += sim_time
            
            cost = result.heuristic_cost()
            cost_combine = result.heuristic_cost_happy(alpha/10)
            happiness = result.heuristic_happiness()
            total_cost += cost
            total_happiness += happiness
            total_combine += cost_combine
            
            print(f"Time: {init_time + sim_time:.2f} seconds")
            print(f"Heuristic cost: {cost} | Heuristic happiness: {happiness} | Assignments: {result.last_assigments()}")
            print()

    print("\n=== Averages over 50 tests ===")
    print(f'Alpha{alpha/10}')
    print(f"Total time: {total_init_time + total_sim_time}")
    print(f"Average execution time: {(total_init_time + total_sim_time) / 50:.2f} seconds")
    print(f"Average heuristic cost: {total_cost / 50:.2f}")
    print(f'Avarage heuristic combine:{total_combine/50:.2f}')
    print(f"Average heuristic happiness: {total_happiness / 50:.2f}")

#### PARA GUARDAR EN CSV LOS DATOS DE LA EVOLUCION DEL COSTE POR CADA CAMBIO EN EL ESTADO

# import csv
# import os

# class CostTrackingProblem:
#     def __init__(self, base_problem, experiment_num):
#         self.base_problem = base_problem
#         self.experiment_num = experiment_num
#         self.iteration = 0
#         self.states_data = []  # Store all states data
#         self.initial = base_problem.initial
        
#     def actions(self, state):
#         return self.base_problem.actions(state)
        
#     def result(self, state, action):
#         new_state = self.base_problem.result(state, action)
        
#         # Store state data
#         self.states_data.append({
#             'iteration': self.iteration,
#             'cost': new_state.heuristic_cost(),
#             'happiness': new_state.heuristic_happiness()
#         })
        
#         self.iteration += 1
#         return new_state
        
#     def value(self, state):
#         return self.base_problem.value(state)
        
#     def path_cost(self, c, state1, action, state2):
#         return self.base_problem.path_cost(c, state1, action, state2)

# def run_simulated_annealing_with_tracking():
#     # Create results directory if it doesn't exist
#     if not os.path.exists('results'):
#         os.makedirs('results')
        
#     for i in range(10):
#         print(f"\n=== Prueba {i + 1} ===")
        
#         start_init = time.time()
#         initial_state = generate_initial_state(n_paquetes=50, seed=i, proporcion=1.2)
#         init_time = time.time() - start_init
        
#         tracking_problem = CostTrackingProblem(AzamonProblem(initial_state, use_entropy=False, mode_simulated_annealing=True), i)
        
#         start_sa = time.time()
#         result = simulated_annealing(
#             tracking_problem,
#             schedule=exp_schedule()
#         )
#         sa_time = time.time() - start_sa
        
#         # Save all states data for this experiment
#         filename = f'results/experiment_{i}_states.csv'
#         with open(filename, 'w', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(['Iteration', 'Cost', 'Happiness'])
#             for state_data in tracking_problem.states_data:
#                 writer.writerow([
#                     state_data['iteration'],
#                     state_data['cost'],
#                     state_data['happiness']
#                 ])
        
#         print(f"Time: {init_time + sa_time:.2f} seconds")
#         print(f"Initial cost: {initial_state.heuristic_cost()}")
#         print(f"Final cost: {result.heuristic_cost()}")
#         print(f"Cost improvement: {initial_state.heuristic_cost() - result.heuristic_cost():.2f}")
#         print(f"Heuristic happiness: {result.heuristic_happiness()}")
#         print(f"Number of iterations: {tracking_problem.iteration}")
#         print(f"Data saved to {filename}")

# if __name__ == "__main__":
#     run_simulated_annealing_with_tracking()


