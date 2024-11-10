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
    for paquete in paquetes: 
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

# total_init_time = 0
# total_sim_time = 0
# total_cost = 0
# total_happiness = 0

seeds=[1234,4321,6298,4015,8603,9914,7260,3281,2119,5673]

def exp_schedule(k=2, lam=0.005, limit=1000):
    return lambda t: (k * math.exp(-lam * t) if t < limit else 0)

# for i in range(10):
#     print(f"\n=== Prueba {i + 1} ===")
    
#     start_init = time.time()
#     initial_state = generate_initial_state_2(n_paquetes=50, seed=i, proporcion=1.2)
#     init_time = time.time() - start_init
#     total_init_time += init_time
    
#     start_sim = time.time()
#     result = simulated_annealing(AzamonProblem(initial_state, use_entropy=False, mode_simulated_annealing=True), schedule=exp_schedule())
#     sim_time = time.time() - start_sim
#     total_sim_time += sim_time
    
#     cost = result.heuristic_cost()
#     happiness = result.heuristic_happiness()
#     total_cost += cost
#     total_happiness += happiness
    
#     print(f"Time: {init_time + sim_time:.2f} seconds")
#     print(f"Heuristic cost: {cost} | Heuristic happiness: {happiness} | Assignments: {result.last_assigments()}")
#     print()

# print("\n=== Averages over 10 tests ===")
# print(f"Average execution time: {(total_init_time + total_sim_time) / 10:.2f} seconds")
# print(f"Average heuristic cost: {total_cost / 10:.2f}")
# print(f"Average heuristic happiness: {total_happiness / 10:.2f}")

import plotext as plt
from termplot import plot
import csv
import os

class CostTrackingProblem:
    def __init__(self, base_problem):
        self.base_problem = base_problem
        self.costs_history = []
        
    def actions(self, state):
        return self.base_problem.actions(state)
        
    def result(self, state, action):
        new_state = self.base_problem.result(state, action)
        self.costs_history.append(new_state.heuristic_cost())
        return new_state
        
    def value(self, state):
        return self.base_problem.value(state)

def run_simulated_annealing_with_tracking():
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
        
    total_init_time = 0
    total_sa_time = 0
    total_cost = 0
    total_happiness = 0
    
    for i in range(10):
        print(f"\n=== Prueba {i + 1} ===")
        
        start_init = time.time()
        initial_state = generate_initial_state(n_paquetes=50, seed=i, proporcion=1.2)
        init_time = time.time() - start_init
        total_init_time += init_time
        
        # Create tracking problem wrapper
        tracking_problem = CostTrackingProblem(AzamonProblem(initial_state, use_entropy=False, mode_simulated_annealing=True))
        
        start_sa = time.time()
        result = simulated_annealing(
            tracking_problem,
            schedule=exp_schedule()
        )
        sa_time = time.time() - start_sa
        total_sa_time += sa_time
        
        # Get final values
        cost = result.heuristic_cost()
        happiness = result.heuristic_happiness()
        total_cost += cost
        total_happiness += happiness
        
        print(f"Time: {init_time + sa_time:.2f} seconds")
        print(f"Initial cost: {initial_state.heuristic_cost()}")
        print(f"Final cost: {cost}")
        print(f"Cost improvement: {initial_state.heuristic_cost() - cost:.2f}")
        print(f"Heuristic happiness: {happiness}")
        print(f"Number of iterations: {len(tracking_problem.costs_history)}")
        
        # 1. Plot using plotext
        plt.clf()
        plt.plot(tracking_problem.costs_history)
        plt.title(f"Cost Evolution (plotext) - Run {i+1}")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.show()
        
        # 2. Plot using termplot
        print(f"\nCost Evolution (termplot) - Run {i+1}")
        plot(tracking_problem.costs_history)
        
        # 3. Save to CSV
        csv_filename = f'results/cost_evolution_run_{i+1}.csv'
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Iteration', 'Cost'])
            for iter_num, cost in enumerate(tracking_problem.costs_history):
                writer.writerow([iter_num, cost])
        print(f"\nSaved evolution data to {csv_filename}")
        
        # Optional: Save aggregated statistics
        stats_filename = f'results/run_{i+1}_stats.csv'
        with open(stats_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Initial Cost', initial_state.heuristic_cost()])
            writer.writerow(['Final Cost', cost])
            writer.writerow(['Cost Improvement', initial_state.heuristic_cost() - cost])
            writer.writerow(['Happiness', happiness])
            writer.writerow(['Execution Time', init_time + sa_time])
            writer.writerow(['Number of Iterations', len(tracking_problem.costs_history)])
        
    # Print averages
    print("\n=== Averages over 10 tests ===")
    print(f"Average execution time: {(total_init_time + total_sa_time) / 10:.2f} seconds")
    print(f"Average heuristic cost: {total_cost / 10:.2f}")
    print(f"Average heuristic happiness: {total_happiness / 10:.2f}")
    
    # Save overall statistics
    with open('results/overall_statistics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Average Execution Time', (total_init_time + total_sa_time) / 10])
        writer.writerow(['Average Heuristic Cost', total_cost / 10])
        writer.writerow(['Average Heuristic Happiness', total_happiness / 10])

# Run the analysis
if __name__ == "__main__":
    run_simulated_annealing_with_tracking()




