# experiments.py

import csv
from abia_azamon import random_paquetes, random_ofertas
from bin_packing_problem_opt_modified import AzamonProblem
from bin_packing_state_opt_modified import StateRepresentation
from bin_packing_problem_parameters_modified import AzamonParameters
from aima.search import hill_climbing
import timeit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Función para generar el estado inicial aleatorio
def generate_initial_al(n_paquetes: int, seed: int, proporcion: float) -> StateRepresentation:
    paquetes = random_paquetes(n_paquetes, seed)
    ofertas = random_ofertas(paquetes, proporcion, seed)

    params = AzamonParameters(
        max_weight=max(oferta.pesomax for oferta in ofertas),
        package_weights=[p.peso for p in paquetes],
        priority_packages=[p.prioridad for p in paquetes],
        max_delivery_days_per_package=[1 if p.prioridad == 0 else 3 if p.prioridad == 1 else 5 for p in paquetes],
        offer_capacities=[o.pesomax for o in ofertas],
        days_limits=[o.dias for o in ofertas],
        price_kg=[o.precio for o in ofertas]
    )

    assignments = [0] * len(paquetes)  # Initial assignments
    return StateRepresentation(params, assignments)
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

# Experimento 1: Análisis de coste total en función de la proporción de peso
def experiment_1(): 
    pass

def experiment_2():
    for i in range(10):
        initial_state = generate_initial_state(n_paquetes=50, seed=i, proporcion=1.2)
        result = hill_climbing(AzamonProblem(initial_state))


        print(f"Prueba {i + 1}")
        print(f"Heuristic cost: {result.heuristic_cost()}")
        print()

    for i in range(10):
        initial_state = generate_initial_al(n_paquetes=50, seed=i, proporcion=1.2)
        result = hill_climbing(AzamonProblem(initial_state))


        print(f"Prueba {i + 1}")
        print(f"Heuristic cost: {result.heuristic_cost()}")
        print()

def experiment_3(): 
    pass

def experiment_4(): 
    # Define parameters for the experiments
    fixed_num_packages = 50
    fixed_proportion = 1.2
    package_increment = 10
    proportion_increment = 0.2

    # Tracking results
    time_results = []


    proportions = np.arange(1.0, 2.2, proportion_increment)  # Proportion range from 1.0 to 2.0
    for prop in proportions:
        initial_state = generate_initial_al(n_paquetes=fixed_num_packages, seed=0, proporcion=prop)
        problem = AzamonProblem(initial_state)
        
        # Measure execution time
        start_time = timeit.default_timer()
        hill_climbing(problem)
        end_time = timeit.default_timer()
        
        elapsed_time = end_time - start_time
        time_results.append((fixed_num_packages, prop, elapsed_time))
        print(f"Proportion: {prop} | Time: {elapsed_time:.4f} seconds")


    num_packages_list = range(50, 101, package_increment)  # Package range from 50 to 100
    for num_packages in num_packages_list:
        initial_state = generate_initial_al(n_paquetes=num_packages, seed=0, proporcion=fixed_proportion)
        problem = AzamonProblem(initial_state)
        
        # Measure execution time
        start_time = timeit.default_timer()
        hill_climbing(problem)
        end_time = timeit.default_timer()
        
        elapsed_time = end_time - start_time
        time_results.append((num_packages, fixed_proportion, elapsed_time))
        print(f"Packages: {num_packages} | Time: {elapsed_time:.4f} seconds")


    # Guardar resultados en un archivo CSV
    with open("time_results.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["NumPackages", "Proportion", "ExecutionTime"])  # Encabezados
        for row in time_results:
            writer.writerow(row)

    # Leer el archivo CSV
    df = pd.read_csv("time_results.csv")

    # Eliminar duplicados basados en las columnas 'NumPackages' y 'Proportion'
    df = df.drop_duplicates(subset=['NumPackages', 'Proportion'])

    max_execution_time = max(df["ExecutionTime"])

    # Graficar tiempo en función de la proporción de peso transportable (número de paquetes fijo)
    df_fixed_packages = df[df["NumPackages"] == 50]
    plt.figure(figsize=(10, 5))
    plt.plot(df_fixed_packages["Proportion"], df_fixed_packages["ExecutionTime"], marker='o')
    plt.xlabel("Proportion of Transportable Weight")
    plt.ylabel("Execution Time (s)")
    plt.title("Execution Time vs Transportable Weight Proportion (50 Packages)")
    plt.ylim(0, max_execution_time)  # Establece el mismo límite en el eje y
    plt.show()

    # Graficar tiempo en función del número de paquetes (proporción fija)
    df_fixed_proportion = df[df["Proportion"] == 1.2]
    plt.figure(figsize=(10, 5))
    plt.plot(df_fixed_proportion["NumPackages"], df_fixed_proportion["ExecutionTime"], marker='o')
    plt.xlabel("Number of Packages")
    plt.ylabel("Execution Time (s)")
    plt.title("Execution Time vs Number of Packages (Proportion 1.2)")
    plt.ylim(0, max_execution_time)  # Establece el mismo límite en el eje y
    plt.show()

def experiment_5(): 
    cost_results = []
    proportions = [1.0 + i * 0.2 for i in range(7)]  # From 1.0 to 2.0 in increments of 0.2
    num_packages = 50

    # Run experiment for each proportion
    for proportion in proportions:
        # Generate initial state with the given proportion
        initial_state = generate_initial_al(n_paquetes=num_packages, seed=0, proporcion=proportion)
        problem = AzamonProblem(initial_state)
        
        # Run hill climbing to optimize
        result = hill_climbing(problem)
        
        # Capture the total heuristic cost
        total_cost = result.heuristic_cost()
        
        # Store results
        cost_results.append((proportion, total_cost))

    # Save the results in a CSV file
    with open("cost_results.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Proportion", "TotalCost"])
        for row in cost_results:
            writer.writerow(row)
            
    # Leer los resultados desde el archivo CSV
    df = pd.read_csv("cost_results.csv")

    # Graficar el coste total en función de la proporción de peso transportable
    plt.figure(figsize=(10, 5))
    plt.plot(df["Proportion"], df["TotalCost"], marker='o', label="Total Cost")
    plt.xlabel("Proportion of Transportable Weight")
    plt.ylabel("Total Cost")
    plt.title("Total Cost vs Transportable Weight Proportion")
    plt.legend()
    plt.show()
    

