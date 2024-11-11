from abia_azamon import random_paquetes, random_ofertas
from azamon_problem_opt import AzamonProblem
from azamon_state_opt import StateRepresentation
from azamon_problem_parameters import AzamonParameters
from aima.search import hill_climbing, simulated_annealing
import math
import csv
import time
import timeit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

def generate_initial_state(num_package: int, seed: int, proportion: float) -> StateRepresentation: #Generación de estado inicial basado en prioridad de entrega 
    # Generamos paquetes y ofertas aleatorios
    packages = random_paquetes(num_package, seed)
    offers = random_ofertas(packages, proportion, seed)

    # Convertimos la lista de paquetes y ofertas en parámetros
    params = AzamonParameters(
        max_weight=max(oferta.pesomax for oferta in offers),
        package_weights=[p.peso for p in packages],
        priority_packages=[p.prioridad for p in packages],
        max_delivery_days_per_package= [1 if p.prioridad == 0 else 3 if p.prioridad == 1 else 5
        for p in packages],
        offer_capacities=[o.pesomax for o in offers],
        days_limits=[o.dias for o in offers],
        price_kg= [o.precio for o in offers]
    )
    # Diccionario de asignaciones de paquetes a ofertas
    asignaciones = {}
    v_p = []

    for prioridad in range(3): #Ordenamos por prioridad de entrega (0,1,2)
        for i, paquete in enumerate(packages):
            if paquete.prioridad == prioridad:
                asignado = False  # Indicador para verificar si el paquete ha sido asignado
                for idx, oferta in enumerate(offers):
                    peso_total_asignado = sum(p['peso'] for p in asignaciones.get(tuple(oferta.__dict__.items()), [])) #Vamos sumando el peso de los paquetes asignados 
                    # Verificamos que la oferta cumple con las retricciones antes de asignar el paquete
                    if peso_total_asignado + paquete.peso <= oferta.pesomax and oferta.dias <= (1 if paquete.prioridad == 0 else 3 if paquete.prioridad == 1 else 5):
                        v_p.append(idx)
                        if tuple(oferta.__dict__.items()) not in asignaciones:
                            asignaciones[tuple(oferta.__dict__.items())] = []
                        asignaciones[tuple(oferta.__dict__.items())].append({'peso': paquete.peso, 'prioridad': paquete.prioridad})
                        asignado = True 
                        break 
                    #Guardamos en v_p el indice de la oferta y actualizamos asignaciones 
                if not asignado:
                    v_p.append(-1) 
                #Si el paquete no esta asignado se marca como -1

    return StateRepresentation(params, v_p)


def generate_initial_state_2(num_package: int, seed: int, proportion: float) -> StateRepresentation: #Generador sin prioridad
    packages = random_paquetes(num_package, seed)
    offers = random_ofertas(packages, proportion, seed)

    # Convertimos la lista de paquetes y ofertas en parámetros
    params = AzamonParameters(
        max_weight=max(oferta.pesomax for oferta in offers),
        package_weights=[p.peso for p in packages],
        priority_packages=[p.prioridad for p in packages],
        max_delivery_days_per_package= [1 if p.prioridad == 0 else 3 if p.prioridad == 1 else 5
        for p in packages],
        offer_capacities=[o.pesomax for o in offers],
        days_limits=[o.dias for o in offers],
        price_kg= [o.precio for o in offers]
    )
    asignaciones = {}
    v_p = []

    for paquete in packages:
        asignado = False

        for idx, oferta in enumerate(offers):
            peso_total_asignado = sum(p['peso'] for p in asignaciones.get(tuple(oferta.__dict__.items()), []))
            if peso_total_asignado + paquete.peso <= oferta.pesomax:
                v_p.append(idx)
                if tuple(oferta.__dict__.items()) not in asignaciones:
                    asignaciones[tuple(oferta.__dict__.items())] = []
                asignaciones[tuple(oferta.__dict__.items())].append({'peso': paquete.peso, 'prioridad': paquete.prioridad})
                asignado = True 
                break

        if not asignado:
            v_p.append(-1)

    return StateRepresentation(params, v_p)


def experiment_1(): 
    seeds=[1234,4321,6298,4015,8603,9914,7260,3281,2119,5673]
    op_combination = list(product([True, False], repeat=3)) #Cogemos todas las combinaciones de True y False posibles para los 3 operadores 
    for op1, op2, op3 in op_combination:
        print(f"\n=== Testing combination: op1={op1}, op2={op2}, op3={op3} ===")
        total_init_time = 0
        total_hill_time = 0
        total_cost = 0
        total_happiness = 0

        for i in range(10):
            # Tiempo de generación del estado inicial
            start_init = time.time()
            initial_state = generate_initial_state(num_package=50, seed=seeds[i], proportion=1.2)
            init_time = time.time() - start_init
            total_init_time += init_time

            # Tiempo de hill climbing usando la combinación actual de operadores
            start_hill = time.time()
            problem_instance = AzamonProblem(initial_state, op1=op1, op2=op2, op3=op3)
            result = hill_climbing(problem_instance)
            hill_time = time.time() - start_hill
            total_hill_time += hill_time

            #Heurística de coste y felicidad
            cost = result.heuristic_cost()
            happiness = result.heuristic_happiness()
            total_cost += cost
            total_happiness += happiness

        # Media de las 10 repeticiones 
        print("\n=== Average Times ===")
        print(f"Total execution time: {total_init_time + total_hill_time:.2f} seconds")
        print(f"Average execution time: {(total_init_time + total_hill_time) / 10:.2f} seconds")
        print(f"Average heuristic cost: {total_cost / 10:.2f}")
        print(f"Average heuristic happiness: {total_happiness / 10:.2f}")

def experiment_2():
    seeds=[1234,4321,6298,4015,8603,9914,7260,3281,2119,5673]

    #Generador de estado inicial con prioridad
    total_init_time = 0
    total_hill_time = 0
    total_cost = 0
    total_happiness = 0
    
    for i in range(10):
        print(f"\n=== Prueba {i + 1} ===")
        # Tiempo de generación del estado inicial
        start_init = time.time()
        initial_state = generate_initial_state(num_package=50, seed=seeds[i], proportion=1.2) 
        init_time = time.time() - start_init
        print(f"Initial state generation time: {init_time:.2f} seconds")
        total_init_time += init_time
        problem = AzamonProblem(initial_state)
        
        # Tiempo de hill climbing
        start_hill = time.time()
        result = hill_climbing(problem)
        hill_time = time.time() - start_hill
        total_hill_time += hill_time
        print(f"Hill climbing time: {hill_time:.2f} seconds")

        #Valores de la heurística 
        cost = result.heuristic_cost()
        happiness = result.heuristic_happiness()
        total_cost += cost
        total_happiness += happiness

        print(f"Time: {init_time + hill_time:.2f} seconds")
        print(f"Initial cost: {initial_state.heuristic_cost()}")
        print(f"Final cost: {cost}")
        print(f"Cost improvement: {initial_state.heuristic_cost() - cost:.2f}")
        print(f"Heuristic happiness: {happiness}")

    print("\n=== Average Times ===")
    print(f"Total execution time: {total_init_time + total_hill_time:.2f} seconds")
    print(f"Average execution time: {(total_init_time + total_hill_time) / 10:.2f} seconds")
    print(f"Average heuristic cost: {total_cost / 10:.2f}")
    print(f"Average heuristic happiness: {total_happiness / 10:.2f}")

    #Generador inicial sin prioridad
    total_init_time_2 = 0
    total_hill_time_2 = 0
    total_cost_2 = 0
    total_happiness_2 = 0
    for i in range(10):
        print(f"\n=== Prueba {i + 1} ===")
        
        #Tiempo de generación del estado inicial
        start_init_2 = time.time()
        initial_state_2 = generate_initial_state_2(num_package=50, seed=seeds[i], proportion=1.2) #cambiar 
        init_time_2 = time.time() - start_init_2
        print(f"Initial state generation time: {init_time_2:.2f} seconds")
        total_init_time_2 += init_time_2
        
        problem_2 = AzamonProblem(initial_state_2)
        
        # Measure hill climbing time
        start_hill_2 = time.time()
        result_2 = hill_climbing(problem_2)
        hill_time_2 = time.time() - start_hill_2
        total_hill_time_2 += hill_time_2
        print(f"Hill climbing time: {hill_time_2:.2f} seconds")

        #Valores de la heurística 
        cost_2 = result_2.heuristic_cost()
        happiness_2 = result_2.heuristic_happiness()
        total_cost_2 += cost_2
        total_happiness_2 += happiness_2

        print(f"Time: {init_time_2 + hill_time_2:.2f} seconds")
        print(f"Initial cost: {initial_state_2.heuristic_cost()}")
        print(f"Final cost: {cost_2}")
        print(f"Cost improvement: {initial_state_2.heuristic_cost() - cost:.2f}")
        print(f"Heuristic happiness: {happiness_2}")

    print("\n=== Average Times ===")
    print(f"Total execution time: {total_init_time_2 + total_hill_time_2:.2f} seconds")
    print(f"Average execution time: {(total_init_time_2 + total_hill_time_2) / 10:.2f} seconds")
    print(f"Average heuristic cost: {total_cost_2 / 10:.2f}")
    print(f"Average heuristic happiness: {total_happiness_2 / 10:.2f}")

def experiment_3(): 
    total_init_time = 0
    total_sim_time = 0
    total_cost = 0
    total_happiness = 0

    seeds=[1234,4321,6298,4015,8603,9914,7260,3281,2119,5673]
    lista_k=[0.001,0.05,1,10,20]
    lista_lam=[0.001,0.005,0.01,0.1,0.5]

    def exp_schedule(k=0.05, lam=0.005, limit=1000):
        return lambda t: (k * math.exp(-lam * t) if t < limit else 0) #Función del simulated annealing
    for lam in lista_lam:
         for k in lista_k:
            print(f"======Test k={k} i lam={lam}======")  

            for e in range(5):  #5 replicas de 10 repeticiones por valores de k y lambda
                print(f"\n=== Replica {e + 1} ===")
                for i in range(10):
                    print(f"\n=== Prueba {i + 1} ===")
                    
                    start_init = time.time()
                    initial_state = generate_initial_state(num_package=50, seed=seeds[i], proportion=1.2)
                    init_time = time.time() - start_init
                    total_init_time += init_time
                    
                    start_sim = time.time()
                    result = simulated_annealing(AzamonProblem(initial_state, mode_simulated_annealing=True), schedule=exp_schedule())
                    sim_time = time.time() - start_sim
                    total_sim_time += sim_time
                    
                    cost = result.heuristic_cost()
                    happiness = result.heuristic_happiness()
                    total_cost += cost
                    total_happiness += happiness
                    
                    print(f"Time: {init_time + sim_time:.2f} seconds")
                    print(f"Heuristic cost: {cost} | Heuristic happiness: {happiness} | Assignments: {result.last_assigments()}")
                    print()

    print("\n=== Averages over 10 tests ===")
    print(f"Total time: {total_init_time + total_sim_time}")
    print(f"Average execution time: {(total_init_time + total_sim_time) / 10:.2f} seconds")
    print(f"Average heuristic cost: {total_cost / 10:.2f}")
    print(f"Average heuristic happiness: {total_happiness / 10:.2f}")



def experiment_4(): 
    fixed_num_packages = 50
    fixed_proportion = 1.2
    package_increment = 10
    proportion_increment = 0.2
    num_repeats = 10 #Númeor de réplicas

    time_results = []

    proportions = np.arange(1.0, 2.0 + proportion_increment, proportion_increment) #Particiones de 0.2 en 0.2 de 1 a 2 para la proporción
    for seed in range(num_repeats):
        for prop in proportions:
            initial_state = generate_initial_state(num_package=fixed_num_packages, seed=seed, proportion=prop)
            problem = AzamonProblem(initial_state)

            start_time = timeit.default_timer()
            hill_climbing(problem)
            end_time = timeit.default_timer()

            elapsed_time = end_time - start_time
            time_results.append((fixed_num_packages, prop, elapsed_time, seed))
            print(f"Seed: {seed} | Proportion: {prop} | Time: {elapsed_time:.4f} seconds")

    num_packages_list = range(50, 101, package_increment)
    
    for seed in range(num_repeats):
        for num_packages in num_packages_list:
            initial_state = generate_initial_state(num_package=num_packages, seed=seed, proportion=fixed_proportion)
            problem = AzamonProblem(initial_state)

            start_time = timeit.default_timer()
            hill_climbing(problem)
            end_time = timeit.default_timer()

            elapsed_time = end_time - start_time
            time_results.append((num_packages, fixed_proportion, elapsed_time, seed))
            print(f"Seed: {seed} | Packages: {num_packages} | Time: {elapsed_time:.4f} seconds")

    with open("time_results.csv", mode="w", newline="") as file:  #Guardamos los resultados del tiempo en un CSV
        writer = csv.writer(file)
        writer.writerow(["NumPackages", "Proportion", "ExecutionTime", "Seed"])
        for row in time_results:
            writer.writerow(row)

    df = pd.read_csv("time_results.csv")  #Leemos los valores del CSV para crear las graficas 

    #Media de tiempos para cada proporción variable y cada número de paquetes variable
    mean_execution_proportion = df[df["NumPackages"] == fixed_num_packages].groupby("Proportion")["ExecutionTime"].mean().reset_index()
    mean_execution_packages = df[df["Proportion"] == fixed_proportion].groupby("NumPackages")["ExecutionTime"].mean().reset_index()

    #Igualamos la escala de las dos gráficas 
    y_min = min(mean_execution_proportion["ExecutionTime"].min(), mean_execution_packages["ExecutionTime"].min())
    y_max = max(mean_execution_proportion["ExecutionTime"].max(), mean_execution_packages["ExecutionTime"].max())

    # Primer gráfico: Tiempo medio de ejecución vs Proporción
    plt.figure(figsize=(10, 5))
    plt.plot(mean_execution_proportion["Proportion"], mean_execution_proportion["ExecutionTime"], marker='o')
    plt.xlabel("Proportion of Transportable Weight")
    plt.ylabel("Average Execution Time (s)")
    plt.title("Average Execution Time vs Transportable Weight Proportion (50 Packages)")
    plt.ylim(y_min, y_max)  # Fijamos el mismo rango para el eje y
    plt.show()

    # Segundo gráfico: Tiempo medio de ejecución vs Número de paquetes
    plt.figure(figsize=(10, 5))
    plt.plot(mean_execution_packages["NumPackages"], mean_execution_packages["ExecutionTime"], marker='o')
    plt.xlabel("Number of Packages")
    plt.ylabel("Average Execution Time (s)")
    plt.title("Average Execution Time vs Number of Packages (Proportion 1.2)")
    plt.ylim(y_min, y_max)  # Fijamos el mismo rango para el eje y
    plt.show()


def experiment_5(): 
    cost_results = []
    proportions = [1.0 + i * 0.2 for i in range(6)]
    num_packages = 50

    for proportion in proportions:
        initial_state = generate_initial_state(num_package=num_packages, seed=0, proportion=proportion)
        problem = AzamonProblem(initial_state)
        result = hill_climbing(problem)
        total_cost = result.heuristic_cost()
        cost_results.append((proportion, total_cost))

    with open("cost_results.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Proportion", "TotalCost"])
        for row in cost_results:
            writer.writerow(row)
            
    df = pd.read_csv("cost_results.csv")

    plt.figure(figsize=(10, 5))
    plt.plot(df["Proportion"], df["TotalCost"], marker='o', label="Total Cost")
    plt.xlabel("Proportion of Transportable Weight")
    plt.ylabel("Total Cost")
    plt.title("Total Cost vs Transportable Weight Proportion")
    plt.legend()
    plt.show()
    
def experiment_6(): 
    alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    n_paquetes = 50
    proporcion = 1.2
    cost_data = {alpha: [] for alpha in alpha_values}
    combined_cost_data = {alpha: [] for alpha in alpha_values}
    execution_time_data = {alpha: [] for alpha in alpha_values}

    for alpha in alpha_values:
        print(f"\n=== Alpha: {alpha} ===")
        costs = []
        combined_costs = []
        execution_times = []

        for seed in range(10):
            initial_state = generate_initial_state(n_paquetes, seed, proporcion)
            problem = AzamonProblem(initial_state=initial_state, combine_heuristic=True, alpha=alpha)

            start_time = time.time()
            result_state = hill_climbing(problem)
            end_time = time.time()
            execution_time = end_time - start_time
            cost_only = result_state.heuristic_cost()
            combined_cost = result_state.heuristic_cost_happy(alpha=alpha)

            costs.append(cost_only)
            combined_costs.append(combined_cost)
            execution_times.append(execution_time)
        
            print(f"Seed: {seed}, Cost: {cost_only}, Combined Cost: {combined_cost}, Execution Time: {execution_time:.2f} seconds")
        cost_data[alpha] = np.mean(costs)
        combined_cost_data[alpha] = np.mean(combined_costs)
        execution_time_data[alpha] = np.mean(execution_times)
    
    alphas = list(cost_data.keys())
    mean_costs = list(cost_data.values())
    mean_combined_costs = list(combined_cost_data.values())
    mean_execution_times = list(execution_time_data.values())

    print("\n=== Mean Values ===")
    for alpha in alpha_values:
        print(f"Alpha: {alpha}")
        print(f"  Mean Cost: {cost_data[alpha]:.2f}")
        print(f"  Mean Combined Cost: {combined_cost_data[alpha]:.2f}")
        print(f"  Mean Execution Time: {execution_time_data[alpha]:.2f} seconds")

    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.plot(alphas, mean_costs, marker='o', linestyle='-', color='b')
    plt.title("Coste Promedio de Transporte y Almacenamiento en función de Alpha")
    plt.xlabel("Alpha")
    plt.ylabel("Heuristica coste")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(alphas, mean_combined_costs, marker='o', linestyle='-', color='g')
    plt.title("Coste Combinado Promedio en función de Alpha")
    plt.xlabel("Alpha")
    plt.ylabel("Heuristica combinada")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(alphas, mean_execution_times, marker='o', linestyle='-', color='r')
    plt.title("Tiempo de Ejecución Promedio en función de Alpha")
    plt.xlabel("Alpha")
    plt.ylabel("Tiempo de Ejecución(s)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def experiment_7(): 
    def exp_schedule(k=0.005, lam=0.5, limit=5000):
        return lambda t: (k * math.exp(-lam * t) if t < limit else 0)

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
                initial_state = generate_initial_state(num_package=50, seed=i, proportion=1.2)
                init_time = time.time() - start_init
                total_init_time += init_time
                
                start_sim = time.time()
                result = simulated_annealing(AzamonProblem(initial_state, mode_simulated_annealing=True, combine_heuristic=True, alpha=alpha/10), schedule=exp_schedule())
                sim_time = time.time() - start_sim
                total_sim_time += sim_time
                
                cost = result.heuristic_cost()
                cost_combine = result.heuristic_cost_happy(alpha/10)
                happiness = result.heuristic_happiness()
                total_cost += cost
                total_happiness += happiness
                total_combine += cost_combine
                
                print(f"Time: {init_time + sim_time:.2f} seconds")
                print(f"Heuristic cost: {cost} | Heuristic combined: {cost_combine}")
                print()

        print("\n=== Averages over 50 tests ===")
        print(f'Alpha{alpha/10}')
        print(f"Total time: {total_init_time + total_sim_time}")
        print(f"Average execution time: {(total_init_time + total_sim_time) / 50:.2f} seconds")
        print(f"Average heuristic cost: {total_cost / 50:.2f}")
        print(f'Avarage heuristic combine:{total_combine/50:.2f}')
        print(f"Average heuristic happiness: {total_happiness / 50:.2f}")




