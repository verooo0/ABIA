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
    # Inicializar `v_p` para almacenar el índice de la oferta asignada para cada paquete
    v_p = []
    prioridad_a_dias_max = {0: 1, 1: 3, 2: 5}
#     # Asignar paquetes a ofertas basándose en la prioridad
#     for prioridad in range(3):
#         for paquete in paquetes:
#             if paquete.prioridad == prioridad:
#                 for idx, oferta in enumerate(ofertas):
#                     # Calcular el peso total si añadimos este paquete a la oferta actual
#                     peso_total_asignado = sum(p['peso'] for p in asignaciones.get(tuple(oferta.__dict__.items()), []))

#                     # Verificar si la oferta tiene suficiente capacidad para este paquete
#                     if peso_total_asignado + paquete.peso <= oferta.pesomax:
#                         # Actualizar `asignaciones` con el paquete para esta oferta
#                         if tuple(oferta.__dict__.items()) not in asignaciones:
#                             asignaciones[tuple(oferta.__dict__.items())] = []
#                         asignaciones[tuple(oferta.__dict__.items())].append({'peso': paquete.peso, 'prioridad': paquete.prioridad})

#                         # Registrar en `v_p` el índice de la oferta asignada para este paquete
#                         v_p.append(idx)
#                         break  # Sale del bucle de ofertas al asignar el paquete

#                 # Si el paquete no fue asignado a ninguna oferta (por cualquier motivo), asignarlo a `v_p` con un marcador
#                 if len(v_p) < len(paquetes):
#                     v_p.append(0)  # Opcional: puedes asignar un marcador diferente si no se asignó ninguna oferta

# # Crear la representación de estado utilizando `v_p` sincronizado con `asignaciones`


#     paquetes_retrasados = []  # Lista para registrar paquetes con retraso

#     for idx, paquete in enumerate(paquetes):
#         oferta_idx = v_p[idx]
        
#         # Saltar paquetes no asignados
#         if oferta_idx is None:
#             continue
        
#         oferta = ofertas[oferta_idx]
#         dias_limite = oferta.dias  # Días de entrega de la oferta
#         dias_max_permitidos = prioridad_a_dias_max[paquete.prioridad]  # Días máximos según la prioridad del paquete
        
#         # Verificar si el paquete excede el tiempo de entrega permitido
#         if dias_limite > dias_max_permitidos:
#             paquetes_retrasados.append((paquete, oferta, dias_limite, dias_max_permitidos))

#     # Opcional: mostrar paquetes que se envían con retraso
#     if paquetes_retrasados:
#         print("Los siguientes paquetes no cumplen con su tiempo de entrega máximo:")
#         for paquete, oferta, dias_limite, dias_max_permitidos in paquetes_retrasados:
#             print(f"- Paquete con peso {paquete.peso} y prioridad {paquete.prioridad} asignado a oferta con entrega de {dias_limite} días (máximo permitido: {dias_max_permitidos} días)")
        
 #   ------------------------------------------------Original----------------------------------------------
    # # Asignar paquetes a ofertas basándose en la prioridad
    # for prioridad in range(3):
    #     for paquete in paquetes:
    #         if paquete.prioridad == prioridad:
    #             for oferta in ofertas:
    #                 peso_total_asignado = sum(peso['peso'] for peso in asignaciones.get(tuple(oferta.__dict__.items()), []))
    #                 if peso_total_asignado + paquete.peso <= oferta.pesomax:
    #                     if tuple(oferta.__dict__.items()) not in asignaciones:
    #                         asignaciones[tuple(oferta.__dict__.items())] = []
    #                     asignaciones[tuple(oferta.__dict__.items())].append({'peso': paquete.peso, 'prioridad': paquete.prioridad})
    #                     break

    # # Convertir asignaciones a representación de estado
    # # Creamos un vector `v_p` que indica la oferta asignada a cada paquete
    # v_p = []
    # for paquete in paquetes:
    #     for idx, oferta in enumerate(ofertas):
    #         # Busca la oferta asignada para este paquete
    #         if {'peso': paquete.peso, 'prioridad': paquete.prioridad} in asignaciones.get(tuple(oferta.__dict__.items()), []):
    #             v_p.append(idx)
    #             break
    #     else:
    #         v_p.append(0)  # Asigna a la primera oferta si no se asignó
#-----------------------------------------------------------------------------------------


    # Asignar paquetes a ofertas basándose en la prioridad
    for prioridad in range(3):
        for paquete in paquetes:
            if paquete.prioridad == prioridad:
                asignado = False  # Indicador para verificar si el paquete ha sido asignado

                for idx, oferta in enumerate(ofertas):
                    # Calcular el peso total si añadimos este paquete a la oferta actual
                    peso_total_asignado = sum(p['peso'] for p in asignaciones.get(tuple(oferta.__dict__.items()), []))

                    # Verificar si la oferta tiene suficiente capacidad para este paquete
                    if peso_total_asignado + paquete.peso <= oferta.pesomax:
                        v_p.append(idx)
                        # Actualizar `asignaciones` con el paquete para esta oferta
                        if tuple(oferta.__dict__.items()) not in asignaciones:
                            asignaciones[tuple(oferta.__dict__.items())] = []
                        asignaciones[tuple(oferta.__dict__.items())].append({'peso': paquete.peso, 'prioridad': paquete.prioridad})

                        # Registrar en `v_p` el índice de la oferta asignada para este paquete
                        
                        asignado = True  # Marca el paquete como asignado
                        break # Puedes cambiar 0 por otro marcador si es necesario

                if not asignado:
                    v_p.append(None) 
# Crear la representación de estado utilizando `v_p` sincronizado con `asignaciones`
    #print(asignaciones)
    #print(v_p)

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
#----------------Original-----------------------------------------------------------   
    # Asignar paquetes a ofertas basándose
    # for paquete in paquetes: 
    #     for oferta in ofertas:
    #         peso_total_asignado = sum(peso['peso'] for peso in asignaciones.get(tuple(oferta.__dict__.items()), []))
    #         if peso_total_asignado + paquete.peso <= oferta.pesomax:
    #             if tuple(oferta.__dict__.items()) not in asignaciones:
    #                 asignaciones[tuple(oferta.__dict__.items())] = []  
    #             asignaciones[tuple(oferta.__dict__.items())].append({'peso': paquete.peso, 'prioridad': paquete.prioridad})  
    #             break

#--------------Original mas o menos-----------------------------------------------------------------------
    # paquetes_no_asignados = []  # Lista para registrar paquetes sin asignación

    # for paquete in paquetes: 
    #     asignado = False  # Indicador para verificar si el paquete se ha asignado
    #     for oferta in ofertas:
    #         peso_total_asignado = sum(peso['peso'] for peso in asignaciones.get(tuple(oferta.__dict__.items()), []))
    #         if peso_total_asignado + paquete.peso <= oferta.pesomax:
    #             if tuple(oferta.__dict__.items()) not in asignaciones:
    #                 asignaciones[tuple(oferta.__dict__.items())] = []  
    #             asignaciones[tuple(oferta.__dict__.items())].append({'peso': paquete.peso, 'prioridad': paquete.prioridad})  
    #             asignado = True  # Marca el paquete como asignado
    #             break  # Sale del bucle ya que el paquete se asignó

    #     # Si el paquete no se asignó, añadirlo a la lista de no asignados
    #     if not asignado:
    #         paquetes_no_asignados.append(paquete)

    
    # print(paquetes_no_asignados)
    # print(asignaciones)

    # for oferta in ofertas:
    #     clave_oferta = tuple(oferta.__dict__.items())
    #     if clave_oferta in asignaciones:
    #         # Calcula el peso total de todos los paquetes asignados a esta oferta
    #         peso_total = sum(paquete['peso'] for paquete in asignaciones[clave_oferta])
            
    #         # Comprueba si el peso total excede la capacidad máxima de la oferta
    #         if peso_total > oferta.pesomax:
    #             print(f"Advertencia: La oferta con capacidad máxima {oferta.pesomax} ha sido excedida con un peso total de {peso_total}.")

    
#--------------------------Original----------------------------------------------------------------------
    # # Convertir asignaciones a representación de estado
    # # Creamos un vector `v_p` que indica la oferta asignada a cada paquete
    # v_p = []
    # for paquete in paquetes:
    #     for idx, oferta in enumerate(ofertas):
    #         # Busca la oferta asignada para este paquete
    #         if {'peso': paquete.peso, 'prioridad': paquete.prioridad} in asignaciones.get(tuple(oferta.__dict__.items()), []):
    #             v_p.append(idx)
    #             break
    #     else:      
    #         v_p.append(0)  # Asigna a la primera oferta si no se asignó
#--------------------------------------------------------------------------------------------------------
    # for idx, oferta in enumerate(ofertas):
    # # Calcular el peso total para la oferta actual usando `v_p`
    #     peso_total = sum(paquetes[i].peso for i, oferta_idx in enumerate(v_p) if oferta_idx == idx)
        
    #     if peso_total > oferta.pesomax:
    #         print(f"Advertencia: La oferta {idx} ha excedido su capacidad con un peso total de {peso_total} (máximo permitido: {oferta.pesomax}).")
#------------------------------------------------------------------------------------------------------------------------
    # Creamos un vector `v_p` que indica la oferta asignada a cada paquete
    v_p = []
    paquetes_no_asignados = []  # Lista para registrar paquetes sin asignación

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
            v_p.append(None)  # Opcional: añadir un marcador para paquetes no asignados
            paquetes_no_asignados.append(paquete)

    # Paso 2: Verificación final en `v_p` usando `asignaciones` para asegurar consistencia
    for idx, oferta in enumerate(ofertas):
        # Calcula el peso total para la oferta actual usando `v_p`
        peso_total = sum(paquetes[i].peso for i, oferta_idx in enumerate(v_p) if oferta_idx == idx)
        
        # Verifica si el peso total excede la capacidad máxima de la oferta
        if peso_total > oferta.pesomax:
            print(f"Advertencia: La oferta {idx} ha excedido su capacidad con un peso total de {peso_total} (máximo permitido: {oferta.pesomax}).")

    # print(asignaciones)
    print(v_p)


    # v_p = []
    # for paquete in paquetes:
    #     asignado = False  # Indicador de asignación para cada paquete
    #     for idx, oferta in enumerate(ofertas):
    #         # Calcula el peso total si añadimos este paquete a la oferta actual
    #         peso_total_asignado = sum(p['peso'] for p in asignaciones.get(tuple(oferta.__dict__.items()), []))
            
    #         # Verifica si la oferta tiene suficiente capacidad para este paquete
    #         if peso_total_asignado + paquete.peso <= oferta.pesomax:
    #             # Asigna el paquete a la oferta actual
    #             if tuple(oferta.__dict__.items()) not in asignaciones:
    #                 asignaciones[tuple(oferta.__dict__.items())] = []
    #             asignaciones[tuple(oferta.__dict__.items())].append({'peso': paquete.peso, 'prioridad': paquete.prioridad})
    #             v_p.append(idx)  # Guarda el índice de la oferta asignada
    #             asignado = True
    #             break 
    
    return StateRepresentation(params, v_p)

# Pruebas aleatorias con asignación por prioridad

total_init_time = 0
total_hill_time = 0
total_cost = 0
total_happiness = 0
max_cost = []
max_happines = []

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
    result = hill_climbing(AzamonProblem(initial_state ,use_entropy=False, combine_heuristic=True))
    hill_time = time.time() - start_hill
    total_hill_time += hill_time
    print(f"Hill climbing time: {hill_time:.2f} seconds")

    # Get heuristic values
    cost = result.heuristic_cost()
    max_cost.append(result.heuristic_cost())
    max_happines.append(result.heuristic_happiness())
    cost_happy = result.heuristic_cost_happy()
    happiness = result.heuristic_happiness()
    total_cost += cost
    total_happiness += happiness

    print(f"Time: {init_time + hill_time:.2f} seconds")
    print(f" Heuristic cost: {result.heuristic_cost()} | Heuristic happiness: {result.heuristic_happiness()} | Assignments: {result.last_assigments()}")
    print()

# Print averages at the end
print("\n=== Average Times ===")
print(f"Total execution time: {total_init_time + total_hill_time:.2f} seconds")
print(f"Average execution time: {(total_init_time + total_hill_time) / 10:.2f} seconds")
print(f"Average heuristic cost: {total_cost / 10:.2f}")
print(f"Average heuristic happiness: {total_happiness / 10:.2f}")
print(f'Max cost :{max(max_cost)}') #618.885
print(f'Max happiness :{max(max_happines)}') #56
