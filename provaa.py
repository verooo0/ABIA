import abia_azamon

def gen_estado_inicial(paquetes, ofertas): 
    asignaciones = {} # Cost intentar linial 
    for paquete in paquetes: 
        for oferta in ofertas:
            peso_total_asignado = sum(p.peso for p in asignaciones.get(oferta, []))
            if peso_total_asignado + paquete.peso <= oferta.pesomax:
                if oferta not in asignaciones:
                    asignaciones[oferta] = []  
                asignaciones[oferta].append(paquete)  
                break
    return asignaciones


paquetes = abia_azamon.random_paquetes(30, 1234)  
ofertas = abia_azamon.random_ofertas(paquetes, 12, 1234)  

estado_inicial = gen_estado_inicial(paquetes, ofertas)

for oferta, paquetes_asignados in estado_inicial.items():
    print(f"Oferta: {oferta}")
    for paquete in paquetes_asignados:
        print(f"  Paquete: {paquete}")

