import math
from random import Random
from typing import List

"""
MUY IMPORTANTE:
ESTE FICHERO NO DEBE SER MODIFICADO NI ENTREGADO JUNTO A LA PRÁCTICA,
ÚNICAMENTE DEBE SER IMPORTADO (import abia_azamon)
"""

"""
SECCIÓN 1
Clases y funciones para ser usadas en la práctica para
generar estados iniciales:
* Paquetes
* Ofertas de transporte (ofertas de envío)

Es OBLIGATORIO usar las funciones random_paquetes y random_ofertas para
generar vuestros experimentos.
"""


class Oferta(object):
    """
    Clase que representa una oferta de transporte con tres atributos,
    el peso máximo que se puede transportar, el precio por kilogramo
    y el número de días en los que se entregara.
    """

    def __init__(self, pesomax: float, precio: float, dias: int):
        """
        Constructora: asigna valores a una oferta de transporte
        :param pesomax: Peso máximo que se puede transportar
        :param precio: Precio por kilogramo
        :param dias: Días hasta que se haga la entrega
        """
        self.pesomax = pesomax
        self.precio = precio
        self.dias = dias

    def __str__(self):
        return f"#Oferta# pesomax ({self.pesomax}) kg" \
               f" precio({self.precio})" \
               f" dias ({self.dias})"


class Paquete(object):
    """
    Clase que representa un paquete con dos atributos,
    su peso y su prioridad
    """

    def __init__(self, peso: float, prioridad: int):
        """
        Constructora: genera un paquete con un peso y una prioridad
        :param peso: Peso de un paquete
        :param prioridad: Prioridad de un paquete
        (valor 0 ⇾ Paquetes de prioridad 1 = entrega en un día,
         valor 1 ⇾ Paquetes de prioridad 2 = entrega entre 2 y 3 días,
         valor 2 ⇾ Paquetes de prioridad 3 = entrega entre 4 y 5 días)
        """
        self.peso = peso
        self.prioridad = prioridad

    def __str__(self):
        return f"#Paquete# peso({self.peso})" \
               f" prioridad({self.prioridad})"


def random_paquetes(npaq: int, seed: int) -> List[Paquete]:
    """
    Función que genera la estructura de paquetes, de manera aleatoria,
    siguiendo cierta distribución binomial sobre los pesos y prioridades de los paquetes
    :param npaq: Número de paquetes a generar
    :param seed: Semilla para el generador de números aleatorios
    :return: Estructura de paquetes de tamaño npaq
    """
    rng = Random(seed)
    list_paquetes: List[Paquete] = []
    for _ in range(npaq):
        rand_peso = rng.randint(0, 5)
        if rand_peso < 3:
            peso = (rng.randint(0, 6) + 1) * 0.5
        elif 3 <= rand_peso < 5:
            peso = (rng.randint(0, 6) + 1) * 0.5 + 3.5
        else:
            peso = (rng.randint(0, 6) + 1) * 0.5 + 7.0
        rand_prioridad = rng.randint(0, 3)
        prioridad = 1
        if rand_prioridad == 0:
            prioridad = 0
        elif rand_prioridad == 3:
            prioridad = 2
        list_paquetes.append(Paquete(peso, prioridad))
    return list_paquetes


def random_ofertas(list_paquetes: List[Paquete],
                   proporcion: float,
                   seed: int) \
        -> List[Oferta]:
    """
    Función que genera un conjunto de ofertas de transporte,
    de manera aleatoria, que permitan transportar todos los paquetes
    que hay en la estructura de paquetes enviada.
    El algoritmo de generación asegura que hay al menos en conjunto
    entre las ofertas una capacidad de transporte de peso indicada
    por el parametro proporcion.
    :param list_paquetes: Estructura de los paquetes a enviar
    :param proporcion: Proporción respecto al peso a utilizar por la
                       generación de ofertas
    :param seed: Semilla del generador de números aleatorios
    :return: Estructura de ofertas de transporte (u ofertas de envíos)
    """

    def truncate(valor: float) -> float:
        # Función auxiliar para generar precios aleatorios
        return math.floor(valor * 100.0) / 100.0

    precios = [[3.0, 1.5], [2.0, 0.9], [1.5, 0.7], [0.7, 0.7], [0.2, 0.7]]

    dist_peso_por_prioridad: List[float] = [0.0, 0.0, 0.0]
    dist_peso_por_dia: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0]
    rng: Random = Random(seed)

    list_ofertas: List[Oferta] = []

    for paquete in list_paquetes:
        prioridad = paquete.prioridad
        dist_peso_por_prioridad[prioridad] += paquete.peso

    dist_peso_por_dia[0] = dist_peso_por_prioridad[0] * proporcion
    dist_peso_por_dia[1] = dist_peso_por_prioridad[1] * proporcion / 2.0
    dist_peso_por_dia[2] = dist_peso_por_prioridad[1] * proporcion / 2.0
    dist_peso_por_dia[3] = dist_peso_por_prioridad[2] * proporcion / 2.0
    dist_peso_por_dia[4] = dist_peso_por_prioridad[2] * proporcion / 2.0

    for dias in range(5):
        peso_acumulado = 0.0
        while peso_acumulado < dist_peso_por_dia[dias]:
            pesomax = (rng.randint(0, 8) + 1) * 5.0
            precio = truncate(rng.random() * precios[dias][1] + precios[dias][0])
            peso_acumulado += pesomax
            oferta = Oferta(pesomax, precio, dias + 1)
            list_ofertas.append(oferta)
    return list_ofertas


"""
SECCIÓN 2
Funciones para probar las clases y funciones de la Sección 1.
Podéis ejecutar el main() para comprobar cómo funcionan, y
revisar el código para haceros una idea de cómo usarlas en
vuestra práctica. No modifiquéis este fichero, pero sí podéis
copiar código de esta Sección 2.
"""


def inspeccionar_paquetes(l_paquetes):
    # Dada una lista de paquetes, obtener información de
    # cada uno: peso y prioridad.
    peso_por_prioridad = [0.0, 0.0, 0.0]
    paqs_por_prioridad = [0, 0, 0]

    print(" -------- Paquetes  ------------")
    for paquete in l_paquetes:
        peso_por_prioridad[paquete.prioridad] += paquete.peso
        paqs_por_prioridad[paquete.prioridad] += 1
    for prioridad in range(3):
        for paquete in l_paquetes:
            if paquete.prioridad == prioridad:
                print(paquete)
    print("\n")
    for prioridad in range(3):
        print(f"Prioridad {prioridad}"
              f" N paq={paqs_por_prioridad[prioridad]}"
              f" Peso total= {peso_por_prioridad[prioridad]}")


def inspeccionar_ofertas(l_ofertas):
    # Dada una lista de ofertas, extraer información potencialmente
    # interesante: número de ofertas y peso máximo, precio y días de
    # cada oferta.
    ofertas_por_prioridad = [0, 0, 0, 0, 0]
    pesomax_por_prioridad = [0.0, 0.0, 0.0, 0.0, 0.0]

    print("\n -------- Ofertas  ------------")
    print(f"num ofertas = {len(l_ofertas)}\n")
    for oferta in l_ofertas:
        print(oferta)
        ofertas_por_prioridad[oferta.dias - 1] += 1
        dia = oferta.dias - 1
        pesomax_por_prioridad[dia] += oferta.pesomax
    print("\n")
    for dia in range(5):
        print(f"Dia {dia + 1} N ofertas={ofertas_por_prioridad[dia]}"
              f" Peso maximo= {pesomax_por_prioridad[dia]}")
    print()


def crear_asignacion_suboptima(l_paquetes, l_ofertas):
    # Función para crear una asignación de paquetes a ofertas
    # de manera aleatoria, de una manera no guiada ni eficiente.
    # ATENCIÓN: No es una buena base para crear una estrategia
    # de solución inicial (principalmente, por la aleatoriedad).
    def asignable(paquete, oferta):
        return not ((paquete.prioridad != 0 or oferta.dias != 1)
                    and (paquete.prioridad != 1 or oferta.dias != 2)
                    and (paquete.prioridad != 1 or oferta.dias != 3)
                    and (paquete.prioridad != 2 or oferta.dias != 4)
                    and (paquete.prioridad != 2 or oferta.dias != 5))

    oferta_por_paquete = [0] * len(l_paquetes)
    peso_por_oferta = [0.0] * len(l_ofertas)
    copia_ofertas = []

    for id_oferta in range(len(l_ofertas)):
        copia_ofertas.append(id_oferta)

    # Bucle para asignar una oferta a cada paquete
    rng_asig = Random(2)  # Extraemos los paquetes aleatoriamente
    for id_paquete in range(len(l_paquetes)):
        paquete_asignado = False
        while not paquete_asignado:
            id_oferta_potencial = rng_asig.randint(0, len(copia_ofertas) - 1)
            oferta_potencial = copia_ofertas[id_oferta_potencial]
            while not asignable(l_paquetes[id_paquete], l_ofertas[oferta_potencial]):
                id_oferta_potencial = rng_asig.randint(0, len(copia_ofertas) - 1)
                oferta_potencial = copia_ofertas[id_oferta_potencial]
            if l_paquetes[id_paquete].peso + peso_por_oferta[oferta_potencial] \
                    <= l_ofertas[oferta_potencial].pesomax:
                peso_por_oferta[oferta_potencial] = peso_por_oferta[oferta_potencial] \
                                                    + l_paquetes[id_paquete].peso
                oferta_por_paquete[id_paquete] = oferta_potencial
                paquete_asignado = True
                print(f"Paq= {id_paquete} Env={oferta_potencial}")
            else:
                copia_ofertas.__delitem__(id_oferta_potencial)
    print()
    for id_paquete in range(len(l_paquetes)):
        print(f"Paq= {id_paquete} Env={oferta_por_paquete[id_paquete]}"
              f" P={l_paquetes[id_paquete].prioridad}"
              f" D={l_ofertas[oferta_por_paquete[id_paquete]].dias}")
    for id_oferta in range(len(l_ofertas)):
        print(f"Env= {id_oferta}"
              f" Weight={peso_por_oferta[id_oferta]}"
              f" MXweight={l_ofertas[id_oferta].pesomax}")
        if l_ofertas[id_oferta].pesomax < peso_por_oferta[id_oferta]:
            print("Esta situación no se debería dar. ¡Reportadlo!")
            raise RuntimeError


if __name__ == '__main__':
    npaq = int(input("Numero de paquetes: "))
    semilla = int(input("Semilla aleatoria: "))
    paquetes = random_paquetes(npaq, semilla)
    ofertas = random_ofertas(paquetes, 1.2, 1234)

    inspeccionar_paquetes(paquetes)
    inspeccionar_ofertas(ofertas)
    crear_asignacion_suboptima(paquetes, ofertas)

