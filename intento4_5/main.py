import experiments

def main():
    print("Seleccione el número de experimento a ejecutar:")
    print("1: Análisis de coste total en función de la proporción de peso")
    # Agrega más opciones si tienes más experimentos

    experiment_number = input("Ingrese el número del experimento: ")

    if experiment_number == "1":
        experiments.experiment_1()

    if experiment_number == "2":
        experiments.experiment_2()

    if experiment_number == "3":
        experiments.experiment_3()

    if experiment_number == "4":
        experiments.experiment_4()
    
    if experiment_number == "5":
        experiments.experiment_5()

    else:
        print("Número de experimento no válido.")

if __name__ == "__main__":
    main()
