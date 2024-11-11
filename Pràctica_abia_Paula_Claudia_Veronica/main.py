import experiments

def main():
    print("Seleccione el número de experimento a ejecutar:")

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
        
    if experiment_number == "6": 
        experiments.experiment_6()

    if experiment_number == "7": 
        experiments.experiment_7()
        
if __name__ == "__main__":
    main()
