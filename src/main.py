import numpy as np
from utils import parse_dataset

dataset_path = '../res/train.csv'
model_path = '../res/model.txt'

def main():
    dataset = utils.parse_dataset(dataset_path)

    while True:
        print("======= MENU =======")
        print("\t1. Treinar Modelo")
        print("\t2. Salvar Modelo")
        print("\t3. Carregar Modelo")
        print("\t0. Sair")
        opt = input()
        break_lines()
        if opt == '1':
            pass
        elif opt == '2':
            pass
        elif opt == '3':
            pass
        elif opt == '0':
            break
        else:
            print("- OPÇÃO INVÁLIDA -")

def break_lines():
    for _ in range(20):
        print('\n')

if __name__ == '__main__':
    main()