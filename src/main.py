import numpy as np
import utils

train_dataset_path = '../res/train.csv'
test_dataset_path = '../res/test.csv'
model_path = '../res/model.txt'

def main():
    global model_path, train_dataset_path, test_dataset_path
    
    model = {}
    train_dataset = utils.parse_dataset(train_dataset_path)
    test_dataset = utils.parse_dataset(test_dataset_path)

    while True:
        print("======= MENU =======")
        print("\t1. Treinar Modelo")
        print("\t2. Salvar Modelo")
        print("\t3. Carregar Modelo")
        print("\t4. Testar Modelo")
        print("\t0. Sair")
        opt = input()
        break_lines()
        if opt == '1':
            model = train(train_dataset)
        elif opt == '2':
            if not model:
                print("> Você precisa ter um modelo treinado para isso!")
            else:
                save(model, model_path)
        elif opt == '3':
            model = load(model_path)
        elif opt == '4':
            if not model:
                print("> Você precisa ter um modelo treinado para isso!")
            else:
                test(model, test_dataset)
        elif opt == '0':
            break
        else:
            print("- OPÇÃO INVÁLIDA -")

def train(dataset):
    model = None
    return model

def test(model, dataset):
    pass

def save(model, model_path):
    utils.write_NN(model, model_path)

def load(model_path):
    return utils.parse_NN(model_path)

def break_lines():
    for _ in range(20):
        print('\n')

if __name__ == '__main__':
    main()