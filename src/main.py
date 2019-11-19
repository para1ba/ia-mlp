import sys
import utils
import random
import numpy as np

train_dataset_path = '../res/train.csv'
test_dataset_path = '../res/test.csv'
model_path = '../res/model.txt'

# n_layers = sys.argv[1]
n_layers = 5

def main():
    global model_path, train_dataset_path, test_dataset_path
    
    model = {}
    train_dataset = utils.parse_dataset(train_dataset_path)
    #test_dataset = utils.parse_dataset(test_dataset_path)

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
    data_test, label_test, data_train, label_train = [], [], [], []
    test_size = 0.1

    for i in range(len(dataset['data'])):
        data, label = utils.get_row(dataset, i)
        if random.random() < test_size:
            data_test.append(data)
            label_test.append(label)
        else:
            data_train.append(data)
            label_train.append(label) 
    print("TAMANHO DO DATASET DE TREINO: ", len(data_train))
    print("TAMANHO DO DATASET DE TESTE: ", len(data_test))

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