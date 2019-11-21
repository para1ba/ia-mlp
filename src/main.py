import sys
import utils
import random
import numpy as np
from pdb import set_trace as pause

train_dataset_path = '../res/train.csv'
test_dataset_path = '../res/test.csv'
model_path = '../res/model.txt'

# n_layers = sys.argv[1]
n_layers = 5
learning_rate = 0.1

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
    global n_layers, learning_rate

    model = utils.initialize_model(n_layers)
    data_test, label_test, data_train, label_train = [], [], [], []
    test_size = 0.1
    epochs = 1

    for i in range(len(dataset['data'])):
        ## sample = (data, labels)
        sample = utils.get_row(dataset, i)
        if random.random() < test_size:
            data_test.append(sample[0])
            label_test.append(sample[1])
        else:
            data_train.append(sample[0])
            label_train.append(sample[1])

    # print("DATASET LENGTH: ", len(data_train))

    for epoch in range(epochs):
        for k in range(len(data_train)):
            # print(k)
            sample, label = data_train[k], label_train[k]
            x_values, outputs = sample, []

            outputs.append(x_values)
            for layer in model:
                x_values = utils.run_layer(layer, x_values)
                outputs.append(x_values)
                # print(x_values)

            predicted = x_values
            for i in range(len(model)-1, -1, -1):
                # print(i)

                layer = model[i]

                if i == len(model)-1:
                    gradient = np.subtract(np.dot(2, predicted), np.dot(2, label)) * predicted * (1 - np.array(predicted))
                else:
                    gradient = np.matmul(np.transpose(model[i+1]['W']), gradient) * outputs[i+1] * (1 - np.array(outputs[i+1]))

                layer = utils.update_layer(layer, gradient, learning_rate, outputs[i] if i > 0 else sample)


            # for i, layer in enumerate(reversed(model)):
            #     print(i)

            #     if layer == model[len(model) - 1]:
            #         gradient = np.subtract(np.dot(2, predicted), np.dot(2, label)) * predicted * (1 - np.array(predicted))
            #         # gradient = np.matmul(np.subtract(np.dot(2, predicted), np.dot(2, label)), predicted * (1 - np.array(predicted)))
            #     else:
            #         pause()
            #         gradient = np.matmul(np.transpose(model[len(model) - i]['W']), gradient) * outputs[len(model) - i] * (1 - np.array(outputs[len(model) - i]))
            #         # gradient = np.matmul(np.dot(np.transpose(model[len(model) - i]['W']), gradient), outputs[len(model) - i] * (1 - np.array(outputs[len(model) - i])))
            #     # pause()
            #     layer = utils.update_layer(layer, gradient, learning_rate, outputs[len(model) - (i + 1)] if len(model) - (i + 1) > 0 else sample)
    test(model, data_test, label_test)

    return model

def test(model, data_test, label_test):
    hit, miss = 0, 0
    for i, sample in enumerate(data_test):
        output = sample
        for layer in model:
            output = utils.run_layer(layer, output)        
        if np.argmax(output) == np.argmax(label_test[i]):
            hit += 1
        else:
            miss += 1
    print("ACERTOS: ", hit)
    print("ERROS: ", miss)



def save(model, model_path):
    utils.write_NN(model, model_path)

def load(model_path):
    return utils.parse_NN(model_path)

def break_lines():
    for _ in range(20):
        print('\n')

if __name__ == '__main__':
    main()