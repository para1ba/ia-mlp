import numpy as np
from pdb import set_trace as pause

neurons = 20

def parse_dataset(file):
    data = np.genfromtxt(file, delimiter=',', skip_header=2)
    problem_args = list(np.genfromtxt(file, skip_footer=len(data), dtype=str))
    dims = int(problem_args[0].split(':')[1])
    classes = int(problem_args[1].split(':')[1])
    dataset = {
        'args': {
            'dims': dims,
            'classes': classes
        },
        'data': data
    }

    return dataset

def run_layer(layer, x_values):
    y = np.add(np.matmul(layer['W'], x_values.transpose()), layer['b'].transpose())
    return apply_signal(y, layer['signal'])

def update_layer(layer, gradient, learning_rate, h_before):
    layer['W'] = np.subtract(layer['W'], np.dot(learning_rate, (np.array(gradient)*np.transpose(np.array(h_before)))))
    layer['b'] = np.subtract(layer['b'], np.dot(learning_rate, gradient))

    return layer

def apply_signal(arr, signal_function="sigmoid"):
    for i in range(len(arr)):
        arr[i] = apply_function(arr[i], signal_function)
    return arr

def apply_function(value, function):
    if function == 'sigmoid':
        return sigmoid(value)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def get_row(dataset, index=-1):
    resp = dataset['data'][random.randrange(len(dataset['data']))] if index == -1 else dataset['data'][index]
    return (resp[:dataset['args']['dims']], resp[dataset['args']['dims']:])

def initialize_model(n_layers = 5):
    global neurons
    
    model = []
    for i in range(n_layers):
        input_length = neurons if i > 0 else 784
        output_length = neurons if i < n_layers-1 else 10

        b = np.random.uniform(low = -1, high = 1, size=(output_length, ))
        W = np.random.uniform(low = -1, high = 1, size=(output_length, input_length))

        layer = {
            'number': i+1,
            'input': input_length,
            'output': output_length,
            'signal': 'sigmoid',
            'b': b,
            'W': W
        }
        model.append(layer)
    return model

def parse_NN(file):
    layers = []
    layer = {
        'number': '',
        'input': 0,
        'output': 0,
        'signal': '',
        'b': [],
        'W': []
    }
    in_W = False
    in_b = False
    arr = []
    for line in open(file, 'r'):
        line = line.rstrip()
        if 'W' in line:
            in_W = True
        elif 'b' in line:
            layer['W'] = np.array(arr)
            arr = []
            in_W = False
            in_b = True
        elif 'entrada' in line:
            words = line.split()
            layer['input'] = int(words[-1])
        elif 'saida' in line:
            words = line.split()
            layer['output'] = int(words[-1])
        elif 'ativacao' in line:
            in_b = False
            words = line.split()
            layer['signal'] = words[-1]
        elif 'camada' in line:
            layer['number'] = int(line[-1])
        elif line == '--':
            layers.append(layer)
            layer = {}
        elif in_W:
            values = line.split()
            for i in range(len(values)):
                values[i] = float(values[i])
            arr.append(values)
        elif in_b:
            values = line.split()
            for i in range(len(values)):
                values[i] = float(values[i])
            layer['b'] = np.array(values)
    return layers

def write_NN(model, filepath):
    with open(filepath, 'w+') as file:
        for layer in model:
            file.write("camada_" + str(layer['number']) + "\n")
            file.write("entrada " + str(layer['input']) + "\n")
            file.write("saida  " + str(layer['output']) + "\n")
            file.write("W" + "\n")
            for lines in layer['W']:
                file.write(" ".join(str(x) for x in lines) + '\n')
            file.write("b" + "\n")
            file.write(" ".join(str(x) for x in layer['b']) + '\n')
            file.write("ativacao " + str(layer['signal']) + "\n")
            file.write("--\n")