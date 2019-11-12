import numpy as np

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


def get_row(dataset, index=-1):
    resp = dataset['data'][random.randrange(len(dataset['data']))] if index == -1 else dataset['data'][index]
    return (resp[:dataset['args']['dims']], resp[dataset['args']['dims']:])

def parse_NN(file):
    layers = []
    layer = {
        'number': '',
        'input': [],
        'output': [],
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