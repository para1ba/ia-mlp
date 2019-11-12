import numpy as np

def parse_dataset(file):
    data = np.genfromtxt(file, delimiter=',', skip_header=2)
    problem_args = list(np.genfromtxt(file, skip_footer=len(data), dtype=str))
    classes = int(problem_args[0].split(':')[1])
    dims = int(problem_args[1].split(':')[1])
    dataset = {
        'args': {
            'dims': dims,
            'classes': classes
        },
        'data': data
    }

    return dataset