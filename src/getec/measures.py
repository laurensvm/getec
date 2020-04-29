import numpy as np

def get_accuracy_measure(preds, y):
    _sum = 0
    for idx, pred in enumerate(preds):
        if np.argmax(pred) == y[idx]:
            _sum += 1

    acc = _sum / len(y)

    return acc